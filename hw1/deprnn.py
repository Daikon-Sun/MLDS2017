from __future__ import division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import sys
import numpy as np
import tensorflow as tf
from collections import Counter

#default values
default_wordvec_src = -1
default_vocab_size = 11
default_hidden_size = 100
default_layer_num = 1
default_rnn_type = 0
default_use_dep = True
default_learning_rate = 0.1
default_max_grad_norm = 100
default_max_epoch = 1000
default_keep_prob = 1.0
default_batch_size = 2
default_data_mode = 2
default_data_dir = './toy_data/'

#functions for arguments of unsupported types
def t_or_f(arg):
  ua = str(arg).upper()
  if 'TRUE'.startswith(ua):
    return True
  elif 'FALSE'.startswith(ua):
    return False
  else:
    raise argparse.ArgumentTypeError('--use_dep can only be True or False!')
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('%r not in range [0.0, 1.0]'%x)
    return x

#argument parser
parser = argparse.ArgumentParser(description=\
    'Dependency-tree-based rnn for MSR sentence completion challenge.')
parser.add_argument('--wordvec_src', type=int, default=default_wordvec_src, nargs='?', \
    choices=range(-1, 7), \
    help='Decide the source of wordvec --> [-1:debug-mode], \
    [0:one-hot], [1:glove.6B.50d], [2:glove.6B.100d], \
    [3:glove.6B.200d], [4:glove.6B.300d], [5:glove.42B], \
    [6:glove.840B]. (default:%d)'%default_wordvec_src)
parser.add_argument('--vocab_size', type=int, default=default_vocab_size, nargs='?', \
    help='The vocabulary size to be trained. (default:%d)'%default_vocab_size)
parser.add_argument('--layer_num', type=int, default=default_layer_num, nargs='?', \
    help='Number of rnn layer.. (default:%d)'%default_layer_num)
parser.add_argument('--rnn_type', type=int, default=default_rnn_type, nargs='?', \
    choices=range(0, 3), \
    help='Type of rnn cell --> [0:Basic], [1:LSTM], [2:GRU]. (default:%d)'\
    %default_rnn_type)
parser.add_argument('--learning_rate', type=float, default=default_learning_rate, \
    nargs='?', help='Value of initial learning rate. (default:%r)'\
    %default_learning_rate)
parser.add_argument('--max_grad_norm', type=float, default=default_max_grad_norm, \
    nargs='?', help='Maximum gradient norm allowed. (default:%r)'\
    %default_max_grad_norm)
parser.add_argument('--use_dep', type=t_or_f, default=default_use_dep, nargs='?', \
    choices=[False, True], \
    help='Use dependency tree or not. (default:%r)'%default_use_dep)
parser.add_argument('--max_epoch', type=int, default=default_max_epoch, nargs='?', \
    help='Maximum epoch to be trained. (default:%d)'%default_max_epoch)
parser.add_argument('--keep_prob', type=restricted_float, \
    default=default_keep_prob, \
    nargs='?', help='Keeping-Probability for dropout layer. (default:%r)'\
    %default_keep_prob)
parser.add_argument('--batch_size', type=int, default=default_batch_size, nargs='?', \
    help='Mini-batch size while training. (default:%d)'%default_batch_size)
parser.add_argument('--hidden_size', type=int, default=default_hidden_size, nargs='?', \
    help='Dimension of hidden layer. (default:%d)'%default_hidden_size)
parser.add_argument('--data_mode', type=int, default=default_data_mode, nargs='?', \
    choices=range(1, 3), \
    help='Data mode for preprocessed data --> [1:one file], [2:two files].\
    (default:%d)'%default_data_mode)
parser.add_argument('--data_dir', type=str, default=default_data_dir, nargs='?', \
    help='Directory where the data are placed. (default:%s)'%default_data_dir)

args = parser.parse_args()

#decide embedding dimension and vocabulary size
if args.wordvec_src == -1:
  args.embed_dim = 50
  args.vocab_size = 11
elif args.wordvec_src == 0:
  args.embed_dim = args.vocab_size
elif args.wordvec_src == 1:
  args.embed_dim = 50
  args.vocab_size = 400000
elif args.wordvec_src == 2:
  args.embed_dim = 100
  args.vocab_size = 400000
elif args.wordvec_src == 3:
  args.embed_dim = 200
  args.vocab_size = 400000
elif args.wordvec_src == 4:
  args.embed_dim = 300
  args.vocab_size = 400000
elif args.wordvec_src == 5:
  args.embed_dim = 300
  args.vocab_size = 1917494
elif args.wordvec_src == 6:
  args.embed_dim = 300
  args.vocab_size = 2196017
else: assert(False)

class data_manager(object):

  def __init__(self, args):
    self._batch_id = 0
    self._para = args

    with open(self._para.data_dir+'wordlist.txt', 'r') as f:
      words = f.read().splitlines()
    word_to_id = dict(zip(words, range(len(words))))
    w_e = np.load(self._para.data_dir+'wordvec.npy', 'r')
    with open(self._para.data_dir+'train_data.txt', 'r') as f:
      raw_data = f.read().splitlines()

    self._sent_num = len(raw_data)
    raw_data = [ sent.split() for sent in raw_data ]

    #id of <unk> = 0
    self._raw_data = [ [word_to_id[word] if word in word_to_id else 0 for word in sent]\
        for sent in raw_data]

  def get_batch(self, name=None):
    batch_data = self._raw_data[self._batch_id:self._batch_id+self._para.batch_size]
    self._batch_id += 1
    if self._batch_id == self._sent_num:
      self._batch_id = 0

    with tf.name_scope(name, "batch", [batch_data, self._para, self._batch_id]):

      #get seq_len for dynamic_rnn and max_seq_len for zero-padding
      seq_len = [ len(sent) for sent in batch_data ]
      max_len = max(seq_len)

      #zero-padding
      for rw in batch_data: rw.extend([0]*(max_len-len(rw)))

      #reshape to size: batch_size * max_seq_len
      batch = np.array(batch_data).reshape((self._para.batch_size, max_len))

      batch_x = batch[:, :-1]
      x = tf.stack([tf.convert_to_tensor(bx, dtype=tf.int32) for bx in batch_x], name='x')
      batch_y = batch[:, 1:]
      y = tf.stack([tf.convert_to_tensor(by, dtype=tf.int32) for by in batch_y], name='y')
      #print('x = %r\n' % x)
      #print('y = %r\n' % y)
      return x, y, seq_len

class DepRNN(object):
  """The deprnn language model"""

  def __init__(self, is_training, para, inputs):
    self._para = para
    self._inputs = inputs

    #build model
    if self._para.rnn_type == 0:#basic rnn
      def unit_cell():
        return tf.contrib.rnn.BasicRNNCell(self._para.hidden_size, activation=tf.tanh)
    else:#TODO (LSTM and GRU)
      assert(False)
    rnn_cell = unit_cell
    if is_training and self._para.keep_prob < 1:#TODO (dropout layer)
      assert(False)

    cell = tf.contrib.rnn.MultiRNNCell([rnn_cell()] * self._para.layer_num, \
        state_is_tuple=True)
    self._initial_state = cell.zero_state(self._para.batch_size, tf.float32)

    #load in pre-trained word embedding
    wordvec = np.load(self._para.data_dir+'wordvec.npy')
    W_E = tf.Variable(tf.constant(0.0, \
        shape=[self._para.vocab_size, self._para.embed_dim]), \
        trainable=False, name='W_E')
    embedding = tf.placeholder(tf.float32, \
        [self._para.vocab_size, self._para.embed_dim])
    #print('embedding = %r\n' % embedding)
    embed_init = W_E.assign(embedding)
    #print('W_E = %r\n' % W_E)
    x, y, seq_len = self._inputs.get_batch()
    inputs = tf.nn.embedding_lookup(W_E, x)
    #print('inputs = %r\n' % inputs)


    if is_training and self._para.keep_prob < 1:#TODO (dropout layer on input)
      assert(False)

    #use dynamic_rnn to build dynamic time-step rnn
    #print('inputs = %r\n' % inputs)
    #print('seq_len = %r\n' % seq_len)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_len, \
        dtype=tf.float32)
    #print('outputs = %r\n' % outputs)
    output = tf.reshape(tf.concat(outputs, 1), [-1, self._para.hidden_size])
    #print('output = %r\n' % output)
    with tf.variable_scope('softmax'):
      softmax_w = tf.get_variable('w', [self._para.hidden_size, self._para.vocab_size], \
          dtype=tf.float32)
      softmax_b = tf.get_variable('b', [self._para.vocab_size], dtype=tf.float32)
    #print('softmax_w = %r\n' % softmax_w)
    #print('softmax_b = %r\n' % softmax_b)
    logits = tf.matmul(output, softmax_w)+softmax_b
    #print('logits = %r\n' % logits)
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], \
        [tf.reshape(y, [-1])], \
        [tf.ones([self._para.batch_size*(max(seq_len)-1)], dtype=tf.float32)])
    self._cost = tf.reduce_sum(loss) / self._para.batch_size
    self._final_state = state
    #print('outputs = %r\n' % outputs)
    #print('state = %r\n' % state)

    #if validation or testing, exit here
    if not is_training: return

    optimizer = tf.train.GradientDescentOptimizer(self._para.learning_rate)

    sys.exit()

  def tmp_init(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs,state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
#with tf.Graph().as_default():
  #sv = tf.train.Supervisor(logdir='./logs')
  #with sv.managed_session() as sess:
trn_data = data_manager(args)
#print(reader(trn_data[:args.batch_size], 0, args))
DepRNN(is_training=True, para=args, inputs=trn_data)
