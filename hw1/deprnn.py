import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
import numpy as np
import tensorflow as tf
from collections import Counter

#default values
default_wordvec_src = 1
default_hidden_size = 128
default_layer_num = 2
default_rnn_type = 1
default_use_dep = False
default_learning_rate = 0.001
default_init_scale = 0.001
default_max_grad_norm = 200
default_max_epoch = 300
default_keep_prob = 0.5
default_batch_size = 128
default_data_dir = './Training_Data_50d/'

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
parser.add_argument('--wordvec_src', type=int, default=default_wordvec_src, nargs='?',\
    choices=range(0, 7),\
    help='Decide the source of wordvec --> [0:debug-mode], [1:glove.6B.50d],\
    [2:glove.6B.100d], [3:glove.6B.200d], [4:glove.6B.300d], [5:glove.42B],\
    [6:glove.840B]. (default:%d)'%default_wordvec_src)
parser.add_argument('--layer_num', type=int, default=default_layer_num, nargs='?',\
    help='Number of rnn layer. (default:%d)'%default_layer_num)
parser.add_argument('--rnn_type', type=int, default=default_rnn_type, nargs='?',\
    choices=range(0, 4),\
    help='Type of rnn cell --> [0:Basic], [1:basic LSTM], [2:full LSTM], [3:GRU].\
        (default:%d)'%default_rnn_type)
parser.add_argument('--learning_rate', type=float, default=default_learning_rate,\
    nargs='?', help='Value of initial learning rate. (default:%r)'\
    %default_learning_rate)
parser.add_argument('--max_grad_norm', type=float, default=default_max_grad_norm,\
    nargs='?', help='Maximum gradient norm allowed. (default:%r)'\
    %default_max_grad_norm)
parser.add_argument('--init_scale', type=float, default=default_init_scale,\
    nargs='?', help='initialize scale. (default:%r)'%default_init_scale)
parser.add_argument('--use_dep', type=t_or_f, default=default_use_dep, nargs='?',\
    choices=[False, True],\
    help='Use dependency tree or not. (default:%r)'%default_use_dep)
parser.add_argument('--max_epoch', type=int, default=default_max_epoch, nargs='?',\
    help='Maximum epoch to be trained. (default:%d)'%default_max_epoch)
parser.add_argument('--keep_prob', type=restricted_float,\
    default=default_keep_prob,\
    nargs='?', help='Keeping-Probability for dropout layer. (default:%r)'\
    %default_keep_prob)
parser.add_argument('--batch_size', type=int, default=default_batch_size, nargs='?',\
    help='Mini-batch size while training. (default:%d)'%default_batch_size)
parser.add_argument('--hidden_size', type=int, default=default_hidden_size, nargs='?',\
    help='Dimension of hidden layer. (default:%d)'%default_hidden_size)
parser.add_argument('--data_dir', type=str, default=default_data_dir, nargs='?',\
    help='Directory where the data are placed. (default:%s)'%default_data_dir)

args = parser.parse_args()

#decide embedding dimension and vocabulary size
wordvec = np.load(args.data_dir+'wordvec.npy')
vocab = open(args.data_dir+'vocab.txt', 'r').read().splitlines()
if len(vocab) != wordvec.shape[0]:
  raise RuntimeError('different numbers of vocabs and vectors!')
if args.wordvec_src == 0:
  args.embed_dim = 50
  args.vocab_size = wordvec.shape[0]
elif args.wordvec_src == 1:
  args.embed_dim = 50
  args.vocab_size = wordvec.shape[0]
elif args.wordvec_src == 2:
  args.embed_dim = 100
  args.vocab_size = wordvec.shape[0]
elif args.wordvec_src == 3:
  args.embed_dim = 200
  args.vocab_size = wordvec.shape[0]
elif args.wordvec_src == 4:
  args.embed_dim = 300
  args.vocab_size = wordvec.shape[0]
elif args.wordvec_src == 5:
  args.embed_dim = 300
  args.vocab_size = wordvec.shape[0]
elif args.wordvec_src == 6:
  args.embed_dim = 300
  args.vocab_size = wordvec.shape[0]
else: assert(False)

def is_train(mode): return mode == 0
def is_valid(mode): return mode == 1
def is_test(mode): return mode == 2

def get_single_example(args):
  '''get one example from TFRecorder file using tensorflow default queue runner'''

  #filename = args.data_dir+'/data.tfr'
  filenames = open(args.data_dir+'train_list.txt', 'r').read().splitlines()
  filenames = [ args.data_dir + fn for fn in filenames ]
  f_queue = tf.train.string_input_producer(filenames, num_epochs=None)
  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(f_queue)

  feature = tf.parse_single_example(serialized_example,\
    features={\
      'content': tf.VarLenFeature(tf.int64),\
      'len': tf.FixedLenFeature([1], tf.int64)})
  return feature['content'], feature['len'][0]

class DepRNN(object):
  '''dependency-tree based rnn'''

  def __init__(self, mode, para):
    '''build multi-layer rnn graph'''
    if para.rnn_type == 0:#basic rnn
      def unit_cell():
        return tf.contrib.rnn.BasicRNNCell(para.hidden_size, activation=tf.tanh)
    elif para.rnn_type == 1:#basic LSTM
      def unit_cell():
        return tf.contrib.rnn.BasicLSTMCell(para.hidden_size, forget_bias=0.0,\
            state_is_tuple=True)
    elif para.rnn_type == 2:#full LSTM
      def unit_cell():
        return tf.contrib.rnn.LSTMCell(para.hidden_size, forget_bias=0.0,\
            state_is_tuple=True)
    elif para.rnn_type == 3:#GRU
      def unit_cell():
        return tf.contrib.rnn.GRUCell(para.hidden_size, state_is_tuple=True)

    rnn_cell = unit_cell

		#dropout layer
    if is_train(mode) and para.keep_prob < 1:
      def rnn_cell():
        return tf.contrib.rnn.DropoutWrapper(\
            unit_cell(), output_keep_prob=para.keep_prob)

    #multi-layer rnn
    cell = tf.contrib.rnn.MultiRNNCell([rnn_cell()] * para.layer_num,\
        state_is_tuple=True)

    #initialize rnn_cell state to zero
    self._initial_state = cell.zero_state(para.batch_size, tf.float32)


    #using pre-trained word embedding
    W_E = tf.Variable(tf.constant(0.0,\
        shape=[para.vocab_size, para.embed_dim]), trainable=False, name='W_E')
    self._embedding = tf.placeholder(tf.float32, [para.vocab_size, para.embed_dim])
    self._embed_init = W_E.assign(self._embedding)

    #feed in data in batches
    one_sent, sq_len = get_single_example(para)
    batch, seq_len = tf.train.batch([one_sent, sq_len],\
        batch_size=para.batch_size, dynamic_pad=True)

    #sparse tensor cannot be sliced
    batch = tf.sparse_tensor_to_dense(batch)

    #seq_len is for dynamic_rnn
    self._seq_len = seq_len = tf.to_int32(seq_len)

    #x and y differ by one position
    batch_x = batch[:, :-1]
    batch_y = batch[:, 1:]

    #word_id to vector
    inputs = tf.nn.embedding_lookup(W_E, batch_x)

    if is_train(mode) and para.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, para.keep_prob)

    #use dynamic_rnn to build dynamic-time-step rnn
    outputs, state = tf.nn.dynamic_rnn(cell, inputs,\
        sequence_length=seq_len, dtype=tf.float32)
    output = tf.reshape(tf.concat(outputs, 1), [-1, para.hidden_size])
    with tf.variable_scope('softmax'):
      softmax_w = tf.get_variable('w', [para.hidden_size, para.vocab_size],\
          dtype=tf.float32)
      softmax_b = tf.get_variable('b', [para.vocab_size], dtype=tf.float32)

    logits = tf.matmul(output, softmax_w)+softmax_b

    if is_test(mode):
      self._prob_op = tf.nn.softmax(logits)

    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],\
        [tf.reshape(batch_y, [-1])],\
        [tf.ones([(tf.reduce_max(seq_len)-1)*para.batch_size], dtype=tf.float32)])

    #self._cost = cost = tf.reduce_sum(loss) /\
    #    tf.to_float(tf.reduce_sum(seq_len)-para.batch_size)
    self._cost = cost = tf.reduce_mean(loss)
    self._final_state = state

    #if validation or testing, exit here
    if not is_train(mode): return

    #clip global gradient norm
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), para.max_grad_norm)
    #optimizer = tf.train.GradientDescentOptimizer(para.learning_rate)
    optimizer = tf.train.AdamOptimizer(para.learning_rate)
    self._eval_op = optimizer.apply_gradients(zip(grads, tvars),\
        global_step=tf.contrib.framework.get_or_create_global_step())

  @property
  def seq_len(self): return self._seq_len
  @property
  def initial_state(self): return self._initial_state
  @property
  def cost(self): return self._cost
  @property
  def final_state(self): return self._final_state
  @property
  def eval_op(self): return self._eval_op
  @property
  def prob_op(self): return self._prob_op

def run_epoch(sess, model, mode):
  '''Runs the model on the given data.'''
  costs = 0.0
  iters = 0
  state = sess.run(model.initial_state)

  fetches = {\
      'cost': model.cost,\
      'final_state': model.final_state,\
      'seq_len': model.seq_len,\
  }

  if is_train(mode): fetches['eval_op'] = model.eval_op
  elif is_test(mode): fetches['prob_op'] = model.prob_op

  for i in range(10):
    fd_dct = {}
    for i, (c, h) in enumerate(model.initial_state):
      fd_dct[c] = state[i].c
      fd_dct[h] = state[i].h

    vals = sess.run(fetches, feed_dict=fd_dct)

    cost = vals['cost']
    state = vals['final_state']
    seq_len = vals['seq_len']
    print('max_len = %d' % (max(seq_len)*args.hidden_size))

    costs += cost
    #iters += ((sum(seq_len)-len(seq_len))/len(seq_len))
    iters += 1
    print( np.exp(costs/iters) )

  return np.exp(costs/iters)

with tf.Graph().as_default():
  initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)

  #mode: 0->train, 1->valid, 2->test
  with tf.name_scope('train'):
    with tf.variable_scope('model', reuse=None, initializer=initializer):
      train_model = DepRNN(mode=0, para=args)
  #with tf.name_scope('valid'):
  #  with tf.variable_scope('model', reuse=True, initializer=initializer):
  #    valid_model = DepRNN(mode=1, para=args)
  #with tf.name_scope('test'):
  #  with tf.variable_scope('model', reuse=True, initializer=initializer):
  #    test_model = DepRNN(mode=2, para=args)

  sv = tf.train.Supervisor(logdir='./logs/', saver=None)
  with sv.managed_session() as sess:

    #load in pre-trained word-embedding
    sess.run(train_model._embed_init, feed_dict={train_model._embedding: wordvec})

    for i in range(args.max_epoch):
      train_perplexity = run_epoch(sess, train_model, mode=0)
      print('Epoch: %d Train Perplexity: %.4f' % (i + 1, train_perplexity))
      #valid_perplexity = run_epoch(sess, valid_model, mode=1)
      #print('Epoch: %d Valid Perplexity: %.3f' % (i + 1, valid_perplexity))
      #test_perplexity = run_epoch(sess, valid_model, mode=2)
      #print('Epoch: %d Test Perplexity: %.3f' % (i + 1, test_perplexity))
