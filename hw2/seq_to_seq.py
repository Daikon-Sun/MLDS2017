#!/usr/bin/python3
import os, copy, csv, sys, json, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.seq2seq as seq2seq
from nltk.tokenize import word_tokenize

#default values (in alphabetic order)
default_batch_size = 512
default_data_dir = './parsed_data/'
default_embed_dim = 500
default_hidden_size = 200
default_info_epoch = 1
default_init_scale = 0.01
default_keep_prob = 1
default_layer_num = 1
default_learning_rate = 0.003
default_rnn_type = 2
default_max_grad_norm = 10
default_max_epoch = 500
default_num_sampled = 2000
default_optimizer = 4
default_output_filename = './submission.csv'
#default_softmax_loss = 0
default_video_dim = 4096
default_video_len = 80
#default_train_num = 522
#default_wordvec_src = 3
optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdadeltaOptimizer,
              tf.train.AdagradOptimizer, tf.train.MomentumOptimizer,
              tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]
#
#src_name = ['6B.50d', '6B.100d', '6B.200d', '6B.300d', '42B.300d', '840B.300d']

#functions for arguments of unsupported types
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('%r not in range [0.0, 1.0]'%x)
    return x

##argument parser
parser = argparse.ArgumentParser(description=
                                 'seq2seq for video caption generation')
#parser.add_argument('-ws', '--wordvec_src',
#                    type=int, default=default_wordvec_src,
#                    nargs='?', choices=range(0, 6), help='Decide the source'
#                    'of wordvec --> [0:glove.6B.50d], [1:glove.6B.100d],'
#                    '[2:glove.6B.200d], [3:glove.6B.300d], [4:glove.42B],'
#                    '[5:glove.840B]. (default:%d)'%default_wordvec_src)
parser.add_argument('-ed', '--embed_dim',
                    type=int, default=default_embed_dim,
                    nargs='?', help='Embedding dimension of vocabularies. '
                    '(default:%d)'%default_embed_dim)
parser.add_argument('-vl', '--video_len',
                    type=int, default=default_video_len,
                    nargs='?', help='Number of a frame in a video. '
                    '(default:%d)'%default_video_len)
parser.add_argument('-vd', '--video_dim',
                    type=int, default=default_video_dim,
                    nargs='?', help='Dimension of a frame from a video. '
                    '(default:%d)'%default_video_dim)
parser.add_argument('-ln', '--layer_num',
                    type=int, default=default_layer_num,
                    nargs='?', help='Number of rnn layer. (default:%d)'
                    %default_layer_num)
parser.add_argument('--info_epoch',
                    type=int, default=default_info_epoch,
                    nargs='?', help='Print information every info_epoch.'
                    '(default:%d)'%default_info_epoch)
parser.add_argument('-ns', '--num_sampled',
                    type=int, default=default_num_sampled,
                    nargs='?', help='Number of classes to be sampled while'
                    'calculating loss (not with full softmax).'
                    '(default:%d)'%default_num_sampled)
parser.add_argument('-opt', '--optimizer',
                    type=int, default=default_optimizer,
                    nargs='?', choices=range(0, 6), help='Optimizers -->'
                    '[0: GradientDescent], [1:Adadelta], [2:Adagrad],'
                    '[3:Momentum], [4:Adam], [5:RMSProp]. (default:%d)'
                    %default_optimizer)
#parser.add_argument('-sl', '--softmax_loss',
#                    type=int, default=default_softmax_loss,
#                    nargs='?', choices=range(0, 3), help='Type of softmax'
#                    'function --> [0:full softmax], [1:sampled softmax],'
#                    '[nce loss]. (default:%d)'%default_softmax_loss)
parser.add_argument('-rt', '--rnn_type',
                    type=int, default=default_rnn_type,
                    nargs='?', choices=range(0, 4), help='Type of rnn cell -->'
                    '[0:Basic], [1:basic LSTM], [2:full LSTM], [3:GRU].'
                    '(default:%d)'%default_rnn_type)
parser.add_argument('-lr', '--learning_rate',
                    type=float, default=default_learning_rate,
                    nargs='?', help='Value of initial learning rate.'
                    '(default:%r)'%default_learning_rate)
parser.add_argument('-mgn', '--max_grad_norm',
                    type=float, default=default_max_grad_norm,
                    nargs='?', help='Maximum gradient norm allowed. (default:%r)'
                    %default_max_grad_norm)
parser.add_argument('-is', '--init_scale',
                    type=float, default=default_init_scale,
                    nargs='?', help='initialize scale. (default:%r)'
                    %default_init_scale)
parser.add_argument('-me', '--max_epoch',
                    type=int, default=default_max_epoch,
                    nargs='?', help='Maximum epoch to be trained.'
                    '(default:%d)'%default_max_epoch)
#parser.add_argument('-tn', '--train_num',
#                    type=int, default=default_train_num,
#                    nargs='?', help='Number of files out of the total 522'
#                    'files to be trained. (default:%d)' %default_train_num)
parser.add_argument('-kp', '--keep_prob',
                    type=restricted_float,
                    default=default_keep_prob, nargs='?', help=
                    'Keeping-Probability for dropout layer.'
                    '(default:%r)'%default_keep_prob)
parser.add_argument('-bs', '--batch_size',
                    type=int, default=default_batch_size,
                    nargs='?', help='Mini-batch size while training.'
                    '(default:%d)'%default_batch_size)
parser.add_argument('-hs', '--hidden_size',
                    type=int, default=default_hidden_size,
                    nargs='?', help='Dimension of hidden layer.'
                    '(default:%d)'%default_hidden_size)
#parser.add_argument('-dd', '--data_dir',
#                    type=str, default=default_data_dir, nargs='?',
#                    help='Directory where the data are placed.'
#                    '(default:%s)'%default_data_dir)
#parser.add_argument('-of', '--output_filename',
#                    type=str, default=default_output_filename, nargs='?',
#                    help='Filename of the final prediction.'
#                    '(default:%s)'%default_output_filename)
args = parser.parse_args()
args.vocab_size = 6089
#
##calculate real epochs
#print('training with about %.3f epochs!'
#      %((args.batch_size*args.max_epoch)/2100000))
#
##load in pre-trained word embedding and vocabulary list
#wordvec = np.load('data/wordvec.'+src_name[args.wordvec_src]+'.npy')
#
##decide vocab_size and embed_dim
#args.vocab_size, args.embed_dim = wordvec.shape
#print('vocab_size = %d'%args.vocab_size)
#print('word embedding dimension = %d'%args.embed_dim)
#
##load in file list for training and validation
#filenames = open('training_list', 'r').read().splitlines()
#filenames = [ args.data_dir+ff[21:-4]+'.tfr' for ff in filenames ]
#assert len(filenames) == 522
#filenames = [filenames[:default_train_num], filenames[default_train_num:],
#    ['testing_data.tfr']]


class DepRNN(object):

  '''dependency-tree based rnn'''

  def __init__(self, para):
    '''build multi-layer rnn graph'''

    self._para = para
    with tf.variable_scope('seq2seq') as scope:
      if para.rnn_type == 0:#basic rnn
        def unit_cell():
          return tf.contrib.rnn.BasicRNNCell(para.hidden_size)
      elif para.rnn_type == 1:#basic LSTM
        def unit_cell():
          return tf.contrib.rnn.BasicLSTMCell(para.hidden_size)
      elif para.rnn_type == 2:#full LSTM
        def unit_cell():
          return tf.contrib.rnn.LSTMCell(para.hidden_size, use_peepholes=True)
      elif para.rnn_type == 3:#GRU
        def unit_cell():
          return tf.contrib.rnn.GRUCell(para.hidden_size)

      rnn_cell = unit_cell

      #dropout layer
      if self.is_train() and para.keep_prob < 1:
        def rnn_cell():
          return tf.contrib.rnn.DropoutWrapper(
              unit_cell(), output_keep_prob=para.keep_prob)

      #multi-layer rnn
      encoder_cell =\
        tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(para.layer_num)])
      decoder_cell =\
        tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(para.layer_num)])

      #feed in data in batches
      video, caption, v_len, c_len = self.get_single_example(para)
      videos, captions, v_lens, c_lens =\
          tf.train.batch([video, caption, v_len, c_len],
                         batch_size=para.batch_size, dynamic_pad=True)
      #videos = tf.reshape(videos, [-1, para.video_dim])

      #sparse tensor cannot be sliced
      targets = tf.sparse_tensor_to_dense(captions)
      self._input = targets

      #seq_len is for dynamic_rnn
      v_lens = tf.to_int32(v_lens)
      self._v_lens = v_lens
      c_lens = tf.to_int32(c_lens)
      self._c_lens = c_lens

      #x and y differ by one position
      #inputs = batch[:, :-1]
      #self._f_target = batch[:, 1:]

      #word_id to vector
      with tf.variable_scope('embedding'):
        W_E = tf.get_variable('W_E', [para.vocab_size, para.embed_dim],
                              dtype=tf.float32)
      #  V_W = tf.get_variable('V_W', [para.video_dim, para.embed_dim],
      #                        dtype=tf.float32)
      #  V_B = tf.get_variable('V_B', [para.embed_dim], dtype=tf.float32)
      inputs =\
        tf.contrib.layers.legacy_fully_connected(videos, para.embed_dim)
      #inputs = tf.nn.xw_plus_b(videos, V_W, V_B)
      targets_embed = tf.nn.embedding_lookup(W_E, targets)

      if self.is_train() and para.keep_prob < 1:
        inputs = tf.nn.dropout(inputs, para.keep_prob)

      #use dynamic_rnn to build dynamic-time-step rnn
      encoder_outputs, encoder_states =\
        tf.nn.dynamic_rnn(encoder_cell, inputs,
                          sequence_length=v_lens, dtype=tf.float32)
      encoder_output = tf.reshape(encoder_outputs, [-1, para.hidden_size])

      #self._states = encoder_states

      #assert isinstance(encoder_states, LSTMStateTuple)

      #pass_state =\
      #  LSTMStateTuple(c=tf.zeros(tf.shape(encoder_states.c)),
      #                 h=encoder_states.h)

      decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_states)
      scope.reuse_variables()
      decoder_outputs, _, _ =\
        seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                    decoder_fn=decoder_fn_train,
                                    inputs=targets_embed,
                                    sequence_length=c_lens)
      decoder_outputs = tf.reshape(decoder_outputs, [-1, para.hidden_size])
      c_len_max = tf.reduce_max(c_lens)

    with tf.variable_scope('softmax') as scope:
      softmax_w = tf.get_variable('w', [para.hidden_size, para.vocab_size],
          dtype=tf.float32)
      softmax_b = tf.get_variable('b', [para.vocab_size], dtype=tf.float32)

    logits = tf.nn.xw_plus_b(decoder_outputs, softmax_w, softmax_b)
    logits = tf.reshape(logits, [para.batch_size, c_len_max, para.vocab_size])
    #loss = softmax[para.softmax_loss](softmax_w, softmax_b,
    #       tf.reshape(targets, [-1, 1]), decoder_outputs,
    #       num_sampled=para.num_sampled, num_classes=para.vocab_size)
    loss =\
      tf.contrib.seq2seq.sequence_loss(logits, targets,
                                       tf.ones([para.batch_size, c_len_max]))

    self._cost = cost = tf.reduce_mean(loss)

    #if validation or testing, exit here
    #if not is_train(para.mode): return

    #clip global gradient norm
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
               para.max_grad_norm)
    optimizer = optimizers[para.optimizer](para.learning_rate)
    self._eval = optimizer.apply_gradients(zip(grads, tvars),
                 global_step=tf.contrib.framework.get_or_create_global_step())

  def is_train(self): return self._para.mode == 0
  def is_valid(self): return self._para.mode == 1
  def is_test(self): return self._para.mode == 2

  def get_single_example(self, para):
    '''get one example from TFRecorder file using tf default queue runner'''
    prefix = 'MLDS_hw2_data/training_data/'
    filenames = prefix+'training_list.txt'
    filelist = open(filenames, 'r').read().splitlines()
    filenames = [ 'parsed_data/'+fl[:-4]+'.tfr' for fl in filelist ]
    f_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(f_queue)

    feature = tf.parse_single_example(serialized_example,
      features={
          'video': tf.FixedLenFeature([para.video_len*para.video_dim],
                                      tf.float32),
          'caption': tf.VarLenFeature(tf.int64)})
    video = tf.reshape(feature['video'], [para.video_len, para.video_dim])
    caption = feature['caption']
    return video, caption, tf.shape(video)[0], tf.shape(caption)[0]

  @property
  def cost(self): return self._cost
  @property
  def eval(self): return self._eval
  @property
  def prob(self): return self._prob
  @property
  def c_lens(self): return self._c_lens
  @property
  def v_lens(self): return self._v_lens
  @property
  def input(self): return self._input

def run_epoch(sess, model, args):
  '''Runs the model on the given data.'''
  fetches = {}
  if not model.is_test():
    #fetches['cost'] = model.cost
    #if model.is_train():
    #  fetches['eval'] = model.eval
    fetches['v_len'] = model.v_lens
    vals = sess.run(fetches)
    return 0
    return np.exp(vals['cost'])

  else:
    fetches['prob'] = model.prob
    fetches['f_target'] = model.f_target
    if args.use_bi: fetches['b_target'] = model.b_target

    vals = sess.run(fetches)
    prob = vals['prob']
    f_target = vals['f_target']
    if args.use_bi: b_target = vals['b_target']

    #shape of choices = 5 x (len(sentence)-1)
    sent_len = f_target.shape[1]
    f_choices = np.array([[prob[k*sent_len+j, f_target[k, j]]
        for j in range(sent_len)] for k in range(5)])

    if not args.use_bi:
      return chr(ord('a')+np.argmax(np.sum(np.log(f_choices), axis=1)))

    b_choices = np.array([[prob[(5+k)*sent_len+j, b_target[k, j]]
        for j in range(sent_len)] for k in range(5)])
    return chr(ord('a')+
           np.argmax(np.sum(np.log(f_choices)+np.log(b_choices), axis=1)))

if __name__ == '__main__':
  #with open('MLDS_hw2_data/training_label.json', 'r') as label_json:
  #  labels = json.load(label_json)
  #  captions = [ [ word_tokenize(sent.lower()) for sent in label['caption'] ]
  #             for label in labels ]
  #  sents = [ sent for caption in captions for sent in caption ]
  #  vocabs = set([ word for sent in sents for word in sent ])
  #  vocabs = ['<pad>', '<unk>', '<bos>', '<eos>'] + list(vocabs)
  #  dct = dict([ (word, i) for i, word in enumerate(vocabs)])

  #with open('parsed_data/vocab.txt', 'w') as f:
  #  for word in dct.keys():
  #    f.write(word+'\n')

  #with open('MLDS_hw2_data/training_label.json', 'r') as label_json:
  #  labels = json.load(label_json)
  #  for i, label in enumerate(labels):
  #    out_name = 'parsed_data/'+label['id']+'.tfr'
  #    if not tf.gfile.Exists(out_name):
  #      video = np.load('MLDS_hw2_data/training_data/feat/'+label['id']+'.npy')
  #      video = video.reshape((-1, 1))
  #      writer = tf.python_io.TFRecordWriter(out_name)
  #      for j, sent in enumerate(captions[i]):
  #        word_ids = [ dct[word] for word in sent ]
  #        word_ids = [2] + word_ids + [3]
  #        example = tf.train.Example(
  #          features=tf.train.Features(
  #            feature={
  #              'video': tf.train.Feature(
  #                float_list=tf.train.FloatList(value=video)),
  #              'caption': tf.train.Feature(
  #                int64_list=tf.train.Int64List(value=word_ids))}))
  #        serialized = example.SerializeToString()
  #        writer.write(serialized)
  #sys.exit(0)
  #def full_softmax(weights, biases, labels, inputs, num_sampled, num_classes):
  #  return tf.contrib.legacy_seq2seq.sequence_loss_by_example(
  #         [tf.matmul(inputs, tf.transpose(weights))+biases],
  #         [tf.reshape(labels, [-1])],
  #         [tf.ones([labels.get_shape()[1]*args.batch_size], tf.float32)])
  #softmax = [full_softmax, tf.nn.sampled_softmax_loss, tf.nn.nce_loss]

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-args.init_scale,
                                                args.init_scale)

    #mode: 0->train, 1->valid, 2->test
    with tf.name_scope('train'):
      train_args = copy.deepcopy(args)
      with tf.variable_scope('model', reuse=None, initializer=initializer):
        train_args.mode = 0
        train_model = DepRNN(para=train_args)
    #if args.train_num < 522:
    #  with tf.name_scope('valid'):
    #    valid_args = copy.deepcopy(args)
    #    with tf.variable_scope('model', reuse=True, initializer=initializer):
    #      valid_args.mode = 1
    #      valid_model = DepRNN(para=valid_args)
    #with tf.name_scope('test'):
    #  test_args = copy.deepcopy(args)
    #  with tf.variable_scope('model', reuse=True, initializer=initializer):
    #    test_args.mode = 2
    #    test_args.batch_size = 5
    #    test_model = DepRNN(para=test_args)

    sv = tf.train.Supervisor(logdir='./logs/')
    with sv.managed_session() as sess:

      #load in pre-trained word-embedding
      #sess.run(train_model._embed_init,
      #         feed_dict={train_model._embedding: wordvec})
      #sess.run(test_model._embed_init,
      #         feed_dict={test_model._embedding: wordvec})

      for i in range(1, args.max_epoch+1):
        train_perplexity = run_epoch(sess, train_model, train_args)
        if i%args.info_epoch == 0:
          print('Epoch: %d Train Perplexity: %.4f'%(i, train_perplexity))
        #if args.train_num < 522:
        #  valid_perplexity = run_epoch(sess, valid_model, valid_args)
        #  if i%args.info_epoch == 0:
        #    print('Epoch: %d Valid Perplexity: %.4f'%(i, valid_perplexity))
      #with open(args.output_filename, 'w') as f:
      #  wrtr = csv.writer(f)
      #  wrtr.writerow(['id', 'answer'])
      #  for i in range(1040):
      #    result = run_epoch(sess, test_model, test_args)
      #    wrtr.writerow([i+1, result])
