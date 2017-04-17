#!/usr/bin/python3
import os, copy, csv, sys, json, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq import sequence_loss as sequence_loss
from tensorflow.contrib.layers import legacy_fully_connected as fully_connected
from nltk.tokenize import word_tokenize

class S2S(object):

  def __init__(self, para):
    para.fac = int(para.bidirectional)+1
    self._para = para
    if para.rnn_type == 0:#basic rnn
      def unit_cell(fac):
        return tf.contrib.rnn.BasicRNNCell(para.hidden_size * fac)
    elif para.rnn_type == 1:#basic LSTM
      def unit_cell(fac):
        return tf.contrib.rnn.BasicLSTMCell(para.hidden_size * fac)
    elif para.rnn_type == 2:#full LSTM
      def unit_cell(fac):
        return tf.contrib.rnn.LSTMCell(para.hidden_size * fac, use_peepholes=True)
    elif para.rnn_type == 3:#GRU
      def unit_cell(fac):
        return tf.contrib.rnn.GRUCell(para.hidden_size * fac)

    rnn_cell = unit_cell

    #dropout layer
    if not self.is_test() and para.keep_prob < 1:
      def rnn_cell(fac):
        return tf.contrib.rnn.DropoutWrapper(
            unit_cell(fac), output_keep_prob=para.keep_prob)

    #multi-layer rnn
    encoder_cell =\
      tf.contrib.rnn.MultiRNNCell([rnn_cell(1) for _ in range(para.layer_num)])
    if para.bidirectional:
      b_encoder_cell =\
        tf.contrib.rnn.MultiRNNCell([rnn_cell(1) for _ in range(para.layer_num)])

    #feed in data in batches
    if not self.is_test():
      video, caption, v_len, c_len = self.get_single_example(para)
      videos, captions, v_lens, c_lens =\
          tf.train.batch([video, caption, v_len, c_len],
                         batch_size=para.batch_size, dynamic_pad=True)
      #sparse tensor cannot be sliced
      targets = tf.sparse_tensor_to_dense(captions)
      decoder_in = targets[:, :-1]
      decoder_out = targets[:, 1:]
      c_lens = tf.to_int32(c_lens)
    else:
      video, v_len = self.get_single_example(para)
      videos, v_lens =\
          tf.train.batch([video, v_len],
                         batch_size=para.batch_size, dynamic_pad=True)
    v_lens = tf.to_int32(v_lens)
    with tf.variable_scope('embedding'):
      W_E = tf.get_variable('W_E', [para.vocab_size, para.embed_dim],
                            dtype=tf.float32)
    if not self.is_test():
      decoder_in_embed = tf.nn.embedding_lookup(W_E, decoder_in)

    inputs = fully_connected(videos, para.embed_dim)

    if not self.is_test() and para.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, para.keep_prob)

    if not para.bidirectional:
      encoder_outputs, encoder_states =\
        tf.nn.dynamic_rnn(encoder_cell, inputs,
                          sequence_length=v_lens, dtype=tf.float32)
    else:
      encoder_outputs, encoder_states =\
        tf.nn.bidirectional_dynamic_rnn(encoder_cell, b_encoder_cell,
                                        inputs, sequence_length=v_lens,
                                        dtype=tf.float32)
      encoder_states = tuple([LSTMStateTuple(tf.concat([f_st.c, f_st.c], 1),
                                             tf.concat([b_st.h, b_st.h], 1))
                              for f_st, b_st in zip(encoder_states[0],
                                                    encoder_states[1])])
      encoder_outputs = tf.concat([encoder_outputs[0], encoder_outputs[1]], 2)

    self._val1 = videos
    self._val2 = inputs
    #ww = [v for v in tf.global_variables() if v.name == "model/fully_connected/weights:0"][0]
    #bb = [v for v in tf.global_variables() if v.name == "model/fully_connected/bias:0"][0]

    with tf.variable_scope('softmax'):
      softmax_w = tf.get_variable('w', [para.hidden_size*para.fac,
                                        para.vocab_size], dtype=tf.float32)
      softmax_b = tf.get_variable('b', [para.vocab_size], dtype=tf.float32)
      output_fn = lambda output: tf.nn.xw_plus_b(output, softmax_w, softmax_b)

    decoder_cell =\
      tf.contrib.rnn.MultiRNNCell([rnn_cell(para.fac)
                                   for _ in range(para.layer_num)])
    if para.attention:
      (at_keys, at_vals, at_score, at_cons) =\
        seq2seq.prepare_attention(
          attention_states=encoder_outputs,
          attention_option="bahdanau",
          num_units=para.hidden_size*para.fac)

    if self.is_test():
      if para.attention:
        decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=encoder_states,
            attention_keys=at_keys,
            attention_values=at_vals,
            attention_score_fn=at_score,
            attention_construct_fn=at_cons,
            embeddings=W_E,
            start_of_sequence_id=2,
            end_of_sequence_id=3,
            maximum_length=30,
            num_decoder_symbols=para.vocab_size)
      else:
        decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
            output_fn=output_fn,
            encoder_state=encoder_states,
            embeddings=W_E,
            start_of_sequence_id=2,
            end_of_sequence_id=3,
            maximum_length=30,
            num_decoder_symbols=para.vocab_size)
      with tf.variable_scope('decode', reuse=True):
        decoder_logits, _, _ =\
          seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                    decoder_fn=decoder_fn_inference)
      self._prob = tf.nn.softmax(decoder_logits)

    else:
      if para.attention:
        decoder_fn_train = seq2seq.attention_decoder_fn_train(
            encoder_state=encoder_states,
            attention_keys=at_keys,
            attention_values=at_vals,
            attention_score_fn=at_score,
            attention_construct_fn=at_cons)
      else:
        decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_states)

      with tf.variable_scope('decode', reuse=None):
        (decoder_outputs, _, _) =\
          seq2seq.dynamic_rnn_decoder(cell=decoder_cell,
                                      decoder_fn=decoder_fn_train,
                                      inputs=decoder_in_embed,
                                      sequence_length=c_lens)
      decoder_outputs =\
        tf.reshape(decoder_outputs, [-1, para.hidden_size*para.fac])
      c_len_max = tf.reduce_max(c_lens)

      logits = output_fn(decoder_outputs)
      logits = tf.reshape(logits, [para.batch_size, c_len_max, para.vocab_size])
      self._prob = tf.nn.softmax(logits)

      msk = tf.sequence_mask(c_lens, dtype=tf.float32)
      loss = sequence_loss(logits, decoder_out, msk)

      self._cost = cost = tf.reduce_mean(loss)

      #if validation or testing, exit here
      if self.is_valid(): return

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
    if self.is_test():
      filelist = open('testing_list.txt', 'r').read().splitlines()
      filenames = [ 'test_tfrdata/'+fl+'.tfr' for fl in filelist ]
      #filelist = open('training_list.txt', 'r').read().splitlines()
      #filenames = [ 'train_tfrdata/'+fl+'.tfr' for fl in filelist ]
      #filelist = open('time_limited_list.txt', 'r').read().splitlines()
      #filenames = [ 'time_limited_tfrdata/'+fl+'.tfr' for fl in filelist ]
      f_queue = tf.train.string_input_producer(filenames, shuffle=False)
    else:
      filelist = open('training_list.txt', 'r').read().splitlines()
      filenames = [ 'train_tfrdata/'+fl+'.tfr' for fl in filelist ]
      if self.is_train(): filenames = filenames[:para.train_num]
      else: filenames = filenames[para.train_num:]
      f_queue = tf.train.string_input_producer(filenames, shuffle=True)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(f_queue)

    if self.is_test():
      feature = tf.parse_single_example(serialized_example, features={
        'video': tf.FixedLenFeature([para.video_len*para.video_dim],
                                    tf.float32)})
      video = tf.reshape(feature['video'], [para.video_len, para.video_dim])
      return video, tf.shape(video)[0]
    else:
      feature = tf.parse_single_example(serialized_example,
        features={
            'video': tf.FixedLenFeature([para.video_len*para.video_dim],
                                        tf.float32),
            'caption': tf.VarLenFeature(tf.int64)})
      video = tf.reshape(feature['video'], [para.video_len, para.video_dim])
      caption = feature['caption']
      return video, caption, tf.shape(video)[0], tf.shape(caption)[0]-1

  @property
  def cost(self): return self._cost
  @property
  def eval(self): return self._eval
  @property
  def prob(self): return self._prob
  @property
  def val1(self): return self._val1
  @property
  def val2(self): return self._val2

def run_epoch(sess, model, args):
  '''Runs the model on the given data.'''
  fetches = {}
  if not model.is_test():
    fetches['cost'] = model.cost
    if model.is_train():
      fetches['eval'] = model.eval
    vals = sess.run(fetches)
    return np.exp(vals['cost'])

  else:
    fetches['prob'] = model.prob
    vals = sess.run(fetches)
    prob = vals['prob']

    bests = []
    for i in range(prob.shape[0]):
      ps, ans = [], []
      for j in range(prob.shape[1]):
        mx_i = np.argmax(prob[i, j, :])
        if mx_i == 3:
          break
        ans.append(dct[mx_i])
      bests.append(ans)
    return bests

if __name__ == '__main__':

  #default values (in alphabetic order)
  default_batch_size = 145
  default_data_dir = './train_tfrdata/'
  default_embed_dim = 512
  default_hidden_size = 256
  default_info_epoch = 1
  default_init_scale = 0.005
  default_keep_prob = 0.7
  default_layer_num = 2
  default_learning_rate = 0.001
  default_rnn_type = 2
  default_max_grad_norm = 5
  default_max_epoch = 10000
  default_num_sampled = 2000
  default_optimizer = 4
  default_output_filename = './output.json'
  #default_softmax_loss = 0
  default_video_dim = 4096
  default_video_len = 80
  default_train_num = 1450
  #default_wordvec_src = 3
  optimizers = [tf.train.GradientDescentOptimizer, tf.train.AdadeltaOptimizer,
                tf.train.AdagradOptimizer, tf.train.MomentumOptimizer,
                tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]

  #functions for arguments of unsupported types
  def restricted_float(x):
      x = float(x)
      if x < 0.0 or x > 1.0:
          raise argparse.ArgumentTypeError('%r not in range [0.0, 1.0]'%x)
      return x

  ##argument parser
  parser = argparse.ArgumentParser(description=
                                   'seq2seq for video caption generation')
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
  parser.add_argument('-tn', '--train_num',
                      type=int, default=default_train_num,
                      nargs='?', help='Number of files out of the total 1450'
                      'files to be trained. (default:%d)' %default_train_num)
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
  parser.add_argument('-bi', '--bidirectional',
                      help='use bidirectional rnn instead of unidirectional '
                      'rnn during encoding', action='store_true')
  parser.add_argument('-at', '--attention',
                      help='add attention', action='store_true')
  #parser.add_argument('-dd', '--data_dir',
  #                    type=str, default=default_data_dir, nargs='?',
  #                    help='Directory where the data are placed.'
  #                    '(default:%s)'%default_data_dir)
  #parser.add_argument('-of', '--output_filename',
  #                    type=str, default=default_output_filename, nargs='?',
  #                    help='Filename of the final prediction.'
  #                    '(default:%s)'%default_output_filename)
  args = parser.parse_args()

  #calculate real epochs
  print('training with about %.3f epochs!'
        %((args.batch_size*args.max_epoch)/1450))

  dct_file = 'train_tfrdata/vocab.txt'
  if tf.gfile.Exists(dct_file):
    vocab = open(dct_file, 'r').read().splitlines()
    dct = dict([[i, word] for i, word in enumerate(vocab)])
  else:
    with open('MLDS_hw2_data/training_label.json', 'r') as label_json:
      labels = json.load(label_json)
      captions = [ [ word_tokenize(sent.lower()) for sent in label['caption'] ]
                 for label in labels ]
      sents = [ sent for caption in captions for sent in caption ]
      vocabs = set([ word for sent in sents for word in sent ])
      vocabs = ['<pad>', '<unk>', '<bos>', '<eos>'] + list(vocabs)
      dct = dict([(word, i) for i, word in enumerate(vocabs)])
    with open('train_tfrdata/vocab.txt', 'w') as f:
      for word in dct.keys():
        f.write(word+'\n')
  args.vocab_size = len(dct)
  print('vocab_size = %d'%args.vocab_size)

  with open('MLDS_hw2_data/training_label.json', 'r') as label_json:
    labels = json.load(label_json)
    for i in tqdm(range(len(labels))):
      label = labels[i]
      out_name = 'train_tfrdata/'+label['id']+'.tfr'
      if not tf.gfile.Exists(out_name):
        video = np.load('MLDS_hw2_data/training_data/feat/'+label['id']+'.npy')
        video = video.reshape((-1, 1))
        max_i = np.argmax([len(caption) for caption in captions[i]])
        writer = tf.python_io.TFRecordWriter(out_name)
        word_ids = [ dct[word] for word in captions[i][max_i] ]
        word_ids = [2] + word_ids + [3]
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'video': tf.train.Feature(
                float_list=tf.train.FloatList(value=video)),
              'caption': tf.train.Feature(
                int64_list=tf.train.Int64List(value=word_ids))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
        writer.close()

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-args.init_scale,
                                                args.init_scale)
    #mode: 0->train, 1->valid, 2->test
    with tf.name_scope('train'):
      train_args = copy.deepcopy(args)
      with tf.variable_scope('model', reuse=None, initializer=initializer):
        train_args.mode = 0
        train_model = S2S(para=train_args)
    if args.train_num < 1450:
      with tf.name_scope('valid'):
        valid_args = copy.deepcopy(args)
        with tf.variable_scope('model', reuse=True, initializer=initializer):
          valid_args.mode = 1
          valid_model = S2S(para=valid_args)
    with tf.name_scope('test'):
      test_args = copy.deepcopy(args)
      with tf.variable_scope('model', reuse=True, initializer=initializer):
        test_args.mode = 2
        test_args.batch_size = 1
        test_model = S2S(para=test_args)

    sv = tf.train.Supervisor(logdir='./logs/')
    with sv.managed_session() as sess:

      for i in range(1, args.max_epoch+1):
        train_perplexity = run_epoch(sess, train_model, train_args)
        if i%args.info_epoch == 0:
          print('Epoch: %d Train Perplexity: %.4f'%(i, train_perplexity))
        if args.train_num < 1450:
          valid_perplexity = run_epoch(sess, valid_model, valid_args)
          if i%args.info_epoch == 0:
            print('Epoch: %d Valid Perplexity: %.4f'%(i, valid_perplexity))
            print('-'*120)
      results = []
      for i in range(1):
        results.extend(run_epoch(sess, test_model, test_args))
      for result in results:
        print(result)
  filelist = open('testing_list.txt', 'r').read().splitlines()
  filenames = [ fl+'.tfr' for fl in filelist ]
  output = [{"caption": result, "id": filename}
         for result, filename in zip(results, filenames)]
  with open('output.json', 'w') as f:
    json.dump(output, f)
