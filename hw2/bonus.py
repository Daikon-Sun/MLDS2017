import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import copy
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell

default_rnn_cell_type         = 1    # 0: BsicRNN, 1: BasicLSTM, 2: FullLSTM, 3: GRU
default_video_dimension       = 4096 # dimension of each frame
default_video_frame_num       = 80   # each video has fixed 80 frames          
default_vocab_size            = 6089
default_max_caption_length    = 20
default_embedding_dimension   = 500  # embedding dimension for video and vocab
default_hidden_units          = 1000 # according to paper
default_batch_size            = 50
default_layer_number          = 1
default_max_gradient_norm     = 10
default_dropout_keep_prob     = 0.5    # for dropout layer
default_init_scale            = 0.005  # for tensorflow initializer
default_max_epoch             = 10000
default_info_epoch            = 1
default_testing_video_num     = 50     # number of testing videos
default_learning_rate         = 0.0001
default_learning_rate_decay_factor = 1


default_optimizer_type = 4
default_optimizers = [tf.train.GradientDescentOptimizer, # 0
                      tf.train.AdadeltaOptimizer,        # 1
                      tf.train.AdagradOptimizer,         # 2
                      tf.train.MomentumOptimizer,        # 3
                      tf.train.AdamOptimizer,            # 4
                      tf.train.RMSPropOptimizer]         # 5


# default value for special vocabs
PAD = 0
BOS = 1
EOS = 2
UNK = 3

# define mode parameters
default_training_mode   = 0
default_validating_mode = 1
default_testing_mode    = 2


class S2VT(object):

  def __init__(self, para):

    self._para = para

    def single_cell():
      if para.rnn_cell_type == 0:
        return tf.contrib.rnn.BasicRNNCell(para.hidden_units, activation=tf.tanh)
      elif para.rnn_cell_type == 1:
        return tf.contrib.rnn.BasicLSTMCell(para.hidden_units, state_is_tuple=False)
      elif para.rnn_cell_type == 2:
        return tf.contrib.rnn.LSTMCell(para.hidden_units, use_peepholes=True, state_is_tuple=False)
      elif para.rnn_cell_type == 3:
        return tf.contrib.rnn.GRUCell(para.hidden_units)

    # dropout layer
    if self.is_train() and para.dropout_keep_prob < 1:
      def rnn_cell():
        return tf.contrib.rnn.DropoutWrapper(
          single_cell(), output_keep_prob=para.dropout_keep_prob)
    else:
      def rnn_cell():
        return single_cell()

    # multi-layer within a layer
    if para.layer_number > 1:
      layer_1_cell = tf.contrib.rnn.MultiRNNCell(
        [rnn_cell() for _ in range(para.layer_number)], state_is_tuple=True)
      layer_2_cell = tf.contrib.rnn.MultiRNNCell(
        [rnn_cell() for _ in range(para.layer_number)], state_is_tuple=True)
    else:
      layer_1_cell = rnn_cell()
      layer_2_cell = rnn_cell()

    # get data in batches
    if self.is_train():
      video, caption, video_len, caption_len = self.get_single_example(para)
      videos, captions, video_lens, caption_lens = tf.train.batch([video, caption, video_len, caption_len],
        batch_size=para.batch_size, dynamic_pad=True)
      # sparse tensor cannot be sliced
      caption_lens = tf.to_int32(caption_lens)
      target_captions = tf.sparse_tensor_to_dense(captions)
      target_captions_input  = target_captions[:,  :-1] # start from <BOS>
      target_captions_output = target_captions[:, 1:  ] # end by <EOS>
    else:
      video, video_len = self.get_single_example(para)
      videos, video_lens = tf.train.batch([video, video_len],
        batch_size=para.batch_size, dynamic_pad=True)
    self._val = videos # why?

    # video and word embeddings as well as word decoding
    with tf.variable_scope('word_embedding'):
      word_embedding_w = tf.get_variable('word_embed',
        [para.vocab_size, para.embedding_dimension])

    with tf.variable_scope('video_embedding'):
      video_embedding_w = tf.get_variable('video_embed',
        [para.video_dimension, para.embedding_dimension])

    with tf.variable_scope('word_decoding'):
      word_decoding_w = tf.get_variable('word_decode',
        [para.embedding_dimension, para.vocab_size])

    # embed videos and captions
    embed_video_inputs = tf.matmul(videos, video_embedding_w)
    embed_targets      = tf.nn.embedding_lookup(word_embedding_w, target_captions)

    # apply dropout to inputs
    if self.is_train() and para.dropout_keep_prob < 1:
      embed_video_inputs = tf.nn.dropout(embed_video_inputs, para.dropout_keep_prob)

    # Initial state of the LSTM memory.
    state_1 = tf.zeros([para.batch_size, layer_1_cell.state_size])
    state_2 = tf.zeros([para.batch_size, layer_2_cell.state_size])
    probabilities = []
    loss = 0.0

    # paddings for 1st and 2nd layers
    layer_1_padding = tf.zeros([para.batch_size, para.embedding_dimension])
    layer_2_padding = tf.zeros([para.batch_size, para.embedding_dimension])

    # ====================== ENCODING STAGE ======================
    for i in range(0, para.video_frame_num):
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      with tf.variable_scope("layer_1"):
        layer_1_output, state_1 = layer_1_cell(embed_video_inputs[:,i,:], state_1) # batch_size x frame_num x embed_dim
      with tf.variable_scope("layer_2"):
        layer_2_output, state_2 = layer_2_cell(tf.concat([layer_2_padding, layer_1_output], 1))

    # ====================== ENCODING STAGE ======================
    for i in range(0, para.max_caption_length):
      current_caption_embed = tf.nn.embedding_lookup(word_embedding_w, target_captions_input[:,i])

      tf.get_variable_scope().reuse_variables()
      with tf.variable_scope("layer_1"):
        layer_1_output, state_1 = layer_1_cell(layer_1_padding, state_1)
      with tf.variable_scope("layer_2"):
        layer_2_output, state_2 = layer_2_cell(tf.concat([current_caption_embed, layer_1_output], 1), state_2)

      labels  = tf.expand_dims(target_captions_output[:,i], 1)
      indices = tf.expand_dims(tf.range(0,para.batch_size,1), 1)
      one_hot = tf.sparse_to_dense(tf.concat([indices, labels], 1),
        tf.pack([para.batch_size, para.vocab_size, 1, 0]))

      layer_2_output_logit = tf.matmul(layer_2_output, word_decoding_w)
      self._prob = layer_2_output_logit

      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(layer_2_output_logit, one_hot)

      self._cost = cost + tf.reduce_mean(cross_entropy)

      # clip gradient norm
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                   para.max_gradient_norm)
      optimizer  = default_optimizers[para.optimizer_type](para.learning_rate)
      self._eval = optimizer.apply_gradients(zip(grads, tvars),
                     global_step=tf.contrib.framework.get_or_create_global_step())

  # ======================== end of __init__ ======================== #

  def is_train(self): return self._para.mode == 0
  def is_valid(self): return self._para.mode == 1
  def  is_test(self): return self._para.mode == 2

  @property
  def cost(self): return self._cost
  @property
  def eval(self): return self._eval
  @property
  def prob(self): return self._prob
  @property
  def val(self):  return self._val

  def get_single_example(self, para):

    # first construct a queue containing a list of filenames.
    filename_queue = tf.train.string_input_producer([para.filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    if self.is_train():
      features = tf.parse_single_example(
        serialized_example,
        features={
          'video': tf.FixedLenFeature([para.video_frame_num*para.video_dimension], tf.float32),
          'caption': tf.VarLenFeature(tf.int64)
        })
      video = tf.reshape(features['video'], [para.video_frame_num, para.video_dimension])
      caption = features['caption']
      return video, caption, tf.shape(video)[0], tf.shape(caption)[0]
    else:
      features = tf.parse_single_example(
        features={
          'video': tf.FixedLenFeature([para.video_frame_num*para.video_dimension], tf.float32)
        })
      video = tf.reshape(features['video'], [para.video_frame_num, para.video_dimension])
      return video, tf.shape(video)[0]

def run_epoch(sess, model, args):
  fetches = {}
  if not model.is_test():
    fetches['cost'] = model.cost
    if model.is_train():
      fetches['evel'] = model.eval
    vals = sess.run(fetches)
    return np.exp(vals['cost'])
  else:
    fetches['prob'] = model.prob
    vals = sess.run(fetches)
    prob = vals['prob']
    bests = []
    for i in range(prob.shape[0]):
      ans = []
      for j in range(prob.shape[1]):
        max_id = np.argmax(prob[i, j, :])
        if max_id == EOS:
          break
        ans.append(dct[max_id])
      bests.append(ans)
    return bests

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='S2VT encoder and decoder')
  argparser.add_argument('-type', '--rnn_cell_type',
    type=int, default=default_rnn_cell_type,
    help='rnn cell type: 0->BasicRNN, 1->BasicLSTM, 2->FullLSTM, 3->GRU')
  argparser.add_argument('-vd', '--video_dimension',
    type=int, default=default_video_dimension,
    help='video dimension (default:%d)' %default_video_dimension)
  argparser.add_argument('-vfn', '--video_frame_num',
    type=int, default=default_video_frame_num,
    help='video frame numbers (default:%d)' %default_video_frame_num)
  #argparser.add_argument('-vs', '--vocab_size',
  #  type=int, default=default_vocab_size,
  #  help='vocab size (default:%d)' %default_vocab_size)
  argparser.add_argument('-mcl', '--max_caption_length',
    type=int, default=default_max_caption_length,
    help='maximum output caption length (default:%d)' %default_max_caption_length)
  argparser.add_argument('-ed', '--embedding_dimension',
    type=int, default=default_embedding_dimension,
    help='embedding dimension of video and caption (default:%d)' %default_embedding_dimension)
  argparser.add_argument('-hu', '--hidden_units',
    type=int, default=default_hidden_units,
    help='hidden units of rnn cell (default:%d)' %default_hidden_units)
  argparser.add_argument('-bs', '--batch_size',
    type=int, default=default_batch_size,
    help='batch size (default:%d)' %default_batch_size)
  argparser.add_argument('-ln', '--layer_number',
    type=int, default=default_layer_number,
    help='layer number within a layer (default:%d)' %default_layer_number)
  argparser.add_argument('-gn', '--max_gradient_norm',
    type=int, default=default_max_gradient_norm,
    help='maximum gradient norm (default:%d' %max_gradient_norm)
  argparser.add_argument('-kp', '--dropout_keep_prob',
    type=int, default=default_dropout_keep_prob,
    help='keep probability of dropout layer (default:%d)' %default_dropout_keep_prob)
  argparser.add_argument('-lr', '--learning_rate',
    type=int, default=default_learning_rate,
    help='learning rate (default:%d' %default_learning_rate)
  argparser.add_argument('-lrdf', '--learning_rate_decay_factor',
    type=int, default=default_learning_rate_decay_factor,
    help='learning rate decay factor (default:%d)' %default_learning_rate_decay_factor)
  argparser.add_argument('-ot', '--optimizer_type',
    type=int, default=default_optimizer_type,
    help='type of optimizer (default:%d)' %default_optimizer_type)
  argparser.add_argument('-is', '--init_scale',
    type=int, default=default_init_scale,
    help='initialization scale for tensorflow initializer (default:%d', %default_init_scale)
  argparser.add_argument('-me', '--max_epoch',
    type=int, default=default_max_epoch,
    help='maximum training epoch (default:%d' %default_max_epoch)
  argparser.add_argument('-ie', '--info_epoch',
    type=int, default=default_info_epoch,
    help='show training information for each (default:%d) epochs' %default_info_epoch)
  args = argparser.parse_args()


  print('S2VT start...\n')
  
  print('Loading vocab dictionary...\n')
  vocab_dictionary_path = 'MLDS_hw2_data/training_data/jason_vocab.json'
  with open(vocab_dictionary_path) as vocab_dictionary_json:
    vocab_dictionary = json.load(vocab_dictionary_json)
  args.vocab_size = len(vocab_dictionary)
  print('vocab_size = %d' %args.vocab_size)

  with tf.Graph().as_default():
    initializer = tf.random_unifrom_initializer(-args.init_scale, args.init_scale)

  # training model
  with tf.name_scope('train'):
    train_args = copy.deepcopy(args)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
      train_args.mode = default_training_mode
      train_model = S2VT(para=train_args)
  
  # testing model
  with tf.name_scope('test'):
    test_args = copy.deepcopy(args)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
      test_args.mode = default_testing_mode
      test_args.batch_size = 1
      test_model = S2VT(para=test_args)

  sv = tf.train.Supervisor(logdir='./jason/logs/')
  with sv.managed_session() as sess:
    # training
    for i in range(1, args.max_epoch + 1):
      train_perplexity = run_epcoh(sess, train_model, train_args)
      if i % args.info_epoch == 0:
        print('Epoch #%d  Train Perplexity: %.4f' %(i, train_perplexity))

    # testing
    results = []
    for i in range(default_testing_video_num)
      results.extend(run_epoch(sess, test_model, test_args))
    print(results)
  
  # compute BLEU score
  filenames = open('MLDS_hw2_data/testing_id.txt', 'r').read().splitlines()
  output = [{"caption": result, "id": filename}
            for result, filename in zip(results, filenames)]
  with open('./jason/output.json', 'w') as f:
    json.dump(output, f)
  os.system('python3 bleu_eval.py ./jason/output.json MLDS_hw2_data/testing_public_label.json')




















