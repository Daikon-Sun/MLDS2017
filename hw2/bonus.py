import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
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
default_dropout_keep_prob     = 0.5  # for dropout layer
default_learning_rate         = 0.0001
default_learning_rate_decay_factor = 1

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
    video, caption, video_len, caption_len = self.get_single_example(para)
    videos, captions, video_lens, caption_lens = tf.train.batch([video, caption, video_len, caption_len],
      batch_size=para.batch_size, dynamic_pad=True)

    # sparse tensor cannot be sliced
    target_captions = tf.sparse_tensor_to_dense(captions)

    # video and word embeddings
    with tf.variable_scope('word_embedding'):
      word_embedding_w = tf.get_variable('word_embed',
        [para.vocab_size, para.embedding_dimension])

    with tf.variable_scope('video_embedding'):
      video_embedding_W = tf.get_variable('video_embed',
        [para.video_dimension, para.embedding_dimension])

    # embed videos and captions
    embed_video_inputs = tf.matmul(videos, video_embedding_W)
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



  # ======================== end of __init__ ======================== #

  def is_train(self): return self._para.mode == 0
  def is_valid(self): return self._para.mode == 1
  def  is_test(self): return self._para.mode == 2

  def get_single_example(self, para):
    # first construct a queue containing a list of filenames.
    filename_queue = tf.train.string_input_producer([para.filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'video': tf.FixedLenFeature([para.video_frame_num*para.video_dimension], tf.float32),
        'caption': tf.VarLenFeature(tf.int64)
      })
    video = tf.reshape(feature['video'], [para.video_frame_num, para.video_dimension])
    caption = features['caption']
    return video, caption, tf.shape(video)[0], tf.shape(caption)[0]

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
default_dropout_keep_prob     = 0.5  # for dropout layer
default_learning_rate         = 0.0001
default_learning_rate_decay_factor = 1

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
  argparser.add_argument('-vocab_size', '--vocab_size',
    type=int, default=default_vocab_size,
    help='vocab size (default:%d)' %default_vocab_size)
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
  args = argparser.parse_args()

