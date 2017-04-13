#!/usr/bin/python3

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import os
import sys

# default values

default_rnn_cell_type         = 1
default_image_dimension       = 4096 # dimension of each frame
default_image_frame_num       = 80 # each video has fixed 80 frames          
default_hidden_units          = 1000 # according to paper
default_batch_size            = 50 
default_layer_num             = 2 # according to paper
default_max_gradient_norm     = 10
default_dropout_keep_prob     = 0.5 # for dropout layer
default_learning_rate         = 0.0001
default_learning_rate_decay_factor = 1

# not implemented yet: ( for seq2seq only )

# for large vocab output: use sampled softmax => output prpjection is needed
# CAUTION!! sampled softmax is different from scheduled softmax sampling!
# scheduled softmax sampling is a requirement in assignment

default_output_projection     = None
default_softmax_loss_function = None


# not implmented yet
def get_one_caption():
  # randomly pick a caption from a video
  # should ad <BOS> to the beginning
  return one_caption, length_of_caption


class S2VT(object):

  def __init__(self,
               running_mode, # training or testing
  	           rnn_cell_type,
               optimizer_type,
  	           image_dimension,
               image_frame_num,
  	           vocab_size,
               embedding_dimension,
  	           hidden_units,
  	           batch_size,
  	           layer_num,
  	           max_gradient_norm,
               dropout_keep_prob,
  	           learning_rate,
  	           learning_rate_decay_factor,
  	           output_projection,
  	           softmax_loss_function,
  	           dtype=tf.float32):

    self.optimizer_type = optimizer_type
    self.image_dimension = image_dimension
    self.image_frame_num = image_frame_num
    self.hidden_units = hidden_units
    self.batch_size = batch_size
    self.max_gradient_norm = max_gradient_norm
    self.dropout_keep_prob = dropout_keep_prob
    self.learning_rate = learning_rate
    self.vocab_size = vocab_size
    self.embedding_dimension = embedding_dimension

    # This is for varying learning rate

    # self.learning_rate = tf.Variable(
    #     float(learning_rate), trainable=False, dtype=dtype)
    # self.learning_rate_decay_op = self.learning_rate.assign(
    #     self.learning_rate * learning_rate_decay_factor)


    # define single cell based on default parameters
    def single_cell():
      if rnn_cell_type == 0:
        return tf.contrib.rnn.BasicRNNCell(hidden_units, activation=tf.tanh)
      elif rnn_cell_type == 1:
        return tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple=False)
      elif rnn_cell_type == 2:
        return tf.contrib.rnn.LSTMCell(hidden_units, use_peepholes=True, state_is_tuple=False)
      elif rnn_cell_type == 3:
        return tf.contrib.rnn.GRUCell(hidden_units)
    
    # embed words to a lower 500 dimension space according to original paper
    self.embed_word_matrix = tf.get_variable("embed_word_matrix",
                               [vocab_size, hidden_units/2]
                               initializer=tf.random_uniform_initializer(-0.1,0.1))

    # two-layer-rnn model according to paper
    self.cell_1 = single_cell()
    self.cell_2 = single_cell()

    # encoding variable for each frame
    #self.image_encoding_w = tf.Variable(tf.random_uniform([image_dimension, hidden_units], -0.1, 0.1), name='image_encoding_w')
    #self.image_encoding_b = tf.Variable(tf.zeros([hidden_units]), name='image_encoding_b')
    self.image_encoding_w = tf.get_variable("image_encoding_w",
                              [image_dimension, hidden_units],
                              initializer=tf.random_uniform_initializer(-0.1,0.1))
    self.image_encoding_b = tf.get_variable("image_encoding_b",
                              hidden_units,
                              initializer=tf.constant_initializer(0))

    # decoding variable for each word
    #self.word_decoding_w = tf.Variable(tf.random_uniform([hidden_units, vocab_size], -0.1,0.1), name='word_decoding_w')
    #self.word_decoding_b = tf.Variable(tf.zeros([vocab_size]), name='word_decoding_b')
    self.word_decoding_w = tf.get_variable("word_decoding_w",
                              [hidden_units, vocab_size],
                              initializer=tf.random_uniform_initializer(-0.1,0.1))
    self.word_decoding_b = tf.get_variable("word_decoding_b",
                              vocab_size,
                              initializer=tf.constant_initializer(0))
  # end of __init__

  def model(self):

    video = tf.placeholder(tf.float32, [self.batch_size, self.image_frame_num, self.image_dimension])
    video_flat = tf.reshape(video, [-1, self.image_dimension])
    video_input = tf.nn.xw_plus_b(video_flat, self.image_encoding_w, self.image_encoding_b)
    video_input = tf.reshape(video_input, [self.batch_size, self.image_frame_num, self.hidden_units])

    one_caption, caption_length = get_one_caption()
    batch, caption_length = tf.train.batch([one_caption, caption_length],
      batch_size=batch_size, dynamic_pad=True)
    # sparse tensor cannot be sliced
    batch = tf.sparse_tensor_to_dense(batch)
    # caption_len is for dynamic_rnn
    caption_length = tf.to_int32(caption_length)


    state_layer_1 = tf.zeros([self.batch_size, self.cell_1.state_size])
    state_layer_2 = tf.zeros([self.batch_size, self.cell_2.state_size])
    pad = tf.zeros([self.batch_size, self.hidden_units])

    probability = []
    loss = 0

    #================== encoding ==================#

    for i in range(0,self.image_frame_num):
      with tf.variable_scope("layer_1"):
        tf.get_variable_scope().reuse_variables()
        output_1, state_layer_1 = self.cell_1(video_input[:,i,:],state_layer_1)
      with tf.variable_scope("layer_2"):
        tf.get_variable_scope().reuse_variables()
        output_2, state_layer_2 = self.cell_2(tf.concat(1, [pad, state_layer_1]), state_layer_2)

    #================== decoding ==================#  

    for i in range(0, caption_length):
      caption_embed = tf.nn.embedding_lookup(self.embed_word_matrix, one_caption[:,i])
      with tf.variable_scope("layer_1"):
        tf.get_variable_scope().reuse_variables()
        output_1, state_layer_1 = self.cell_1(pad, state_layer_1)

      # scheduled softmax sampling is not implemented yet
      with tf.variable_scope("layer_2"):
        tf.get_variable_scope().reuse_variables()
        output_2, state_layer_2 = self.cell_2(tf.concat(1,caption_embed, state_layer_1), state_layer_2)

      labels = tf.expand_dims(one_caption[:, i+1], 1)
      indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
      indices_labels = tf.concat(1, [indices, labels])
      correct_ans = tf.sparse_tensor_to_dense(indices_labels, tf.pack([self.batch_size, self.vocab_size]), 1.0, 0.0)

      predict_ans = tf.nn.xw_plus_b(output_2, self.word_decoding_w, self.word_decoding_b)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predict_ans, correct_ans)
      probability.append(predict_ans)

      loss = loss + tf.reduce_sum(cross_entropy)/self.batch_size


    #if validation or testing, exit here
    if running_mode != 0:
      return loss, video, caption, probability

    #clip global gradient norm
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.max_gradient_norm)
    optimizer = optimizers[self.optimizer_type](self.learning_rate)
    evaluate = optimizer.apply_gradients(zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step())

    return loss, video, caption, probability, evaluate

  

    # This is for seq2seq
    # if layer_num > 1:
    #   cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(layer_num)])

    # output_projection = default_output_projection
    # softmax_loss_function = default_softmax_loss_function

    # 我们将feed_previous设置为False。这意味着解码器将使用提供的decode_inputs张量。
    # 如果我们将feed_previous设置为True，解码器将只使用decoder_inputs的第一个元素，
    # 来自该列表的所有其他张量将被忽略，并且将使用解码器的先前输出。

    # This is for seq2seq model ( not implemented yet)
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=image_dimension(0)*image_dimension(1),
          num_decoder_symbols=vocab_size,
          embedding_size=hidden_units,
          output_projection=output_projection,
          feed_previous=do_decode, # adjust here to implement scheduled softmax sampling
          dtype=dtype)

















