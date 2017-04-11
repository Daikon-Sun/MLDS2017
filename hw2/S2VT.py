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
# default_vocab_size          
default_hidden_units          = 1000 # according to paper
default_batch_size            = 
default_layer_num             = 2 # according to paper
default_max_gradient_norm     = 10
default_learning_rate         = 0.0001
default_learning_rate_decay_factor = 1

# not implemented yet: ( for seq2seq only )

# for large vocab output: use sampled softmax => output prpjection is needed
# CAUTION!! sampled softmax is different from scheduled softmax sampling!
# scheduled softmax sampling is a requirement in assignment

default_output_projection     = None
default_softmax_loss_function = None

class S2VT(object):

  def __init__(self,
  	           rnn_cell_type
  	           image_dimension,
               image_frame_num,
  	           vocab_size,
  	           hidden_units,
  	           batch_size,
  	           layer_num,
  	           max_gradient_norm,
  	           learning_rate,
  	           learning_rate_decay_factor,
               vocab_size,
               embedding_dimension,
  	           output_projection,
  	           softmax_loss_function,
  	           dtype=tf.float32):

    self.image_dimension = image_dimension
    self.image_frame_num = image_frame_num
    self.hidden_units = hidden_units
    self.batch_size = batch_size
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
    

    # using pretrained word embedding

    w_emb = tf.Variable(tf.constant(0.0,
              shape=[vocab_size, embedding_dimension]),
              trainable=False,
              name='w_emb')
    self.word_embedding = tf.placeholder(tf.float32, [vocab_size, embedding_dimension])
    self.word_embedding_init = w_emb.assign(self.word_embedding)


    # two-layer-rnn model according to paper

    self.cell_1 = single_cell()
    self.cell_2 = single_cell()


    # encoding variable for each frame

    #self.image_encoding_w = tf.Variable(tf.random_uniform([image_dimension, hidden_units], -0.1, 0.1), name='image_encoding_w')
    #self.image_encoding_b = tf.Variable(tf.zeros([hidden_units]), name='image_encoding_b')
    self.image_encoding_w = tf.get_variable("image_encoding_w",
                              [image_dimension, hidden_units],
                              initializer=tf.random_normal_initializer(-0.1,0.1))
    self.image_encoding_b = tf.get_variable("image_encoding_b",
                              hidden_units,
                              initializer=tf.constant_initializer(0))


    # decoding variable for each word

    #self.word_decoding_w = tf.Variable(tf.random_uniform([hidden_units, vocab_size], -0.1,0.1), name='word_decoding_w')
    #self.word_decoding_b = tf.Variable(tf.zeros([vocab_size]), name='word_decoding_b')
    self.word_decoding_w = tf.get_variable("word_decoding_w",
                              [hidden_units, vocab_size],
                              initializer=tf.random_normal_initializer(-0.1,0.1))
    self.word_decoding_b = tf.get_variable("word_decoding_b",
                              vocab_size,
                              initializer=tf.constant_initializer(0))
  # end of __init__

    def model(self):
      

    # #load in pre-trained word-embedding
    sess.run(train_model._embed_init,
             feed_dict={train_model._embedding: wordvec})
    sess.run(test_model._embed_init,
             feed_dict={test_model._embedding: wordvec})










    #

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

















