#!/usr/bin/python3

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys

# default values

default_rnn_cell_type         = 3 # Full LSTM with peephole
default_image_dimension       = [80, 4096] # dimension of each frame
default_vocab_size            = 
default_hidden_units          = 1000 # according to paper
default_batch_size            = 
default_layer_num             = 2 # according to paper
default_max_gradient_norm     = 10
default_learning_rate         = 0.001
default_learning_rate_decay_factor = 1

# not implemented yet:

# for large vocab output: use sampled softmax => output prpjection is needed
# CAUTION!! sampled softmax is different from scheduled softmax sampling!
# scheduled softmax sampling is a requirement in assignment

default_output_projection     = None
default_softmax_loss_function = None

class S2VT(object):

  def __init__(self,
  	           rnn_cell_type
  	           image_dimension,
  	           vocab_size,
  	           hidden_units,
  	           batch_size,
  	           layer_num,
  	           max_gradient_norm,
  	           learning_rate,
  	           learning_rate_decay_factor,
  	           output_projection,
  	           softmax_loss_function,
  	           dtype=tf.float32):

    self.image_dimension = image_dimension
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)

    def single_cell();
      if rnn_cell_type == 0:
        return tf.contrib.rnn.BasicRNNCell(hidden_units, activation=tf.tanh)
      elif rnn_cell_type == 1:
        return tf.contrib.rnn.BasicLSTMCell(hidden_units)
      elif rnn_cell_type == 2:
        return tf.contrib.rnn.LSTMCell(hidden_units, use_peepholes=True)
      elif rnn_cell_type == 3:
        return tf.contrib.rnn.GRUCell(hidden_units)

    cell = single_cell()
    if layer_num > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(layer_num)])

    output_projection = default_output_projection
    softmax_loss_function = default_softmax_loss_function

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















