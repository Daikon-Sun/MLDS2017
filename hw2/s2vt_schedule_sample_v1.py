#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import copy
import json
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.seq2seq import sequence_loss as sequence_loss

default_rnn_cell_type         = 2    # 0: BsicRNN, 1: BasicLSTM, 2: FullLSTM, 3: GRU
default_video_dimension       = 4096 # dimension of each frame
default_video_frame_num       = 80   # each video has fixed 80 frames
default_vocab_size            = 6089
default_max_caption_length    = 20
default_embedding_dimension   = 500  # embedding dimension for video and vocab
default_hidden_units          = 1000 # according to paper
default_batch_size            = 290
default_layer_number          = 1
default_max_gradient_norm     = 10
default_dropout_keep_prob     = 0.7    # for dropout layer
default_init_scale            = 0.01 # for tensorflow initializer
default_max_epoch             = 10000
default_info_epoch            = 10
default_testing_video_num     = 50     # number of testing videos
default_video_step            = 4
default_schedule_sample_porb  = 0.7
default_learning_rate         = 0.001
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
        return tf.contrib.rnn.BasicRNNCell(para.hidden_units)
      elif para.rnn_cell_type == 1:
        return tf.contrib.rnn.BasicLSTMCell(para.hidden_units)
      elif para.rnn_cell_type == 2:
        return tf.contrib.rnn.LSTMCell(para.hidden_units, use_peepholes=True)
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
    with tf.variable_scope('layer_1'):
      layer_1_cell = tf.contrib.rnn.MultiRNNCell(
        [rnn_cell() for _ in range(para.layer_number)])
    with tf.variable_scope('layer_2'):
      layer_2_cell = tf.contrib.rnn.MultiRNNCell(
        [rnn_cell() for _ in range(para.layer_number)])

    # get data in batches
    if self.is_train():
      video, caption, video_len, caption_len = self.get_single_example(para)
      videos, captions, video_lens, caption_lens = tf.train.batch([video, caption, video_len, caption_len],
        batch_size=para.batch_size, dynamic_pad=True)
      
      # construct caption mask for loss computing
      caption_lens = tf.to_int32(caption_lens)
      caption_lens_reshape = tf.reshape(caption_lens, [-1]) # reshape to 1D
      max_len = tf.reduce_max(caption_lens)
      caption_mask = tf.sequence_mask(caption_lens_reshape-1, max_len-1, dtype=tf.float32)
      
      # sparse tensor cannot be sliced
      target_captions = tf.sparse_tensor_to_dense(captions)
      target_captions_input  = target_captions[:,  :-1] # start from <BOS>
      target_captions_output = target_captions[:, 1:  ] # end by <EOS>
    else:
      video, video_len = self.get_single_example(para)
      videos, video_lens = tf.train.batch([video, video_len],
        batch_size=para.batch_size, dynamic_pad=True)
      max_len = para.max_caption_length

    # video and word embeddings as well as word decoding
    with tf.variable_scope('word_embedding'):
      word_embedding_w = tf.get_variable('word_embed',
        [para.vocab_size, para.embedding_dimension])

    with tf.variable_scope('video_embedding'):
      video_embedding_w = tf.get_variable('video_embed',
        [para.video_dimension, para.embedding_dimension])

    with tf.variable_scope('word_decoding'):
      word_decoding_w = tf.get_variable('word_decode',
        [para.hidden_units, para.vocab_size])

    # embed videos
    video_flat = tf.reshape(videos, [-1, para.video_dimension])
    embed_video_inputs = tf.matmul(video_flat, video_embedding_w)
    embed_video_inputs = tf.reshape(embed_video_inputs,
      [para.batch_size, para.video_frame_num//para.video_step, para.embedding_dimension])

    # apply dropout to inputs
    if self.is_train() and para.dropout_keep_prob < 1:
      embed_video_inputs = tf.nn.dropout(embed_video_inputs, para.dropout_keep_prob)

    # initialize cost
    cost = tf.constant(0.0)

    # paddings for 1st and 2nd layers
    layer_1_padding = tf.zeros([para.batch_size, max_len-1, para.embedding_dimension])
    layer_2_padding = tf.zeros([para.batch_size, para.video_frame_num//para.video_step, para.embedding_dimension])
    
    # preparing sequence length
    video_frame_num = tf.constant(para.video_frame_num//para.video_step, dtype=tf.int32,
                                  shape=[para.batch_size])
    if not self.is_test():
      sequence_length = tf.add(video_frame_num, caption_lens-1)
    else:
      sequence_length = tf.add(video_frame_num, max_len-1)
    # reshape for rnn
    sequence_length = tf.reshape(sequence_length, [-1])

    # =================== layer 1 ===================
    layer_1_inputs = tf.concat([embed_video_inputs, layer_1_padding], 1)
    with tf.variable_scope('layer_1'):
      layer_1_outputs, layer_1_final_state = tf.nn.dynamic_rnn(layer_1_cell,
                                                  layer_1_inputs,
                                                  sequence_length=sequence_length,
                                                  dtype=tf.float32)
    
    # =================== layer 2 ===================
    if self.is_train():
      caption_embed = tf.nn.embedding_lookup(word_embedding_w, target_captions_input)
      layer_2_pad_and_embed = tf.concat([layer_2_padding, caption_embed], 1)
      layer_2_inputs = tf.concat([layer_2_pad_and_embed, layer_1_outputs], 2)
      layer_1_outputs = tf.transpose(layer_1_outputs, perm=[1,0,2])
      layer_1_outputs_ta = tf.TensorArray(dtype=tf.float32,
                                          size=para.video_frame_num//para.video_step+max_len-1)
      layer_1_outputs_ta = layer_1_outputs_ta.unstack(layer_1_outputs)
    else:
      layer_2_inputs = layer_1_outputs

    layer_2_inputs = tf.transpose(layer_2_inputs, perm=[1,0,2]) # for time major unstack
    layer_2_inputs_ta = tf.TensorArray(dtype=tf.float32,
                                       size=para.video_frame_num//para.video_step+max_len-1)
    layer_2_inputs_ta = layer_2_inputs_ta.unstack(layer_2_inputs)
    
    rand = tf.random_uniform([para.video_frame_num//para.video_step+max_len-1], dtype=tf.float32)
    rand_ta = tf.TensorArray(dtype=tf.float32,
                             size=para.video_frame_num//para.video_step+max_len-1)
    rand_ta = rand_ta.unstack(rand)
    schedule_sample_porb = tf.constant(para.schedule_sample_porb, dtype=tf.float32, shape=[1,1])

    if self.is_train():
      def layer_2_loop_fn(time, cell_output, cell_state, loop_state):
        def input_fn():
          def normal_feed_in():
            return layer_2_inputs_ta.read(time)
          def schedule_sample():
            if cell_output is None:
              return tf.zeros([para.batch_size, para.embedding_dimension+para.hidden_units], dtype=tf.float32)
            def feed_previous():
              output_logit = tf.matmul(cell_output, word_decoding_w)
              prediction = tf.argmax(output_logit, axis=1)
              prediction_embed = tf.nn.embedding_lookup(word_embedding_w, prediction)
              next_input = tf.concat([prediction_embed, layer_1_outputs_ta.read(time)], 1)
              return next_input
            sample = (schedule_sample_porb[0,0] < rand_ta.read(time))
            sample = tf.reduce_all(sample)
            return tf.cond(sample, feed_previous, normal_feed_in)
          start_decoding = (time >= video_frame_num+1) # first input should keep to be <BOS>
          start_decoding = tf.reduce_all(start_decoding)
          return tf.cond(start_decoding, schedule_sample, normal_feed_in)

        def zeros():
          return tf.zeros([para.batch_size, para.embedding_dimension+para.hidden_units], dtype=tf.float32)

        emit_output = cell_output
        if cell_output is None: # time == 0
          next_cell_state = layer_2_cell.zero_state(para.batch_size, dtype=tf.float32)
        else:
          next_cell_state = cell_state
        is_finished = (time >= sequence_length)
        finished = tf.reduce_all(is_finished)
        next_input = tf.cond(finished, zeros, input_fn)#layer_2_inputs_ta.read(time))
        return (is_finished, next_input, next_cell_state, emit_output, loop_state)
    else:
      def layer_2_loop_fn(time, cell_output, cell_state, loop_state):
        def encode_input():
          layer_2_inputs = layer_2_inputs_ta.read(time)
          padding = tf.zeros([para.batch_size, para.embedding_dimension], dtype=tf.float32)
          return tf.concat([padding, layer_2_inputs], 1)

        def decode_input():
          if cell_output is None:
            return tf.zeros([para.batch_size, para.embedding_dimension+para.hidden_units], dtype=tf.float32)
          else:
            def is_begin():
              begin_of_sentence = tf.ones([para.batch_size, para.embedding_dimension], dtype=tf.float32)
              next_input = tf.concat([begin_of_sentence, layer_2_inputs_ta.read(time)], 1)
              return next_input
            def not_begin():
              output_logit = tf.matmul(cell_output, word_decoding_w)
              prediction = tf.argmax(output_logit, axis=1)
              prediction_embed = tf.nn.embedding_lookup(word_embedding_w, prediction)
              next_input = tf.concat([prediction_embed, layer_2_inputs_ta.read(time)], 1)
              return next_input
            begin = tf.equal(time,video_frame_num)
            begin = tf.reduce_all(begin)
            next_input = tf.cond(begin, is_begin, not_begin)
            return next_input

        emit_output = cell_output
        if cell_output is None: # time == 0
          next_cell_state = layer_2_cell.zero_state(para.batch_size, dtype=tf.float32)
        else:
          next_cell_state = cell_state
        all_finished = (time >= (sequence_length-1))
        start_decoding = (time >= video_frame_num)
        start_decoding = tf.reduce_all(start_decoding)
        next_input = tf.cond(start_decoding, decode_input, encode_input)

        return (all_finished, next_input, next_cell_state, emit_output, loop_state)

    layer_2_outputs_ta, layer_2_final_state, _ = tf.nn.raw_rnn(layer_2_cell, layer_2_loop_fn)
    layer_2_outputs = layer_2_outputs_ta.stack()
    layer_2_outputs = layer_2_outputs[para.video_frame_num//para.video_step:, :, :]
    layer_2_outputs = tf.transpose(layer_2_outputs, perm=[1,0,2]) # batch_size x time x embed_dim

    if self.is_train():
      layer_2_outputs = tf.reshape(layer_2_outputs, [-1, para.hidden_units])
      layer_2_output_logit = tf.matmul(layer_2_outputs, word_decoding_w)
      layer_2_output_logit = tf.reshape(layer_2_output_logit,
                               [para.batch_size, max_len-1, para.vocab_size])
      self._prob = tf.nn.softmax(layer_2_output_logit)

      loss = sequence_loss(layer_2_output_logit, target_captions_output, caption_mask)
      self._cost = cost = tf.reduce_mean(loss)

      # clip gradient norm
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                   para.max_gradient_norm)
      optimizer  = default_optimizers[para.optimizer_type](para.learning_rate)
      self._eval = optimizer.apply_gradients(zip(grads, tvars),
                     global_step=tf.contrib.framework.get_or_create_global_step())
    else:
      layer_2_outputs = tf.reshape(layer_2_outputs, [-1, para.hidden_units])
      layer_2_output_logit = tf.matmul(layer_2_outputs, word_decoding_w)
      layer_2_output_logit = tf.reshape(layer_2_output_logit, [para.batch_size, -1, para.vocab_size])
      self._prob = tf.nn.softmax(layer_2_output_logit)

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
  def val1(self):  return self._val1
  @property
  def val2(self):  return self._val2

  def get_single_example(self, para):
    if self.is_train():
      file_list_path = 'MLDS_hw2_data/training_data/Training_Data_TFR/training_list.txt'
      filenames = open(file_list_path).read().splitlines()
      files = ['MLDS_hw2_data/training_data/Training_Data_TFR/'+filename for filename in filenames]
      file_queue = tf.train.string_input_producer(files, shuffle=True)
    else:
      file_list_path = 'MLDS_hw2_data/testing_data/Testing_Data_TFR/testing_list.txt'
      filenames = open(file_list_path).read().splitlines()
      files = ['MLDS_hw2_data/testing_data/Testing_Data_TFR/'+filename for filename in filenames]
      file_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    if self.is_train():
      features = tf.parse_single_example(
        serialized_example,
        features={
          'video': tf.FixedLenFeature([para.video_frame_num*para.video_dimension], tf.float32),
          'caption': tf.VarLenFeature(tf.int64),
        })
      video = tf.reshape(features['video'], [-1, para.video_dimension])[::para.video_step]
      caption = features['caption']
      return video, caption, tf.shape(video)[0], tf.shape(caption)[0]
    else:
      features = tf.parse_single_example(
        serialized_example,
        features={
          'video': tf.FixedLenFeature([para.video_frame_num*para.video_dimension], tf.float32)
        })
      video = tf.reshape(features['video'], [-1, para.video_dimension])[::para.video_step]
      return video, tf.shape(video)[0]

def run_epoch(sess, model, args):
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
      ans = []
      for j in range(prob.shape[1]):
        mx_id = np.argmax(prob[i, j, :])
        if mx_id == EOS:
          break
        ans.append(vocab_dictionary[str(mx_id)])
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
    help='maximum gradient norm (default:%d' %default_max_gradient_norm)
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
    help='initialization scale for tensorflow initializer (default:%d)' %default_init_scale)
  argparser.add_argument('-me', '--max_epoch',
    type=int, default=default_max_epoch,
    help='maximum training epoch (default:%d' %default_max_epoch)
  argparser.add_argument('-ie', '--info_epoch',
    type=int, default=default_info_epoch,
    help='show training information for each (default:%d) epochs' %default_info_epoch)
  argparser.add_argument('-vs', '--video_step',
    type=int, default=default_video_step,
    help='Choose a frame per step. (default:%d)' %default_video_step)
  argparser.add_argument('-ss', '--schedule_sample_porb',
    type=float, default=default_schedule_sample_porb,
    help='scheduled sampling probability. (default:%d)' %default_schedule_sample_porb)
  args = argparser.parse_args()


  print('S2VT start...\n')

  print('Loading vocab dictionary...\n')
  vocab_dictionary_path = 'MLDS_hw2_data/training_data/jason_reverse_vocab.json'
  with open(vocab_dictionary_path) as vocab_dictionary_json:
    vocab_dictionary = json.load(vocab_dictionary_json)
  args.vocab_size = len(vocab_dictionary)
  print('vocab_size = %d' %args.vocab_size)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)

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

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    sv = tf.train.Supervisor(logdir='logs_schedule_sample/')
    with sv.managed_session(config=config) as sess:
      # training
      for i in range(1, args.max_epoch + 1):
        train_perplexity = run_epoch(sess, train_model, train_args)
        if i % args.info_epoch == 0:
          print('Epoch #%d  Train Perplexity: %.4f' %(i, train_perplexity))

      # testing
      results = []
      for i in range(default_testing_video_num):
        results.extend(run_epoch(sess, test_model, test_args))
      results = [ ' '.join(result[:-1]) for result in results ]
      for result in results: print(result)

    # compute BLEU score
    filenames = open('MLDS_hw2_data/testing_id.txt', 'r').read().splitlines()
    output = [{"caption": result, "id": filename}
              for result, filename in zip(results, filenames)]
    with open('output.json', 'w') as f:
      json.dump(output, f)
