import os, sys
import argparse
import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Parsing given datas '
      'into the format of TFRecorder file.')
  argparser.add_argument('-v', '--vocab_file',
      type=str, default='MLDS_hw2_data/training_data/jason_vocab.json',
      help='output dictionary of the table for vocabs',)
  argparser.add_argument('-l','--training_label',
  	  type=str, default='MLDS_hw2_data/training_label.json',
  	  help='training label with video id and captions in .json format')
  argparser.add_argument('-i', '--input_dir',
      type=str, default='MLDS_hw2_data/training_data/feat',
      help='the input directory of training .npy files',)
  argparser.add_argument('-o', '--output_dir',
      type=str, default='MLDS_hw2_data/training_data/Training_Data_TFR',
      help='the output directory of training TFRecorder files',)
  argparser.add_argument('-tid', '--testing_id',
      type=str, default='MLDS_hw2_data/testing_id.txt',
      help='testing id of testing data',)
  argparser.add_argument('-ti', '--testing_input_dir',
      type=str, default='MLDS_hw2_data/testing_data/feat',
      help='the input directory of testing .npy files',)
  argparser.add_argument('-to', '--testing_output_dir',
      type=str, default='MLDS_hw2_data/testing_data/Testing_Data_TFR',
      help='the output directoy of testing TFRecorder files')
  argparser.add_argument('-c', '--convert',
      help='convert testing .npy in .tfr only without training data',
      action='store_true')
  args = argparser.parse_args()

  if args.convert:
    sys.stderr.write('start converting testing data into TFR format...\n')
    with open(args.testing_id) as testing_id:
      for file_name in tqdm(testing_id):
        video_array = np.load(args.testing_input_dir+'/'+file_name[:-1]+'.npy')
        video_array_flat = np.reshape(video_array, 80*4096)
        writer = tf.python_io.TFRecordWriter(args.testing_output_dir+'/'+file_name[:-1]+'.tfr')
        example = tf.train.Example(
            features=tf.train.Features(
              feature={
                'video': tf.train.Feature(
                  float_list=tf.train.FloatList(value=video_array_flat))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
    exit()

  # default value for special vocabs
  PAD = 0
  BOS = 1
  EOS = 2
  UNK = 3

  # dictionary initialize
  vocab_table = dict()
  vocab_table['<PAD>'] = PAD
  vocab_table['<BOS>'] = BOS
  vocab_table['<EOS>'] = EOS
  vocab_table['<UNK>'] = UNK
  index = 4

  with open(args.training_label) as training_label_json:
    training_label = json.load(training_label_json)

    sys.stderr.write('start building vocab dictionary...\n')
    for i in tqdm(range(len(training_label))):
      for j in range(len(training_label[i]["caption"])):
      	words = word_tokenize(training_label[i]["caption"][j].lower())
      	for w in words:
      	  if w in vocab_table:
      	    continue;
      	  else:
      	  	vocab_table[w] = index
      	  	index = index + 1;
    with open(args.vocab_file, 'w') as vocab_file:
      json.dump(vocab_table, vocab_file)

    sys.stderr.write('start converting training data into TFR format...\n')
    for i in tqdm(range(len(training_label))):
      video_array = np.load(args.input_dir+'/'+training_label[i]["id"]+'.npy')
      video_array_flat = np.reshape(video_array, 80*4096)
      writer = tf.python_io.TFRecordWriter(args.output_dir+'/'+training_label[i]["id"]+'.tfr')
      for j in range(len(training_label[i]["caption"])):
        words = word_tokenize(training_label[i]["caption"][j].lower())
        words_id = []
        words_id.append(BOS)
        for w in words:
          words_id.append(UNK if w not in vocab_table else vocab_table[w])
        words_id.append(EOS)
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'video': tf.train.Feature(
                float_list=tf.train.FloatList(value=video_array_flat)),
              'caption': tf.train.Feature(
                int64_list=tf.train.Int64List(value=words_id))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
