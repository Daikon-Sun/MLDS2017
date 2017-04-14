import os
import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def word_normalize(word):
  if any(c.isdigit() for c in word):
    return '0'
  if any(c.isalpha() for c in word):
    if not all(c.isalpha() for c in word):
      if len(word)>1 and word[0]=='\'' and word[1].isalpha() and len(word)<4:
        return word
      return ''.join(c if c.isalpha() else '' for c in word)
  return word

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Parsing given datas '
      'into the format of TFRecorder file.')
  argparser.add_argument('-f', '--file_list',
      type=str, default='MLDS_hw2_data/training_data/training_list.txt',
      help='given feature file list',)
  argparser.add_argument('-tl','--training_label',
  	  type=str, default='MLDS_hw2_data/training_label.json',
  	  help='training label with video id and captions in .json format')
  argparser.add_argument('-i', '--input_dir',
      type=str, default='MLDS_hw2_data/training_data/feat',
      help='the input directory of .npy files',)
  argparser.add_argument('-o', '--output_dir',
      type=str, default='MLDS_hw2_data/training_data/Training_Data_TFR',
      help='the output directory of TFRecorder files',)
  args = argparser.parse_args()

  with open(args.training_label) as training_label_json:
    training_label = json.load(training_label_json)
    for i in tqdm(range(len(training_label))):
      with np.load(args.input_dir+'/'+training_label[i]["id"]+'.npy') as video_array:
      	video_array_flat = np.reshpe(video_array, (1,-1))
      	writer = tf.python_io.TFRecordWriter(args.output_dir+'/'+training_label[i]["id"]+'.tfr')
        for j in range(len(training_label[i]["caption"])):
          words = word_tokenize(training_label[i]["caption"][j].lower())
          words_id = []
          for word in words:
            w = word_normalize(word)
            words_id.append(0 if w not in vocab_table else vocab_table[w])
          example = tf.train.Example(
            features=tf.train.Features(
              feature={
                'video': tf.train.Feature(
                  float_list=tf.train.FloatList(value=video_array_flat)),
                'caption': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=words_id))}))
          serialized = example.SerializeToString()
          writer.write(serialized)
