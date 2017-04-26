#!/usr/bin/python3
import os, sys, argparse, json, re
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec

def normalize(sent):
  s = sent.lower()
  for deli in ['\'','.','?','!',',',';',':','\"', '(', ')']:
    s = re.sub('['+deli+']', ' '+deli, s)
  return '<bos> ' + ' '.join(s.split()) + ' <eos>'

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Parsing given datas '
      'into the format of TFRecorder file.')
  argparser.add_argument('-v', '--vocab_file', type=str, default='vocab.json',
      help='output dictionary of the table for vocabs in .json format')
  argparser.add_argument('-l','--training_label', type=str,
      default='MLDS_hw2_data/training_label.json',
  	  help='training label with video id and captions in .json format')
  argparser.add_argument('-i', '--input_dir', type=str,
      default='MLDS_hw2_data/training_data/feat',
      help='the input directory of training .npy files')
  argparser.add_argument('-o', '--output_dir', default='train_tfrdata',
      type=str, help='the output directory of training TFRecorder files')
  argparser.add_argument('-tid', '--testing_id', type=str,
      default='MLDS_hw2_data/testing_id.txt', help='testing id of testing data')
  argparser.add_argument('-ti', '--testing_input_dir', type=str,
      default='MLDS_hw2_data/testing_data/feat',
      help='the input directory of testing .npy files')
  argparser.add_argument('-to', '--testing_output_dir', default='test_tfrdata',
      type=str, help='the output directoy of testing TFRecorder files')
  argparser.add_argument('-c', '--convert',
      help='convert testing .npy in .tfr only without training data',
      action='store_true')
  argparser.add_argument('-s', '--short',
      help='select single caption among captions for each video',
      action='store_true')
  argparser.add_argument('-d', '--dimension',
        type=int, default=0,
        help='The embedding words will be a DIMENSION-dimension '
             'real value vector.'
             '(default: %(default)s)',)
  argparser.add_argument('-ov', '--output_vector_file',
        type=str, default='vector_100d.npy',
        help='OUTPUT_VECTOR_FILE is the file storing embedding word '
             'vector in the order of embedding in the numpy array format.'
             '(default: %(default)s)',)
  args = argparser.parse_args()

  with open(args.training_label, 'r') as label_json:
    labels = json.load(label_json)
    captions = [ [ normalize(sent.lower()).split()[1:-1]
                  for sent in label['caption'] ] for label in labels ]
    sents = [ sent for caption in captions for sent in caption ]

    vocabs = set([ word for sent in sents for word in sent ])
    vocabs = ['<pad>', '<unk>', '<bos>', '<eos>'] + list(vocabs)
    dct = dict([ (word, i) for i, word in enumerate(vocabs)])

    if args.dimension > 0:
      model = word2vec.Word2Vec(sents, size=args.dimension, window=5,
                                min_count=0, workers=4)
      vecs = [ model.wv[word] if word in model else np.zeros(args.dimension)
              for word in vocabs ]
      np.save(args.output_vector_file, vecs)

    rev_dct = dict([(i, word) for word, i in dct.items()])
    with open(args.vocab_file, 'w') as vocab_file:
      json.dump(rev_dct, vocab_file, indent=2, separators=(',', ':'))

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  with open(args.training_label, 'r') as label_json:
    labels = json.load(label_json)
    for i, label in tqdm(enumerate(labels)):
      out_name = args.output_dir+'/'+label['id']+'.tfr'
      video = np.load(args.input_dir+'/'+label['id']+'.npy')
      video = video.reshape((-1, 1))
      writer = tf.python_io.TFRecordWriter(out_name)
      if args.short:
        words_len = []
        for j, sent in enumerate(label['caption']):
          words = normalize(sent).split()
          words_len.extend([len(words)])
        words_len.sort()
        median = words_len[len(words_len)//2]
        for j, sent in enumerate(label['caption']):
          if len(normalize(sent).split()) == median:
            word_ids = [ dct[word] for word in normalize(sent).split() ]
            example = tf.train.Example(
              features=tf.train.Features(
                feature={
                  'video': tf.train.Feature(
                    float_list=tf.train.FloatList(value=video)),
                  'caption': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=word_ids))}))
            serialized = example.SerializeToString()
            writer.write(serialized)
            break
      else:
        for j, sent in enumerate(label['caption']):
          word_ids = [ dct[word] for word in normalize(sent).split() ]
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

  if not os.path.exists(args.testing_output_dir):
    os.makedirs(args.testing_output_dir)
  sys.stderr.write('start converting testing data into TFR format...\n')
  with open(args.testing_id) as testing_id:
    for file_name in tqdm(testing_id.read().splitlines()):
      video_array = np.load(args.testing_input_dir+'/'+file_name+'.npy')
      video_array = np.reshape(video_array, 80*4096)
      out_file_name = args.testing_output_dir+'/'+file_name+'.tfr'
      writer = tf.python_io.TFRecordWriter(out_file_name)
      example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'video': tf.train.Feature(
                float_list=tf.train.FloatList(value=video_array))}))
      serialized = example.SerializeToString()
      writer.write(serialized)
      writer.close()
