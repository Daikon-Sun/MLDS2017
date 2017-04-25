import os, sys, argparse, json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from embedding import normalized

if __name__ == '__main__':
  argparser = argparse.ArgumentParser(description='Parsing given datas '
      'into the format of TFRecorder file.')
  argparser.add_argument('-v', '--vocab_file', type=str, default='vocab.txt',
      help='output dictionary of the table for vocabs')
  argparser.add_argument('-l','--training_label', type=str,
      default='MLDS_hw2_data/training_label.json',
  	  help='training label with video id and captions in .json format')
  argparser.add_argument('-i', '--input_dir', type=str,
      default='MLDS_hw2_data/training_data/feat',
      help='the input directory of training .npy files')
  argparser.add_argument('-o', '--output_dir', default='train_tfrdata',
      type=str, help='the output directory of training TFRecorder files')
  argparser.add_argument('-ti', '--testing_id', type=str,
      default='MLDS_hw2_data/testing_id.txt', help='testing id of testing data')
  argparser.add_argument('-fp', '--feature_path', type=str,
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
  args = argparser.parse_args()

  # convert only testing data if args.convert == true
  if args.convert:
    if not os.path.exists(args.testing_output_dir):
      os.makedirs(args.testing_output_dir)
    sys.stderr.write('start converting testing data into TFR format...\n')
    with open(args.testing_id) as testing_id:
      for file_name in tqdm(testing_id.read().splitlines()):
        video_array = np.load(args.feature_path+'/'+file_name+'.npy')
        video_array_flat = np.reshape(video_array, 80*4096)
        out_file_name = args.testing_output_dir+'/'+file_name+'.tfr'
        writer = tf.python_io.TFRecordWriter(out_file_name)
        example = tf.train.Example(
            features=tf.train.Features(
              feature={
                'video': tf.train.Feature(
                  float_list=tf.train.FloatList(value=video_array_flat))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
        writer.close()
    exit()
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  # default value for special vocabs
  PAD = 0
  UNK = 1
  BOS = 2
  EOS = 3

  # dictionary initialize
  vocab_table = dict() # word to int
  vocab_table['<PAD>'] = PAD
  vocab_table['<UNK>'] = UNK
  vocab_table['<BOS>'] = BOS
  vocab_table['<EOS>'] = EOS
  index = 4

  # reverse dictionary initialize
  reverse_vocab_table = dict() # int to word
  reverse_vocab_table[0] = '<PAD>'
  reverse_vocab_table[1] = '<UNK>'
  reverse_vocab_table[2] = '<BOS>'
  reverse_vocab_table[3] = '<EOS>'

  with open(args.training_label) as training_label_json:
    training_label = json.load(training_label_json)
    sys.stderr.write('start building vocab dictionary...\n')
    for i in tqdm(range(len(training_label))):
      for j in range(len(training_label[i]['caption'])):
        words = normalized(training_label[i]['caption'][j].lower()).split()
        for w in words:
          if w in vocab_table: continue;
          else:
            vocab_table[w] = index
            reverse_vocab_table[index] = w
            index += 1
    with open(args.vocab_file, 'w') as reverse_vocab_file:
      json.dump(reverse_vocab_table, reverse_vocab_file)

    sys.stderr.write('start converting training data into TFR format...\n')
    for i in tqdm(range(len(training_label))):
      video_array = np.load(args.input_dir+'/'+training_label[i]['id']+'.npy')
      video_array_flat = np.reshape(video_array, 80*4096)
      out_file_name = args.output_dir+'/'+training_label[i]['id']+'.tfr'
      writer = tf.python_io.TFRecordWriter(out_file_name)
      for j in range(len(training_label[i]['caption'])):
        words = normalized(training_label[i]['caption'][j].lower()).split()
        words_id = []
        counter = 1
        for w in words:
          words_id.append(UNK if w not in vocab_table else vocab_table[w])
          counter += 1
        counter += 1
        caption_length = counter
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'video': tf.train.Feature(
                float_list=tf.train.FloatList(value=video_array_flat)),
              'caption': tf.train.Feature(
                int64_list=tf.train.Int64List(value=words_id))}))
        serialized = example.SerializeToString()
        writer.write(serialized)

        if args.short: break

      writer.close()
