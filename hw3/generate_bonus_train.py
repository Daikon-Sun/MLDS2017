#!/usr/bin/python3
import os
from os.path import join, isfile
import numpy as np
import argparse
from Utils import skipthoughts
import h5py
import time
import threading
from multiprocessing import Process, Queue
import multiprocessing

vector_type_name = ['uni_skip', 'bi_skip', 'combine_skip', 'one_hot',
               'glove_50', 'glove_100', 'glove_200', 'glove_300']
cpus = multiprocessing.cpu_count()
print('number of cpus = %d' % cpus)

def wanted_words(words):
  if len(words) == 0: return False
  if words[0] == 'ponytail' or words[0] == 'twintails': return True
  if len(words) == 1: return False
  if words[0] == 'long' and words[1] == 'hair': return True
  if words[0] == 'short' and words[1] == 'hair': return True
  return False

def wanted_tag(tag):
  return len(tag) == 1

def to_same(tags):
  assert len(tags) == 1
  if len(tags[0]) == 2: return tags
  return [tags[0] + ['hair']]

def get_vector(model, keys, image_captions, out_q):
  vectors = {}
  for key in keys:
    vectors[key] = skipthoughts.encode(model, image_captions[key])
  out_q.put(vectors)

def save_caption(args):

  img_dir = join(args.data_set, args.imgs_dir)
  image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]

  print('number of train images: %d' % len(image_files))

  tag_file = join(args.data_set, args.tag_name)
  tags = open(tag_file, 'r').read().splitlines()
  tags = [tag.split(',') for tag in tags]
  tags = [[tag[0], tag[1].split('\t')] for tag in tags]
  tags = [[tag[0], [words.split(':')[0].split() for words in tag[1]]]
          for tag in tags]
  tags = [[tag[0], [words for words in tag[1] if wanted_words(words)]]
           for tag in tags]
  tags = dict([[tag[0], to_same(tag[1])]
               for tag in tags if wanted_tag(tag[1])])

  if args.vector_type <= 2:
    image_captions = {}
    for key, val in tags.items():
      image_captions[str(key)+'.jpg'] = ['the girl has '+val[0][0]+' '+val[0][1]
                                        +' and '+val[1][0]+' '+val[1][1]]
    model = skipthoughts.load_model()
    encoded_captions = {}

    parallel_keys = [[] for i in range(cpus)]
    quantity = (len(image_captions)+cpus-1)//cpus

    for i, key_val in enumerate(image_captions.items()):
      parallel_keys[i//quantity].append(key_val[0])

    out_q = Queue()
    thrds =\
      [Process(target=get_vector, args=(model, parallel_keys[i],
                                        image_captions, out_q))
       for i in range(cpus)]

    for thrd in thrds: thrd.start()
    encoded_captions = {}
    for i in range(cpus):
      encoded_captions.update(out_q.get())
    for thrd in thrds: thrd.join()

  elif args.vector_type == 3:
    hair_list, eye_list = [], []
    for key, val in tags.items():
      if val[0][0] not in hair_list: hair_list.append(val[0][0])
      if val[1][0] not in eyes_list: eyes_list.append(val[1][0])

    h = h5py.File(join(data_dir, dict_file), 'w')
    h.create_group('hair')
    h.create_group('eyes')

    for key in hair_list:
      h['hair'].create_dataset(key, data=hair_list.index(key), dtype='i')
    for key in eyes_list:
      h['eyes'].create_dataset(key, data=eyes_list.index(key), dtype='i')

    h['hair'].attrs['size'] = len(hair_list)
    h['eyes'].attrs['size'] = len(eyes_list)
    h.close()

    encoded_captions = {}
    for key, val in tags.items():
      encoded_captions[str(key)+'.jpg'] = [hair_list.index(val[0][0]),
                                           eyes_list.index(val[1][0])]
      num_classes_hair = len(hair_list)
      num_classes_eyes = len(eyes_list)
      one_hot_hair =\
        np.eye(num_classes_hair)[encoded_captions[str(key)+'.jpg'][0]]
      one_hot_eyes =\
        np.eye(num_classes_eyes)[encoded_captions[str(key)+'.jpg'][1]]
      encoded_captions[str(key)+'.jpg'] = \
        np.array([np.concatenate((one_hot_hair,one_hot_eyes),axis=0)])

  elif args.vector_type == 4:
    wordvecs = open(args.glove_file, 'r').read().splitlines()
    wordvecs = [wordvec.split() for wordvec in wordvecs]
    wordvecs = dict([[wordvec[0], np.array([wordvec[1:]], dtype=np.float32)]
                      for wordvec in wordvecs])

    encoded_captions = {}
    for key, val in tags.items():
      print(val)
      if val[0][0] == 'twintails':
        encoded_captions[key+'.jpg'] =\
          (wordvecs['twin'] + wordvecs['tails'] + wordvecs[val[0][1]]*2)/4
      else:
        encoded_captions[key+'.jpg'] =\
          (wordvecs[val[0][0]] + wordvecs[val[0][1]])/2

  h = h5py.File(join(args.data_set, args.method_dir, args.out_file), 'w')
  for key in encoded_captions:
    h.create_dataset(key,data=encoded_captions[key])
  h.close()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', '-ds', type=str, default='faces',
                      help='data set')
  parser.add_argument('--method_dir', '-md', type=str, default='',
                      help='method directory')
  parser.add_argument('--tag_name', '-tn', type=str, default='tags.csv',
                      help='tags filename')
  parser.add_argument('--imgs_dir', '-id', type=str, default='imgs',
                      help='images directory')
  parser.add_argument('--vector_type', '-vt', type=int, default=4,
                      choices=range(0, 5),
                      help='method to encode captions,options: '
                           '0. uni_skip, 1. bi_skip, 2. combine_skip, '
                           '3. one_hot, 4. glove')
  parser.add_argument('--out_file', '-of', default='caption_vectors.hdf5',
                      type=str, help='output file name')
  parser.add_argument('--dict_file', '-df', default='onehot_hair_eyes.hdf5',
                      type=str, help='output dictionary name')
  parser.add_argument('--glove_file', '-gf', type=str, help='input glove file',
                      default='glove/glove.6B.300d.txt')
  args = parser.parse_args()
  if args.method_dir == '':
    print('Need to specify the method directory!')
    exit(1)
  save_caption(args)

if __name__ == '__main__':
  main()
