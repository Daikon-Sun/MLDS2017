import os
from os.path import join, isfile
import numpy as np
import argparse
import skipthoughts
import h5py
import time
import threading
from multiprocessing import Process
import multiprocessing

vector_name = ['uni_skip', 'bi_skip', 'combine_skip', 'one_hot',
               'glove_50', 'glove_100', 'glove_200', 'glove_300']

def wanted_words(words):
  if len(words) != 2:
    return False
  if 'hair' in words or 'eyes' in words:
    return words[0] != 'long' and words[0] != 'short'\
       and words[0] != 'bicolored' and words[0] != '11' and words[0] != 'pubic'
  return False

def wanted_tag(tag):
  if len(tag) != 2: return False
  tmp = [word for tg in tag for word in tg]
  return 'hair' in tmp and 'eyes' in tmp

def to_same(tags):
  assert len(tags) == 2
  if tags[0][1] == 'eyes' and tags[1][1] == 'hair':
    return [tags[1], tags[0]]
  elif tags[1][1] == 'eyes' and tags[0][1] == 'hair':
    return tags
  else:
    assert False

def get_vector(num, model, keys, image_captions, encoded_captions):
  for key in keys:
    print(num)
    encoded_captions[key] = skipthoughts.encode(model, image_captions[key])

def save_caption_vectors_flowers(data_set, data_dir,
                                 tag_name, vector, out_file):

  img_dir = join(data_dir, data_set)
  image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]

  print('number of train images: %d' % len(image_files))

  tag_file = join(data_dir, tag_name)
  tags = open(tag_file, 'r').read().splitlines()
  tags = [tag.split(',')[1].split('\t') for tag in tags]
  tags = [[words.split(':')[0].split() for words in tag] for tag in tags]
  tags = [[ words for words in tag if wanted_words(words)] for tag in tags]
  tags = dict([[i, to_same(tag)]
               for i, tag in enumerate(tags) if wanted_tag(tag)])

  if vector in [1,2,3]:
    image_captions = {}
    for key, val in tags.items():
      image_captions[str(key)+'.jpg'] = ['the girl has '+val[0][0]+' '+val[0][1]
                                        +' and '+val[1][0]+' '+val[1][1]]
    model = skipthoughts.load_model()
    encoded_captions = {}

    cpus = multiprocessing.cpu_count()
    print('number of cpus = %d' % cpus)
    parallel_keys = [[] for i in range(cpus)]
    quantity = (len(image_captions)+cpus-1)//cpus
    print('quantity = %d' % quantity)

    for i, key_val in enumerate(image_captions.items()):
      parallel_keys[i//quantity].append(key_val[0])

    for i in range(cpus):
      print(len(parallel_keys[i]))

    thrds =\
      [Process(target=get_vector, args=(i, model, parallel_keys[i],
                                        image_captions, encoded_captions,))
       for i in range(cpus)]

    for i in range(cpus):
      thrds[i].start()
    for i in range(cpus):
      thrds[i].join()

    h = h5py.File(join(data_dir, out_file), 'w')
    for key in encoded_captions:
      h.create_dataset(key, data=encoded_captions[key])
    h.close()

  elif vector == 4:
    hair_list, eye_list = [], []
    for key, val in tags.items():
      if val[0][0] not in hair_list: hair_list.append(val[0][0])
      if val[1][0] not in eyes_list: eyes_list.append(val[1][0])

    h = h5py.File(join(data_dir, out_file), 'w')
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

    h = h5py.File(join(data_dir,vector_name[vector],'train_vector.hdf5'), 'w')
    for key in encoded_captions:
      h.create_dataset(key,data=encoded_captions[key])
    h.close()

  elif vector==5:
    h = h5py.File(join(data_dir, vector_name[vector],'glove.6B.'
                       +vector_name[vector][6:]+'d.hdf5'), 'r')
    encoded_captions = {}
    for key, val in tags.items():
      encoded_captions[str(key)+'.jpg'] =\
        np.concatenate((h['__'+val[0][0]+'__'].value,
                        h['__'+val[1][0]+'__'].value),axis=0)

    h = h5py.File(join(data_dir,vector_name[vector],'train_vector.hdf5'), 'w')
    for key in encoded_captions:
      h.create_dataset(key,data=encoded_captions[key])
    h.close()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', '-dd', type=str, default='hw3_data',
                      help='Data directory')
  parser.add_argument('--data_set', '-ds', type=str, default='faces',
                      help='Data Set : faces')
  parser.add_argument('--tag_name', '-tn', type=str, default='tags_clean.csv',
                      help='tags')
  parser.add_argument('--vector', '-v', type=int, default=2,
                      choices=range(0, 8),
                      help='method to encode captions,options: '
                           '0. uni_skip, 1. bi_skip, 2. combine_skip, '
                           '3. one_hot, 4. glove_50 ,5. glove_100 '
                           '6. glove_200 7. glove_300')
  parser.add_argument('--out_file', '-of', default='train_vector.hdf5',
                      type=str, help='output file name')
  args = parser.parse_args()
  save_caption_vectors_flowers(args.data_set, args.data_dir,
                               args.tag_name, args.vector, args.out_file)

if __name__ == '__main__':
  main()
