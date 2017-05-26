#!/usr/bin/python3
import os
from os.path import join, isfile
import numpy as np
import argparse
from Utils import skipthoughts
import h5py
import multiprocessing
from multiprocessing import Process, Queue

vector_name = ['uni_skip', 'bi_skip', 'combine_skip', 'one_hot', 'glove_50',
               'glove_100', 'glove_200', 'glove_300']
cpus = multiprocessing.cpu_count()
print('number of cpus = %d' % cpus)

def get_vector(keys, model, captions, out_q):
  vectors = {}
  for key in keys:
    vectors[key] = skipthoughts.encode(model, captions[key])
  out_q.put(vectors)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--caption_file', '-cf', type=str, help='caption file',
                      default='sample_testing_text.txt')
  parser.add_argument('--method_dir', '-md', type=str, default='',
                      help='data directory')
  parser.add_argument('--data_set', '-ds', type=str, default='faces',
                      help='data set name')
  parser.add_argument('--vector_type', '-vt', type=int, default=4,
                      choices=(0, 8),
                      help='method to encode caption, options: '
                           '0. uni-skip, 1. bi-skip, 2. combine-skip, '
                           '3. one-hot, 4. glove')
  parser.add_argument('--out_file', '-of', default='test_caption_vectors.hdf5',
                      type=str, help='output file name')
  parser.add_argument('--dict_file', '-df', default='onehot_hair_eyes.hdf5',
                      type=str, help='input dictionary name')
  parser.add_argument('--glove_file', '-gf', type=str, help='glove file',
                      default='glove/glove.6B.300d.txt')
  args = parser.parse_args()

  if args.method_dir == '':
    print('need to specify method_dir!')
    exit(1)

  with open(join(args.data_set, args.caption_file)) as f:
    captions = f.read().splitlines()
  captions = [cap for cap in captions]
  captions = [cap.split(',') for cap in captions]
  captions = dict([[cap[0], cap[1].split()]
                   for i, cap in enumerate(captions)])

  if args.vector_type < 3:
    for key, val in captions.items():
      captions[key] = ['the girl has '+val[0]+' '+val[1]
                      +' and '+val[2]+' '+val[3]]
    model = skipthoughts.load_model()

    parallel_keys = [[] for i in range(cpus)]
    quantity = (len(captions)+cpus-1)//cpus
    for i, key_val in enumerate(captions.items()):
      parallel_keys[i//quantity].append(key_val[0])

    out_q = Queue()
    thrds =\
      [Process(target=get_vector, args=(parallel_keys[i], model,
                                        captions, out_q))
       for i in range(cpus)]

    for i, thrd in enumerate(thrds): thrd.start()
    caption_vectors = {}
    for i in range(len(thrds)):
      caption_vectors.update(out_q.get())
    for i, thrd in enumerate(thrds): thrd.join()

  elif args.vector_type == 3:
    h = h5py.File(join(args.data_dir, args.dict_file),'r')
    for key, val in captions.items():
      new_val = [h['hair'][val[0]].value, h['eyes'][val[2]].value]
      new_val = [np.eye(h['hair'].attrs['size'])[new_val[0]],
                 np.eye(h['eyes'].attrs['size'])[new_val[1]]]
      caption_vectors[key] = new_val

  elif args.vector_type == 4:
    wordvecs = open(args.glove_file, 'r').read().splitlines()
    wordvecs = [wordvec.split() for wordvec in wordvecs]
    wordvecs = dict([[wordvec[0], np.array([wordvec[1:]], dtype=np.float32)]
                      for wordvec in wordvecs])
    caption_vectors = {}
    for key, val in captions.items():
      if val[0] == 'twintails':
        caption_vectors[key] =\
          (wordvecs['twin'] + wordvecs['tails'] + 2*wordvecs[val[1]])/4
      else:
        caption_vectors[key] =\
          (wordvecs[val[0]] + wordvecs[val[1]])/2

  filename = join(args.data_set, args.method_dir, args.out_file)
  if os.path.isfile(filename):
    os.remove(filename)
  h = h5py.File(filename, 'w')
  for key in caption_vectors.keys():
    h.create_dataset(key, data=caption_vectors[key])
  h.close()

if __name__ == '__main__':
  main()
