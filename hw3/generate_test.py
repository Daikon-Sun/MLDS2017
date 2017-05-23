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
                      default='hw3_data/sample_testing_text.txt')
  parser.add_argument('--data_dir', '-dd', type=str, default='hw3_data',
                      help='Data Directory')
  parser.add_argument('--vector', '-v', type=int, default=2,
                      choices=(0, 8),
                      help='method to encode caption, options: '
                           '0. uni-skip, 1. bi-skip, 2. combine-skip, '
                           '3. one-hot, 4. glove_50, 5. glove_100, '
                           '6. glove_200, 7. glove_300 ')
  parser.add_argument('--out_file', '-of', default='test_vector.hdf5',
                      type=str, help='output file name')
  parser.add_argument('--dict_file', '-df', default='onehot_hair_eyes.hdf5',
                      type=str, help='input dictionary name')
  args = parser.parse_args()

  with open( args.caption_file ) as f:
    captions = f.read().splitlines()
  captions = [cap for cap in captions]
  captions = [cap.split(',') for cap in captions]
  captions = dict([[cap[0], cap[1].split()]
                   for i, cap in enumerate(captions)])

  if args.vector < 3:
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

  elif args.vector == 3:
    h = h5py.File(join(args.data_dir, args.dict_file),'r')
    for key, val in captions.items():
      new_val = [h['hair'][val[0]].value, h['eyes'][val[2]].value]
      new_val = [np.eye(h['hair'].attrs['size'])[new_val[0]],
                 np.eye(h['eyes'].attrs['size'])[new_val[1]]]
      caption_vectors[key] = new_val

  elif args.vector >= 4:
    h = h5py.File(join(args.data_dir, vector_name[args.vector],
                       'glove.6B.'+vector_name[args.vector][6:]+'d.hdf5'),'r')
    for key, val in captions.items():
      new_val = [val[0], val[2]]
      caption_vectors[key] =\
        [np.hstack((h['__'+cap[0]+'__'].value, h['__'+cap[1]+'__'].value))
         for cap in captions]

  print(caption_vectors)
  if os.path.isfile(join(args.data_dir, args.out_file)):
    os.remove(join(args.data_dir, args.out_file))
  h = h5py.File(join(args.data_dir, args.out_file,), 'w')
  for key in caption_vectors.keys():
    h.create_dataset(key, data=caption_vectors[key])
  h.close()

if __name__ == '__main__':
  main()
