import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--caption_file', type=str, default='hw3_data/sample_testing_text.txt',
             help='caption file')
  parser.add_argument('--data_dir', type=str, default='hw3_data',
             help='Data Directory')

  args = parser.parse_args()
  with open( args.caption_file ) as f:
    captions = f.read().split('\n')

  captions = [cap for cap in captions if len(cap) > 0]
  captions = [cap.split(',')[1] for cap in captions]
  captions = [cap.split(' ') for cap in captions]
  captions = ['the girl has '+cap[0]+' '+cap[1]+' and '+cap[2]+' '+cap[3]
              for cap in captions]
  print(captions)
  model = skipthoughts.load_model()
  caption_vectors = skipthoughts.encode(model, captions)

  if os.path.isfile(join(args.data_dir, 'sample_caption_vectors.hdf5')):
    os.remove(join(args.data_dir, 'sample_caption_vectors.hdf5'))
  h = h5py.File(join(args.data_dir, 'sample_caption_vectors.hdf5'))
  h.create_dataset('vectors', data=caption_vectors)
  h.close()

if __name__ == '__main__':
  main()
