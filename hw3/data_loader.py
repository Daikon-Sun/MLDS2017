import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

def wanted_words(words):
  if len(words) != 2:
    return False
  if 'hair' in words or 'eyes' in words:
    return words[0] != 'long' and words[0] != 'short'\
                              and words[0] != 'bicolored'
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

def save_caption_vectors_flowers(data_set, data_dir, tag_name):
  import time

  img_dir = join(data_dir, data_set)
  image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]

  print('number of train images: %d' % len(image_files))

  tag_file = join(data_dir, tag_name)
  tags = open(tag_file, 'r').read().splitlines()
  tags = [tag.split(',')[1].split('\t') for tag in tags]
  tags = [[words.split(':')[0].split() for words in tag] for tag in tags]
  tags = [[ words for words in tag if wanted_words(words)] for tag in tags]
  tags = dict([[i, to_same(tag)] for i, tag in enumerate(tags) if wanted_tag(tag)])
  print(tags)
  exit()

  image_captions = {}
  for key, val in tags.items():
    image_captions[str(key)+'.jpg'] = ['the girl has '+val[0][0]+' '+val[0][1]+
                                       ' and '+val[1][0]+' '+val[1][1]]

  model = skipthoughts.load_model()
  encoded_captions = {}

  for i, key_val in enumerate(image_captions.items()):
    key = key_val[0]
    st = time.time()
    encoded_captions[key] = skipthoughts.encode(model, image_captions[key])
    print(i, len(image_captions), key)
    print("Seconds", time.time() - st)

  h = h5py.File(join(data_dir, data_set+'.hdf5'))
  for key in encoded_captions:
    h.create_dataset(key, data=encoded_captions[key])
  h.close()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir','-dd',type=str, default='hw3_data',
                       help='Data directory')
  parser.add_argument('--data_set','-ds',type=str, default='faces',
                       help='Data Set : faces')
  parser.add_argument('--tag_name','-tn',type=str, default='tags_clean.csv',
                       help='tags')
  #parser.add_argument('--vector','-v',type=int, default='skip-thought vector',
  #                     help='word vector to use, options: 1. skip-thought vector, combine-skip 2. skip-thought vector, uni-skip 3. skip-thought vector, bi-skip,4. glove vector dimensoin = 300,5. one hot vector')
  #parser.add_argument('--gan type','-gt',type=int, default='dcgan',
  #        help='type of conditional gan to use, options: 1.conditional dcgan, 2.conditional lsgan, 3.conditional gan, 4.conditional wgan')
  args = parser.parse_args()
  save_caption_vectors_flowers(args.data_set, args.data_dir, args.tag_name)

if __name__ == '__main__':
  main()
