import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py
import time


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir','-dd',type=str, default='hw3_data',
                     help='Data directory')
parser.add_argument('--data_set','-ds',type=str, default='bonus',
                     help='Data Set : faces')
parser.add_argument('--tag_name','-tn',type=str, default='hw3_data/bonus/tags_bonus_clean.csv',
                     help='tags')
parser.add_argument('--top_num', '-num', type=int, default=200,
                     help='top num')
args = parser.parse_args()

# retrieve all tags for each image
image_tags = dict()
tags = open(args.tag_name, 'r').read().splitlines()
for tag in tags:
  tag_tmp = tag.split(',')[1].split('\t')
  t = [t.split(':')[0] for t in tag_tmp]
  image_tags[tag.split(',')[0]] = t


# finding most popular tags 
tags = [tag.split(',')[1].split('\t') for tag in tags]
tags = [[t.split(':')[0] for t in tag] for tag in tags]

all_tags = dict()

for tag in tags:
  for t in tag:
  	if (t != ''):
  	  if t in all_tags:
  	    all_tags[t] = all_tags[t] + 1
  	  else:
  	    all_tags[t] = 1

s = [k for k in sorted(all_tags, key=all_tags.get, reverse=True)]
s = s[:args.top_num]

# generate image caption
image_captions = {}
for key, value in image_tags.items():
  image_captions[key] = ['the girl has']
  match = False
  for tag in value:
    if (tag in s) and (match == False):
      image_captions[key][0] = image_captions[key][0] + ' ' + tag
      match = True
    elif (tag in s) and match:
      image_captions[key][0] = image_captions[key][0] + ' and ' + tag
  if (match == False):
    del image_captions[key] # no matching tags, do not selected as input image

model = skipthoughts.load_model()
encoded_captions = {}

for i, key_val in enumerate(image_captions.items()):
  key = key_val[0]
  st = time.time()
  encoded_captions[key] = skipthoughts.encode(model, image_captions[key])
  print(i, len(image_captions), key)
  print("Seconds", time.time() - st)

h = h5py.File(join(args.data_dir, args.data_set+'.hdf5'))
for key in encoded_captions:
  h.create_dataset(key, data=encoded_captions[key])
h.close()



