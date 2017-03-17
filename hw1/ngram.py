from nltk import ngrams
import argparse
import re, string
import os, sys
import numpy as np

# output file is ngram_result.npy

argparser = argparse.ArgumentParser(description='Calculate probability based on N-gram')
argparser.add_argument('-i', '--file_list',
  type=str, default='training_list',
  help='FILE_LIST is the file storing the file names of training datas.'
       ' (default: %(default)s)')
argparser.add_argument('-n', '--ngram_num',
  type=int, default=3,
  help='NGRAM_NUM is the number n of ngram, ex. trigram has n = 3')
argparser.add_argument('-t', '--test_data',
  type=str, default='testing_data_out.txt',
  help='TEST_DATA is the file of testing data with underline filled with answers'
       ' (default: %(default)s)')
argparser.set_defaults(comma_split=False)
args = argparser.parse_args()

ngram = []

with open(args.file_list,'r') as file_list:
  for file_name in file_list:
    with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
      sys.stderr.write('start proccessing file ' + file_name[:-1] + '\n')
      lines = [l.strip('\n') for l in f.readlines()]
      if not lines:
        continue
      else:
        for sentence in lines:
          ngram.extend((ngrams(sentence.split(),
            args.ngram_num,
            pad_left=True,
            pad_right=True,
            left_pad_symbol='<s>',
            right_pad_symbol='</s>'
          )))

# counting ngram appearance
ngram_dictionary = dict()
counter = 0
for g in ngram:
  counter = counter + 1
  if ngram_dictionary.get(g, -1) == -1:
    ngram_dictionary[g] = 1
  else:
    ngram_dictionary[g] = ngram_dictionary[g] + 1

#counting probability
minimum_prob = 2 # maximal probability should be 1, initialize to 2 just in case
for key in ngram_dictionary:
  ngram_dictionary[key] = float(ngram_dictionary[key] / counter)
  if minimum_prob > ngram_dictionary[key]:
    minimum_prob = ngram_dictionary[key]

# calculate probability of the result
# one probability per sentence

sys.stderr.write('Calculating testing data...\n')
prob = []
with open(args.test_data, 'r') as test_data:
  lines = [l.strip('\n') for l in test_data.readlines()]
  for sentence in lines:
    ans_ngrams = ngrams(sentence.split(),
      args.ngram_num,
        pad_left=True,
        pad_right=True,
        left_pad_symbol='<s>',
        right_pad_symbol='</s>'
    )
    tmp_prob = 1
    for g in ans_ngrams:
      if g in ngram_dictionary:
        tmp_prob = tmp_prob * ngram_dictionary[g]
      else:
        tmp_prob = tmp_prob * minimum_prob
    prob.append(tmp_prob)

# save result in npy format
np.save("gram_result", np.array(prob))



