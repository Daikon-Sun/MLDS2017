import os, sys, argparse, json, re
import numpy as np

top_num = 50

tags = open('tags_clean.csv', 'r').read().splitlines()
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

s = [(k, all_tags[k]) for k in sorted(all_tags, key=all_tags.get, reverse=True)]

with open('bonus_tags.txt', 'w') as of:
  for i in range(top_num):
    of.write(s[i][0] + '\n')

print(s[:top_num])
exit()
