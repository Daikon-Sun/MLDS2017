#!/usr/bin/python3
import re
import string
import sys

# Check whether given sentence is a valid English sentence
def valid(sentence):
  if len(sentence.split()) < 3:
    return False
  for word in sentence.split():
    if not all(c in string.printable for c in word):
      return False
    has_digit = any(c in string.digits for c in word)
    has_alpha = any(c in string.ascii_letters for c in word)
    if has_digit and has_alpha:
      return False
    if has_alpha:
      if not any(c in ['a','e','i','o','u','y'] for c in word):
        return False      
  return True
  
# Parse the given content into dependency trees
def Parse(f, of):
  content = re.sub('\n|\r', ' ', ''.join( i for i in f ))
  content = re.split('[.!?;]+', content)
  for sentence in content:
    sent = re.sub(r'[^[a-zA-Z0-9 \']]*', '', re.sub(r' +', ' ', sentence))
    if not valid(sent): continue
    of.write(' '.join(sent.lower().split()) + "\n")

sys.stderr.write('start parsing...\n')
with open('training_list','r') as file_list:
  for file_name in file_list:
    with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
      sys.stderr.write('start parsing file ' + file_name[:-1] + '\n')
      with open("Training_Data/"+file_name[21:-5]+".txt",'w') as of:
        Parse(f,of)
      sys.stderr.write('finished parsing file ' + file_name[:-1] + '\n')
