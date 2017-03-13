import string
import sys
import re

# convert words into vectors using pretrained word vectors from GloVe

if len(sys.argv) < 3:
  sys.stderr.write("Usage: python3 " + sys.argv[0] + " < input file >" + " < pretrained word vectors >" + " \n")
  sys.exit()
sys.stderr.write("Start converting...\n")

dictionary = dict()

sys.stderr.write("Processing GloVe...\n")

with open(sys.argv[2],'r') as glove:
  match = re.findall(r'\d+',sys.argv[2])
  if len(match) < 2:
    sys.stderr.write("Error: incorrect pretrained word vectors file name format")
  dimension = int(match[1])
  counter = 0
  vector = ""
  key = ""
  words = glove.read().split()
  for w in words:
    if counter == 0:
      key = w
      sys.stderr.write("processing word " + w + "\n")
      counter = counter + 1
    elif (counter > 0 and counter < dimension):
      vector = vector + w + " "
      counter = counter + 1
    else:
      vector = vector + w + " "
      dictionary[key] = vector
      vector = ""
      counter = 0

with open(sys.argv[1],'r') as read_file:
  with open(sys.argv[1][:-4] + "_word2vec_" + match[1] + "d.txt",'w') as output_file:
    words = read_file.read().split()
    for w in words:
      sys.stderr.write("converting word " + w + "\n")
      if w in dictionary:
        output_file.write(dictionary[w])
      else:
        sys.stderr.write(w + " is not in GloVe pretrained word vectors\n")



