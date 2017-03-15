import string
import sys
import numpy as np

# GloVe pretrained word vectors should be put under the same directory

if len(sys.argv) < 3:
  sys.stderr.write("Usage: python3 " + sys.argv[0] + " < #B > < #d >\n")
  sys.exit()
sys.stderr.write('*** Seperate pre-trained word vectors from GloVe ***\n')

with open("glove." + sys.argv[1] + "B." + sys.argv[2] + "d.txt",'r') as glove:
  with open("vocab_" + sys.argv[1] + "B_" + sys.argv[2] + "_d.txt",'w') as word_list:
    words = glove.read().split()
    dimension = int(sys.argv[2])
    counter = 0
    wordnum = 0
    arr = []

    # unknown vocab
    word_list.write("<unk>\n")
    arr.append([])
    for i in range(dimension):
      arr[wordnum].append(float(0))
    wordnum = wordnum + 1

    for w in words:
      if counter == 0:
        word_list.write(w + "\n")
        print("processing word:" + w)
        wordnum = wordnum + 1
        counter = counter + 1
        arr.append([])
      elif ( counter > 0 and counter < dimension ):
        counter = counter + 1
        arr[wordnum - 1].append(float(w))
      else:
        counter = 0
        arr[wordnum - 1].append(float(w))
    np.save("wordvec_" + sys.argv[1] + "B_" + sys.argv[2] + "_d",np.array(arr))
