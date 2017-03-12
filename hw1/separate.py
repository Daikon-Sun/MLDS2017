import string
import sys

# GloVe pretrained word vectors should be put under the same directory

if len(sys.argv) < 3:
  sys.stderr.write("Usage: python3 " + sys.argv[0] + " < #B > < #d >\n")
  sys.exit()
sys.stderr.write('*** Seperate pre-trained word vectors from GloVe ***\n')

with open("glove." + sys.argv[1] + "B." + sys.argv[2] + "d.txt",'r') as glove:
  with open("pretrained_word_list.txt",'w') as word_list:
    with open("pretrained_word_vectors_" + sys.argv[1] + "d.txt",'w') as vector_list:
      words = glove.read().split()
      dimension = int(sys.argv[2])
      counter = 0
      for w in words:
        if counter == 0:
          word_list.write(w + "\n")
          print("processing word:" + w)
          counter = counter + 1
        elif ( counter > 0 and counter < dimension ):
          vector_list.write(w + " ")
          counter = counter + 1
        else:
          counter = 0
          vector_list.write(w + "\n")
