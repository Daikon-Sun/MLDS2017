import string
import sys

# GloVe pretrained word vectors should be put under the same directory

if len(sys.argv) < 2:
	sys.stderr.write("Usage: python3 " + sys.argv[0] + " <vector dimension>\n")
	sys.exit()
sys.stderr.write('*** Seperate pre-trained word vectors from GloVe ***\n')

with open("glove.6B."+sys.argv[1]+"d.txt",'r') as glove:
	with open("pretrained_word_list.txt",'w') as word_list:
		with open("pretrained_word_vectors.txt",'w') as vector_list:
			words = glove.read().split()
			counter = 0
			for w in words:
				if counter == 0:
					word_list.write(w + "\n")
					print("processing word:" + w)
					counter = counter + 1
				elif ( counter > 0 and counter < 50 ):
					vector_list.write(w + " ")
					counter = counter + 1
				else:
					counter = 0
					vector_list.write(w + "\n")
