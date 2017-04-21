#!/usr/bin/python3
import argparse
import json
import numpy as np
import re, sys
from gensim.models import word2vec

def normalized(sent):
  s = sent.lower()
  for deli in ['\'','.','?','!',',',';',':','\"']:
    s = re.sub('['+deli+']', ' '+deli, s)
  return '<bos> ' + ' '.join(s.split()) + ' <eos>'

# Parse the given content into dependency trees
def embedding(input_file, output_word_file, output_vector_file, dimension):
  with open(input_file) as f:
    data = json.load(f)
    sents = []
    sys.stderr.write('start reading json file....\n')
    for unit in data:
      for sent in unit['caption']:
        sents.append(normalized(sent).split())
    sys.stderr.write('start calculating embedding word....\n')
    model = word2vec.Word2Vec(sents, size=dimension, window=5, min_count=0, workers=4)
    model.save('embedding_model')
    words, vectors = [], []
    got = set()
    sys.stderr.write('start storing embedding word....\n')
    for sent in sents:
      for word in sent:
        if word in got:
          continue
        got.add(word)
        words.append(word)
        vectors.append(model.wv[word])
    np.save(output_word_file, np.array(words))
    np.save(output_vector_file, np.array(vectors))

if __name__ == '__main__':
  # parsing arguments
  argparser = argparse.ArgumentParser(description='Embedding words from '
        'given datas.')
  argparser.add_argument('-i', '--input_file',
        type=str, default='MLDS_hw2_data/training_label.json',
        help='INPUT_FILE is the file storing the words to embed.'
             'It should be a JSON format file.'
             ' (default: %(default)s)')
  argparser.add_argument('-ow', '--output_word_file',
        type=str, default='word_100d.npy',
        help='OUTPUT_WORD_FILE is the file storing word in '
             'the order of embedding in the numpy array format.'
             '(default: %(default)s)',)
  argparser.add_argument('-ov', '--output_vector_file',
        type=str, default='vector_100d.npy',
        help='OUTPUT_VECTOR_FILE is the file storing embedding word '
             'vector in the order of embedding in the numpy array format.'
             '(default: %(default)s)',)
  argparser.add_argument('-d', '--dimension',
        type=int, default=100,
        help='The embedding words will be a DIMENSION-dimension '
             'real value vector.'
             '(default: %(default)s)',)
  args = argparser.parse_args()
  embedding(args.input_file, args.output_word_file,
            args.output_vector_file, args.dimension)
