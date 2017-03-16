#!/usr/bin/python3
import argparse
import re, string
import os, sys
import numpy as np
import tensorflow as tf

# Generating dependency tree
def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
  else:
    return node.orth_

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
def Parse(f, writer, dependency_tree, quote_split, comma_split):
  content = re.sub('\n|\r', ' ', ''.join( i for i in f ))
  content = re.sub('[,]',' , ',content)
  target = '.!?;\"' if quote_split else '.!?;'
  if comma_split: target += ','
  for cc in target:
    content = re.sub('['+cc+']',' '+cc+'\n', content)
  content = re.sub('[\']',' \'', content)
  for sentence in content.split('\n'):
    sent = re.sub('[^A-Za-z0-9\s\',.!?;]', '', re.sub('\s+', ' ', sentence))
    if not valid(sent): continue
    if dependency_tree:
      for sent in en_nlp(sent.lower()).sents:
        tree = to_nltk_tree(sent.root)
        if tree == None: continue
        if type(tree) != str:
          sys.stdout = open('tmp','w')
          tree.pprint()
          sys.stdout.flush()
          # of.write(' '.join(line.strip() for line in open('tmp','r')) + '\n' )
    else:
      if writer is None:
        for word in sent.lower().split():
          if word in corpus:
            corpus[ word ] += 1
          else:
            corpus[ word ] = 1
      else:
        words_id = []
        for word in sent.lower().split():
          if word in vocab_table:
            words_id.append(vocab_table[word])
          else:
            words_id.append(0)
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'content': tf.train.Feature(
                int64_list=tf.train.Int64List(value=words_id)),
              'len': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[len(words_id)]))}))
        serialized = example.SerializeToString()
        writer.write(serialized)

if __name__ == '__main__':
  # parsing arguments
  argparser = argparse.ArgumentParser(description='Parsing Given datas '
        'into the format of TFRecorder file.')
  argparser.add_argument('-d', '--dependency_tree',
        help='output will be in the format of dependency tree'
             ' (default: raw string).',
        action='store_true')
  argparser.add_argument('-i', '--file_list',
        type=str, default='training_list',
        help='FILE_LIST is the file storing the file names of training datas.'
             ' (default: %(default)s)')
  argparser.add_argument('-g', '--glove_file',
        type=str, default='glove.6B.50d.txt',
        help='GLOVE_FILE is the file containning glove data.'
             'Should be in the format of glove.#B.#d.txt'
             ' (default: %(default)s)')
  argparser.add_argument('-o', '--output_dir',
        type=str, default='Training_Data',
        help='OUTPUT_DIR is the directory where the output files will be '
             'stored in. (default: %(default)s)',)
  argparser.add_argument('-q', '--quote_split',
        help='Sentences will be split between quotation marks (\'\"\')'
             ' (default: split only by {\'.\',\'!\',\'?\',\';\'}).',
        action='store_true')
  argparser.add_argument('-c', '--comma_split',
        help='Sentences will also be split between comma.'
             ' (default: it won\'t split between comma).',
        action='store_true')
  argparser.add_argument('-de', '--debug',
        help='Show more debug infos',
        action='store_true')
  args = argparser.parse_args()

  if args.dependency_tree:
    import spacy
    from nltk import Tree
    sys.stderr.write('loading lauguage parser...\n')
    # English language parser
    en_nlp = spacy.load('en')

  corpus = dict()

  sys.stderr.write('start parsing...\n')
  with open(args.file_list,'r') as file_list:
    for file_name in file_list:
      with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
        if args.debug:
          sys.stderr.write('start parsing file ' + file_name[:-1] + '\n')
        Parse(f, None, args.dependency_tree, args.quote_split, args.comma_split)
        if args.debug:
          sys.stderr.write('finished parsing file ' + file_name[:-1] + '\n')
  if args.dependency_tree:
    os.remove('tmp')

  sys.stderr.write('start embedding words...\n')
  with open(args.glove_file,'r') as glove:
    vocab_name = re.sub('glove','vocab',args.glove_file)
    wordvec_name = re.sub('txt','npy',re.sub('glove','wordvec',args.glove_file))
    with open(vocab_name,'w') as word_list:
      dimension = int(args.glove_file.split('.')[2][:-1])
      word_list.write("<unk>\n")
      res = [[]]
      for word in glove:
        ret = word.split()
        if ret[0] in corpus and corpus[ret[0]] > 1:
          word_list.write(ret[0]+'\n')
          res.append(list(map(float,ret[1:])))
      for i in res:
        while len(i) < dimension:
          i.append(float(0))
      np.save(wordvec_name,np.array(res))
      sys.stderr.write('number of useful words : %d\n' % (len(res)))
      
  sys.stderr.write('start transforming into the format of TFRecoder file...\n')
  vocab_table = dict()
  vocab_table_idx = 0
  with open(vocab_name,'r') as vocab:
    for w in vocab:
      vocab_table[w[:-1]] = vocab_table_idx
      vocab_table_idx = vocab_table_idx + 1

  with open(args.file_list,'r') as file_list:
    for file_name in file_list:
      with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
        if args.debug:
          sys.stderr.write('start converting file ' + file_name[:-1] + '\n')
        writer = tf.python_io.TFRecordWriter(args.output_dir+'/'+file_name[21:-5]+'.tfr')
        Parse(f, writer, args.dependency_tree, args.quote_split, args.comma_split)

  sys.stderr.write('cooling down...\n')
