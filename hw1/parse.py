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

def traverse_tree(writer, tree, dep, words_id):
  if isinstance(tree, string_types):
    word = tree
  else:
    word = tree._label if isinstance(tree._label, string_types) \
                       else unicode_repr(self._label)
  if len(words_id) <= dep:
    words_id.append(0)
  words_id[dep] = 0 if word not in vocab_table else vocab_table[word]
  leaf = True
  if not isinstance(tree, string_types):
    for child in tree:
      leaf = False
      if isinstance(child, Tree):
        traverse_tree(writer, child, dep+1, words_id)
      elif isinstance(child, tuple):
        for cc in child:
          traverse_tree(writer, cc, dep+1, words_id)
      elif isinstance(child, string_types):
        traverse_tree(writer, child, dep+1, words_id)
      else:
        traverse_tree(writer, unicode_repr(child), dep+1, words_id)
  if leaf:
    global count_sentences
    count_sentences += 1
    counter += 1
    example = tf.train.Example(
      features=tf.train.Features(
        feature={
          'content': tf.train.Feature(
          int64_list=tf.train.Int64List(value=words_id[:dep+1])),
          'len': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[dep+1]))}))
    serialized = example.SerializeToString()
    writer.write(serialized)
  
# Parse the given content into dependency trees
def Parse(f, writer, dependency_tree, quote_split, comma_split,
          min_words, max_words):
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
      if len(sent.split()) < min_words or len(sent.split()) > max_words:
        continue
      for sent in en_nlp(sent.lower()).sents:
        tree = to_nltk_tree(sent.root)
        if tree == None or type(tree) == str: continue
        words_id = []
        traverse_tree(writer, tree, 0, words_id)
    else:
      if writer is None:
        for word in sent.lower().split():
          if word in corpus:
            corpus[ word ] += 1
          else:
            corpus[ word ] = 1
      else:
        sent = sent.lower().split()
        if len(sent) < min_words or len(sent) > max_words: continue
        global count_sentences
        count_sentences += 1
        words_id = []
        for word in sent:
          words_id.append(0 if word not in vocab_table else vocab_table[word])
        global unk_words
        global total_words
        unk_words += sum(1 if i == 0 else 0 for i in words_id)
        total_words += len(words_id)
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'content': tf.train.Feature(
                int64_list=tf.train.Int64List(value=words_id)),
              'len': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[len(words_id)]))}))
        serialized = example.SerializeToString()
        writer.write(serialized)

def Parse_testing(f, writer, dependency_tree):
  number_of_tree = []
  for question in f:
    ret = question.split(',')
    if ret[0] == 'id':
      continue
    sent = ','.join(ret[1:-5])
    for choice in ret[-5:]:
      cand = re.sub('_____', choice, sent)
      for cc in '.!?;,':
        cand = re.sub('['+cc+']',' '+cc+' ', cand)
      cand = re.sub('[\']',' \'', cand)
      cand = re.sub('[^A-Za-z0-9\s\',.!?;]', '', cand)
      if writer is None:
        for word in cand.lower().split():
          corpus_testing[ word ] = 1  
      elif dependency_tree:
        global counter
        counter = 0
        for sent in en_nlp(cand.lower()).sents:
          tree = to_nltk_tree(sent.root)
          if tree == None or type(tree) == str: continue
          words_id = []
          traverse_tree(writer, tree, 0, words_id)
        number_of_tree.append(counter)
      else:
        if args.debug:
          sys.stderr.write(cand + '\n')
        words_id = []
        for word in cand.lower().split():
          words_id.append(0 if word not in vocab_table else vocab_table[word])
        example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'content': tf.train.Feature(
                int64_list=tf.train.Int64List(value=words_id)),
              'len': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[len(words_id)]))}))
        serialized = example.SerializeToString()
        writer.write(serialized)
  if dependency_tree:
    np.save('number_of_tree.npy',np.array(number_of_tree))

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
  argparser.add_argument('-t', '--testing_data',
        type=str, default='testing_data.csv',
        help='TESTING_DATA is the file containing testing data.'
             ' (default: %(default)s)')
  argparser.add_argument('-g', '--glove_file',
        type=str, default='data/glove.6B.50d.txt',
        help='GLOVE_FILE is the file containning glove data.'
             'Should be in the format of glove.#B.#d.txt'
             ' (default: %(default)s)')
  argparser.add_argument('-o', '--output_dir',
        type=str, default='Training_Data',
        help='OUTPUT_DIR is the directory where the '
             'output files of training datas will be '
             'stored in. (default: %(default)s)',)
  argparser.add_argument('-of', '--output_file',
        type=str, default='testing_data.tfr',
        help='OUTPUT_FILE is the TFRecorder file from '
             'testing data (default: %(default)s)',)
  argparser.add_argument('-c', '--count',
        type=int, default=16,
        help='If the occurrence of a word is less than COUNT'
             'it won\'t be taken into consideration.'
             '(Words in testing data is always counted.)'
             '(default: %(default)s)',)
  argparser.add_argument('-mi', '--min_words',
        type=int, default=4,
        help='If the sentence has less than MIN_WORDS words, '
             'it won\'t be taken into consideration.(Marks also count)'
             '(default: %(default)s)',)
  argparser.add_argument('-ma', '--max_words',
        type=int, default=100,
        help='If the sentence has more than MAX_WORDS words, '
             'it won\'t be taken into consideration.(Marks also count)'
             '(default: %(default)s)',)
  argparser.add_argument('-q', '--quote_split',
        help='Sentences will be split between quotation marks (\'\"\')'
             ' (default: split only by {\'.\',\'!\',\'?\',\';\'}).',
        action='store_true')
  argparser.add_argument('-co', '--comma_split',
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
  corpus_testing = dict()

  sys.stderr.write('start parsing datas...\n')
  with open(args.file_list,'r') as file_list:
    for file_name in file_list:
      with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
        if args.debug:
          sys.stderr.write('start parsing file ' + file_name[:-1] + '\n')
        Parse(f, None, False, args.quote_split, args.comma_split,
              args.min_words, args.max_words)
        if args.debug:
          sys.stderr.write('finished parsing file ' + file_name[:-1] + '\n')
  with open(args.testing_data,'r',encoding="utf-8",errors='ignore') as f:
    Parse_testing(f, None, args.dependency_tree)

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
        if (ret[0] in corpus and corpus[ret[0]] >= args.count) or \
            ret[0] in corpus_testing:
          word_list.write(ret[0]+'\n')
          res.append(list(map(float,ret[1:])))
      for i in res:
        while len(i) < dimension:
          i.append(float(0))
      np.save(wordvec_name,np.array(res))
      sys.stderr.write('number of useful words : %d\n' % (len(res)))
      
  sys.stderr.write('start transforming training datas'
                   ' into the format of TFRecoder files...\n')
  vocab_table = dict()
  vocab_table_idx = 0
  with open(vocab_name,'r') as vocab:
    for w in vocab:
      vocab_table[w[:-1]] = vocab_table_idx
      vocab_table_idx = vocab_table_idx + 1

  global counter
  global unk_words
  global total_words
  global count_sentences
  unk_words, total_words, count_sentences = 0, 0, 0
  with open(args.file_list,'r') as file_list:
    for file_name in file_list:
      with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
        if args.debug:
          sys.stderr.write('start converting file ' + file_name[:-1] + '\n')
        writer = tf.python_io.TFRecordWriter(args.output_dir+'/'+file_name[21:-5]+'.tfr')
        Parse(f, writer, args.dependency_tree, args.quote_split, args.comma_split,
              args.min_words, args.max_words)
  sys.stderr.write('unk_words: %d, total_words: %d, perc: %f%%\n' %
                   (unk_words, total_words, unk_words * 100 / total_words))
  sys.stderr.write('Number of sentences: %d\n' % count_sentences)

  sys.stderr.write('start transforming testing data '
                   'into the format of TFRecoder file...\n')
  with open(args.testing_data,'r',encoding="utf-8",errors='ignore') as f:
    writer = tf.python_io.TFRecordWriter(args.output_file)
    Parse_testing(f, writer, args.dependency_tree)

  sys.stderr.write('cooling down...\n')
