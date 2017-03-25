#!/usr/bin/python3
import argparse
import re, string
import os, sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

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

def word_normalize(word):
  if any(c.isdigit() for c in word):
    return '0'
  if any(c.isalpha() for c in word):
    if not all(c.isalpha() for c in word):
      if len(word)>1 and word[0]=='\'' and word[1].isalpha() and len(word)<4:
        return word
      return ''.join(c if c.isalpha() else '' for c in word)
  return word

def traverse_tree(writer, tree, dep, words_id, useful, choice):
  if isinstance(tree, str):
    word = tree
  else:
    word = tree._label if isinstance(tree._label, str) \
                       else unicode_repr(self._label)
  if len(words_id) <= dep:
    words_id.append(0)
  if word == choice:
    useful = True
  if word == '*':
    dep -= 1
  else:
    words_id[dep] = 0 if word not in vocab_table else vocab_table[word]
  leaf = True
  if not isinstance(tree, str):
    for child in tree:
      leaf = False
      if isinstance(child, Tree):
        traverse_tree(writer, child, dep+1, words_id, useful, choice)
      elif isinstance(child, tuple):
        for cc in child:
          traverse_tree(writer, cc, dep+1, words_id, useful, choice)
      elif isinstance(child, str):
        traverse_tree(writer, child, dep+1, words_id, useful, choice)
      else:
        traverse_tree(writer, unicode_repr(child), dep+1, words_id, useful, choice)
  if leaf and useful:
    global count_sentences
    global counter_tree
    count_sentences += 1
    counter_tree += 1
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
          min_words, max_words, slice_out, self_parse):
  content = re.sub('\n|\r', ' ', ''.join( i for i in f ))
  if slice_out:
    ret = re.search(r"\bC[hH][aA][pP][tT][eE][rR] (I|One|ONE)\b", content)
    st_idx = 0 if ret is None else ret.start()
    ret = re.search(r"\bEnd of (|The )Project\b", content)
    ed_idx = len(content) if ret is None else ret.start()
    content = content[st_idx:ed_idx]

  global count_sentences
  global unk_words
  global total_words

  if not self_parse:
    for sent in sent_tokenize(content):
      if args.debug:
        sys.stderr.write(sent+'\n')
      words = word_tokenize(sent.lower())
      mod_sent = ' '.join( word_normalize(w) for w in words )
      if not any(word_normalize(word) in corpus_testing and\
                 word_normalize(word).isalpha() for word in words):
        continue
      if dependency_tree:
        if len(words) < min_words or len(words) > max_words:
          continue
        for sub_sent in en_nlp(mod_sent).sents:
          tree = to_nltk_tree(sub_sent.root)
          if tree == None or type(tree) == str: continue
          words_id = []
          traverse_tree(writer, tree, 0, words_id, True, None)
      elif writer is None:
        for word in words:
          w = word_normalize(word)
          if w in corpus:
            corpus[ w ] += 1
          else:
            corpus[ w ] = 1
      else:
        if len(words) < min_words or len(words) > max_words: continue
        count_sentences += 1
        words_id = []
        for word in words:
          w = word_normalize(word)
          words_id.append(0 if w not in vocab_table else vocab_table[w])
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
    return
  """End of nltk parse"""

def Parse_testing(f, writer, dependency_tree, self_parse):
  number_of_tree = []
  global counter_tree
  for question in f:
    question = question[:-1]
    ret = question.split(',')
    if ret[0] == 'id':
      continue
    sent = ','.join(ret[1:-5])
    for choice in ret[-5:]:
      cand = re.sub('_____', choice, sent)
      if not self_parse:
        words = word_tokenize(cand.lower())
        if writer is None:
          for word in words:
            w = word_normalize(word)
            corpus_testing[ w ] = 1
        elif dependency_tree:
          counter_tree = 0
          mod_cand = ' '.join( word_normalize(w) for w in words )
          for sub_sent in en_nlp(mod_cand).sents:
            tree = to_nltk_tree(sub_sent.root)
            if tree == None or type(tree) == str: continue
            words_id = []
            traverse_tree(writer, tree, 0, words_id, False, word_normalize(choice))
          number_of_tree.append(counter_tree)
        else:
          if args.debug:
            sys.stderr.write(cand + '\n')
          words_id = []
          for word in words:
            w = word_normalize(word)
            words_id.append(0 if w not in vocab_table else vocab_table[w])
          example = tf.train.Example(
            features=tf.train.Features(
              feature={
                'content': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=words_id)),
                'len': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[len(words_id)]))}))
          serialized = example.SerializeToString()
          writer.write(serialized)
      """End of nltk parse"""
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
  argparser.add_argument('-og', '--output_glove_dir',
        type=str, default='data/',
        help='OUTPUT_GLOVE_DIR is the directory where the '
             'wordvec and vocab files from glove file be '
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
  argparser.add_argument('-p', '--self_parse',
        help='Parsed by self written method w/o pretrained knowledge'
             ' (default: parsed by nltk datas)',
        action='store_true')
  argparser.add_argument('-s', '--slice_out',
        help='Head and tail part of content will be sliced out.'
             ' (default: It won\'t be sliced out)',
        action='store_true')
  argparser.add_argument('-sk', '--skip',
        help='Skip out transforming training data parrs'
             ' (default: It won\'t be skipped)',
        action='store_true')
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
  global count_sentences
  global counter
  global counter_tree
  count_sentences = 0
  counter = 0
  counter_tree = 0

  if not args.skip:
    sys.stderr.write('start parsing datas...\n')
    with open(args.testing_data,'r',encoding="utf-8",errors='ignore') as f:
      Parse_testing(f, None, args.dependency_tree, args.self_parse)
    with open(args.file_list,'r') as file_list:
      for file_name in tqdm(file_list):
        with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
          if args.debug:
            sys.stderr.write('start parsing file ' + file_name[:-1] + '\n')
          Parse(f, None, False, args.quote_split, args.comma_split,
                args.min_words, args.max_words, args.slice_out, args.self_parse)
          if args.debug:
            sys.stderr.write('finished parsing file ' + file_name[:-1] + '\n')

  sys.stderr.write('start embedding words...\n')
  vocab_name = re.sub('glove','vocab',args.glove_file)
  if not args.skip:
    with open(args.glove_file,'r') as glove:
      ret = re.search(r"/", vocab_name)
      if ret is None: vocab_name = args.output_glove_dir + vocab_name
      else: vocab_name = args.output_glove_dir + vocab_name[ret.start():]
      wordvec_name = re.sub('txt','npy',re.sub('glove','wordvec',args.glove_file))
      if ret is None: wordvec_name = args.output_glove_dir + wordvec_name
      else: wordvec_name = args.output_glove_dir + wordvec_name[ret.start():]
      with open(vocab_name,'w') as word_list:
        dimension = int(args.glove_file.split('.')[2][:-1])
        word_list.write("<unk>\n")
        res = [[]]
        for word in tqdm(glove):
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

  vocab_table = dict()
  vocab_table_idx = 0
  with open(vocab_name,'r') as vocab:
    for w in vocab:
      vocab_table[w[:-1]] = vocab_table_idx
      vocab_table_idx = vocab_table_idx + 1

  if not args.skip:
    sys.stderr.write('start transforming training datas'
                     ' into the format of TFRecoder files...\n')
    global unk_words
    global total_words
    unk_words, total_words = 0, 0
    with open(args.file_list,'r') as file_list:
      for file_name in tqdm(file_list):
        with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
          if args.debug:
            sys.stderr.write('start converting file ' + file_name[:-1] + '\n')
          writer = tf.python_io.TFRecordWriter(args.output_dir+'/'+file_name[21:-5]+'.tfr')
          Parse(f, writer, args.dependency_tree, args.quote_split, args.comma_split,
                args.min_words, args.max_words, args.slice_out, args.self_parse)
    #sys.stderr.write('unk_words: %d, total_words: %d, perc: %f%%\n' %
    #                 (unk_words, total_words, unk_words * 100 / total_words))
    sys.stderr.write('Number of sentences: %d\n' % count_sentences)

  sys.stderr.write('start transforming testing data '
                   'into the format of TFRecoder file...\n')
  with open(args.testing_data,'r',encoding="utf-8",errors='ignore') as f:
    writer = tf.python_io.TFRecordWriter(args.output_file)
    Parse_testing(f, writer, args.dependency_tree, args.self_parse)

  sys.stderr.write('cooling down...\n')
