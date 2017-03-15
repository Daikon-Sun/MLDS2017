#!/usr/bin/python3
import argparse
import re, string
import os, sys

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
def Parse(f, of, dependency_tree, quote_split, comma_split):
  content = re.sub('\n|\r', ' ', ''.join( i for i in f ))
  content = re.sub('[,]',' , ',content)
  target = '.!?;\'\"' if quote_split else '.!?;'
  if comma_split: target += ','
  for cc in target:
    content = re.sub('['+cc+']',' '+cc+'\n', content)
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
          tf = open('tmp','r')
          of.write(' '.join(line.strip() for line in tf) + '\n' )
    else:
      of.write(' '.join(sent.lower().split()) + "\n")

if __name__ == '__main__':
  # parsing arguments
  argparser = argparse.ArgumentParser(description='Parsing Given datas '
        'into raw strings or in the format of dependency tree')
  argparser.add_argument('-d', '--dependency_tree',
        help='output will be in the format of dependency tree'
             ' (default: raw string).',
        action='store_true')
  argparser.add_argument('-i', '--file_list',
        type=str, default='training_list',
        help='FILE_LIST is the file storing the file names of training datas.'
             ' (default: %(default)s)')
  argparser.add_argument('-o', '--output_dir',
        type=str, default='Training_Data',
        help='OUTPUT_DIR is the directory where the output files will be '
             'stored in. (default: %(default)s)',)
  argparser.add_argument('-q', '--quote_split',
        help='Sentences will be split between quotation marks (\'\'\',\'\"\')'
             ' (default: split only by {\'.\',\'!\',\'?\',\';\'}).',
        action='store_true')
  argparser.add_argument('-c', '--comma_split',
        help='Sentences will also be split between comma.'
             ' (default: it won\'t split between comma).',
        action='store_false')
  argparser.set_defaults(comma_split=False)
  args = argparser.parse_args()

  if args.dependency_tree:
    import spacy
    from nltk import Tree
    sys.stderr.write('loading lauguage parser...\n')
    # English language parser
    en_nlp = spacy.load('en')
  else:
    sys.stderr.write('Parsing datas into raw strings. \n')

  sys.stderr.write('start parsing...\n')
  with open(args.file_list,'r') as file_list:
    for file_name in file_list:
      with open(file_name[:-1],'r',encoding="utf-8",errors='ignore') as f:
        sys.stderr.write('start parsing file ' + file_name[:-1] + '\n')
        with open(args.output_dir+"/"+file_name[21:-5]+".txt",'w') as of:
          Parse(f, of, args.dependency_tree, args.quote_split, args.comma_split)
        sys.stderr.write('finished parsing file ' + file_name[:-1] + '\n')
  if args.dependency_tree:
    os.remove('tmp')
