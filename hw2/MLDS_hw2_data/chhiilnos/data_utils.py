import re
import json
from pprint import pprint
from tensorflow.python.platform import gfile
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *

_PAD = b"_PAD"
_BOS = b"_BOS"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD,_BOS,_EOS,_UNK]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

def basic_tokenizer(sentence):
  tokenizer = RegexpTokenizer(r'\w+')
  sentence = sentence.lower()
  tokens =  tokenizer.tokenize(sentence.decode("utf-8"))
  return tokens

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=None, tokenizer=None):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    sentences = []
    with open(data_path) as data_file:
      data = json.load(data_file)
      for single_video in data:
        for caption in single_video["caption"]:
          sentences.append(caption)
    print("len of sentences is")
    print(len(sentences))

    counter = 0
    for sentence in sentences:
      counter +=1
      if counter % 200 ==0:
        print("processing line %d" % counter)
      sentence = tf.compat.as_bytes(sentence)
      tokens = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
      for word in tokens:
        if word in vocab:
          vocab[word]+=1
        else:
          vocab[word]=1

    vocab_list = [w.decode('utf-8') for w in _START_VOCAB] + sorted(vocab, key = vocab.get, reverse=True)
    print("len")
    print(len(vocab_list))
    if(max_vocabulary_size):
      if len(vocab_list)>max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

    vocab = {}
    index = 0

    for word in vocab_list:
      vocab.update({word:index})
      index = index+1

    with open(vocabulary_path,'w') as vocab_file:
      json.dump(vocab,vocab_file)

def sentence_to_token_ids(sentence,vocabulary,tokenizer=None):
  if(tokenizer):
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w,UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
  with open(vocabulary_path,'rb') as vocab_file:
    vocab = json.load(vocab_file)
  with open(data_path) as data_file:
    data = json.load(data_file)
  target = []
  for single_video in data:
    captions = []
    for caption in single_video["caption"]:
      captions.append(sentence_to_token_ids(tf.compat.as_bytes(caption),vocab))
    target.append({"caption":captions,"id":single_video["id"]})

  with open(target_path,'w') as target_file:
    json.dump(target,target_file)


create_vocabulary(vocabulary_path = 'vocab.json' , data_path = 'training_label.json')
data_to_token_ids(data_path ='training_label.json' ,vocabulary_path = 'vocab.json',target_path = 'training_label_id.json')
data_to_token_ids(data_path ='testing_public_label.json' ,vocabulary_path = 'vocab.json',target_path = 'testing_public_label_id.json')
