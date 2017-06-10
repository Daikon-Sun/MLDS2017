#!/usr/bin/python3
import os, sys, time, copy, h5py, argparse
import numpy as np
from tf_cnnvis import *
import tensorflow as tf
from subprocess import call
from scipy.misc import imread, imresize

with tf.gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'input' : t_preprocessed})

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', '-p', type=str, default='',
                    help='prefix of log and output directory')
parser.add_argument('--layers', '-l', type=str, default='rpc',
                    help='layers to be visualized')
parser.add_argument('--method', '-m', type=int, default=0,
                    choices=(0, 1), help='methods to visualize')
parser.add_argument('--image', '-i', type=str, default='NULL',
                    help='input image')

args = parser.parse_args()
layers = list(args.layers)
logdir = args.prefix+'_Log'
outdir = args.prefix+'_Output'
im = np.expand_dims(imresize(imread(os.path.join("../imgs", args.image)), (224, 224)), axis = 0)
visual_func = [deconv_visualization, activation_visualization]

start = time.time()
is_success =\
  visual_func[args.method](graph_or_path=tf.get_default_graph(),
                           value_feed_dict={t_input : im}, layers=layers,
                           path_logdir=logdir, path_outdir=outdir)
start = time.time() - start
print("Total Time = %f" % (start))
