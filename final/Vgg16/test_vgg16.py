#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import vgg16, os, utils, time, argparse
from scipy.misc import imread, imresize
from tf_cnnvis import *

X = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
vgg = vgg16.Vgg16()
vgg.build(X)

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
im = np.expand_dims(imresize(imresize(imread(
  os.path.join("../imgs", args.image)), (256, 256)), (224, 224)), axis = 0)

visual_func = [deconv_visualization, activation_visualization]

start = time.time()
is_success =\
  visual_func[args.method](graph_or_path=tf.get_default_graph(),
                           value_feed_dict={X : im}, layers=layers,
                           path_logdir=logdir, path_outdir=outdir)
start = time.time() - start
print("Total Time = %f" % (start))
