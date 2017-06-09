import os, sys, time, copy, h5py
import numpy as np
from tf_cnnvis import *
import tensorflow as tf
from subprocess import call
from scipy.misc import imread, imresize

# importing InceptionV5 model
with tf.gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = t_input-imagenet_mean
tf.import_graph_def(graph_def, {'input' : t_preprocessed})
# reading sample image
im = np.expand_dims(imresize(imread(os.path.join("../imgs", "images.jpg")), (224, 224)), axis = 0)
# deepdream visualization
layer = "import/softmax2_pre_activation"
# api call
start = time.time()
is_success = deepdream_visualization(graph_or_path=tf.get_default_graph(), value_feed_dict = {t_input : im}, layer=layer,
                                     classes = [1, 2, 3, 4, 5], path_logdir="./Log", path_outdir="./Output")
start = time.time() - start
print("Total Time = %f" % (start))
