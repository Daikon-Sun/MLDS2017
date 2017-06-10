import numpy as np
import tensorflow as tf
import vgg19, os, utils, time
from scipy.misc import imread, imresize
from tf_cnnvis import *

X = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
vgg = vgg19.Vgg19()
vgg.build(X)

layers = ['r', 'p', 'c']
im = np.expand_dims(imresize(imresize(imread(
  os.path.join("../imgs", "keyboard.jpg")), (256, 256)), (224, 224)), axis = 0)
start = time.time()
is_success = deconv_visualization(graph_or_path=tf.get_default_graph(),
                                  value_feed_dict={X : im}, layers=layers,
                                  path_logdir="./Keyboard_Log",
                                  path_outdir="./Keyboard_Output")
start = time.time() - start
print("Total Time = %f" % (start))
