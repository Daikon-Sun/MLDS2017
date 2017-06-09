import numpy as np
import tensorflow as tf
import vgg16, os, utils, time
from scipy.misc import imread, imresize
from tf_cnnvis import *

X = tf.placeholder(tf.float32, shape = [None, 224, 224, 3]) # placeholder for input images
vgg = vgg16.Vgg16()
vgg.build(X)

layers = ['r', 'p', 'c']
im = np.expand_dims(imresize(imresize(imread(os.path.join("../imgs", "car.jpg")), (256, 256)), (224, 224)), axis = 0)

start = time.time()
is_success = deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {X : im}, layers=layers, path_logdir="./Log", path_outdir="./Output")
start = time.time() - start
print("Total Time = %f" % (start))

#batch = img1.reshape((1, 224, 224, 3))
#
# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
#with tf.device('/cpu:0'):
#  with tf.Session() as sess:
#    images = tf.placeholder("float", batch.shape)
#    feed_dict = {images : batch}
#
#    vgg = vgg16.Vgg16()
#    with tf.name_scope("content_vgg"):
#      vgg.build(images)
#
#      prob = sess.run(vgg.prob, feed_dict=feed_dict)
#      print(prob)
#      utils.print_prob(prob[0], './synset.txt')
#      utils.print_prob(prob[1], './synset.txt')
