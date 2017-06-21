#!/usr/bin/python3
import os, time, argparse, inspect, cv2
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from tf_cnnvis import *
import tensorpack as tp
import tensorpack.utils.viz as viz

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
  def __init__(self, vgg19_npy_path=None):
    if vgg19_npy_path is None:
      path = inspect.getfile(Vgg19)
      path = os.path.abspath(os.path.join(path, os.pardir))
      path = os.path.join(path, "vgg19.npy")
      vgg19_npy_path = path
      print(vgg19_npy_path)

    self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
    print("npy file loaded")

  def build(self, rgb):
    """
    load variable from npy to build the VGG

    :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
    """

    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
      blue - VGG_MEAN[0],
      green - VGG_MEAN[1],
      red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    self.conv1_1 = self.conv_layer(bgr, "conv1_1")
    self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
    self.pool1 = self.max_pool(self.conv1_2, 'pool1')

    self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
    self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
    self.pool2 = self.max_pool(self.conv2_2, 'pool2')

    self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
    self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
    self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
    self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
    self.pool3 = self.max_pool(self.conv3_4, 'pool3')

    self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
    self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
    self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
    self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
    self.pool4 = self.max_pool(self.conv4_4, 'pool4')

    self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
    self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
    self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
    self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
    self.pool5 = self.max_pool(self.conv5_4, 'pool5')

    self.fc6 = self.fc_layer(self.pool5, "fc6")
    assert self.fc6.get_shape().as_list()[1:] == [4096]
    self.relu6 = tf.nn.relu(self.fc6)

    self.fc7 = self.fc_layer(self.relu6, "fc7")
    self.relu7 = tf.nn.relu(self.fc7)

    self.fc8 = self.fc_layer(self.relu7, "fc8")

    self.prob = tf.nn.softmax(self.fc8, name="prob")

    self.data_dict = None
    print(("build model finished: %ds" % (time.time() - start_time)))

  def avg_pool(self, bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)

  def conv_layer(self, bottom, name):
    with tf.variable_scope(name):
      filt = self.get_conv_filter(name)

      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

      conv_biases = self.get_bias(name)
      bias = tf.nn.bias_add(conv, conv_biases)

      relu = tf.nn.relu(bias)
      return relu

  def fc_layer(self, bottom, name):
    with tf.variable_scope(name):
      shape = bottom.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim *= d
      x = tf.reshape(bottom, [-1, dim])

      weights = self.get_fc_weight(name)
      biases = self.get_bias(name)

      # Fully connected layer. Note that the '+' operation automatically
      # broadcasts the biases.
      fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

      return fc

  def get_conv_filter(self, name):
    return tf.constant(self.data_dict[name][0], name="filter")

  def get_bias(self, name):
    return tf.constant(self.data_dict[name][1], name="biases")

  def get_fc_weight(self, name):
    return tf.constant(self.data_dict[name][0], name="weights")


parser = argparse.ArgumentParser()
parser.add_argument('--prefix', '-p', type=str, default='',
                    help='prefix of log and output directory')
parser.add_argument('--layers', '-l', type=str, default='rpc',
                    help='layers to be visualized')
parser.add_argument('--method', '-m', type=int, default=0,
                    choices=(0, 1, 2), help='methods to visualize')
parser.add_argument('--image', '-i', type=str, default='NULL',
                    help='input image')
args = parser.parse_args()
layers = list(args.layers)
logdir = args.prefix+'_Log'
outdir = args.prefix+'_Output'
im = np.expand_dims(imresize(imresize(imread(args.image), (256, 256)), (224, 224)), axis = 0)

if args.method <= 1:
  X = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
  vgg = Vgg19()
  vgg.build(X)
  visual_func = [deconv_visualization, activation_visualization]
  is_success =\
    visual_func[args.method](graph_or_path=tf.get_default_graph(),
                             value_feed_dict={X : im}, layers=layers,
                             path_logdir=logdir, path_outdir=outdir)
else:
  IMAGE_SIZE = 224
  class Model(tp.ModelDesc):
    def _get_inputs(self):
      return [tp.InputDesc(tf.float32, (IMAGE_SIZE, IMAGE_SIZE, 3), 'image')]

    def _build_graph(self, inputs):
      with tp.symbolic_functions.guided_relu():
        vgg = Vgg19()
        vgg.build(tf.expand_dims(inputs[0], 0))
        tp.symbolic_functions.saliency_map(vgg.fc8, inputs[0], name='saliency')
  def run(model_path, image_path):
    predictor = tp.OfflinePredictor(tp.PredictConfig(
      model=Model(),
      session_init=tp.get_model_loader(model_path),
      input_names=['image'],
      output_names=['saliency']))
    im = cv2.imread(image_path)
    assert im is not None and im.ndim == 3, image_path

    # resnet expect RGB inputs of 224x224x3
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = im.astype(np.float32)[:, :, ::-1]

    saliency_images = predictor([im])[0]

    abs_saliency = np.abs(saliency_images).max(axis=-1)
    pos_saliency = np.maximum(0, saliency_images)
    neg_saliency = np.maximum(0, -saliency_images)

    pos_saliency -= pos_saliency.min()
    pos_saliency /= pos_saliency.max()
    cv2.imwrite(args.prefix+'_pos.jpg', pos_saliency * 255)

    neg_saliency -= neg_saliency.min()
    neg_saliency /= neg_saliency.max()
    cv2.imwrite(args.prefix+'_neg.jpg', neg_saliency * 255)
    # bgr
    abs_saliency = viz.intensity_to_rgb(abs_saliency, normalize=True)[:, :, ::-1]
    cv2.imwrite(args.prefix+'_abs-saliency.jpg', abs_saliency)

    rsl = im * 0.2 + abs_saliency * 0.8
    cv2.imwrite(args.prefix+'_blended.jpg', rsl)
  run('vgg19.npy', args.image)
