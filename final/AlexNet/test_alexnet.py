#!/usr/bin/python3
import os, sys, time, copy, h5py, argparse, cv2
import numpy as np
import tensorflow as tf
from subprocess import call
from scipy.misc import imread, imresize
from tf_cnnvis import *
import tensorpack as tp
import tensorpack.utils.viz as viz

cv2.ocl.setUseOpenCL(False)

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
mean = np.load('../img_mean.npy').transpose((1, 2, 0))
im = np.expand_dims(imresize(imresize(imread(args.image), (256, 256))-mean,
                             (224, 224)), axis=0)
if args.method <= 1:
  f = h5py.File('./alexnet_weights.h5','r')

  conv_1 = [f['conv_1']['conv_1_W'], f['conv_1']['conv_1_b']]
  conv_2_1 = [f['conv_2_1']['conv_2_1_W'], f['conv_2_1']['conv_2_1_b']]
  conv_2_2 = [f['conv_2_2']['conv_2_2_W'], f['conv_2_2']['conv_2_2_b']]
  conv_3 = [f['conv_3']['conv_3_W'], f['conv_3']['conv_3_b']]
  conv_4_1 = [f['conv_4_1']['conv_4_1_W'], f['conv_4_1']['conv_4_1_b']]
  conv_4_2 = [f['conv_4_2']['conv_4_2_W'], f['conv_4_2']['conv_4_2_b']]
  conv_5_1 = [f['conv_5_1']['conv_5_1_W'], f['conv_5_1']['conv_5_1_b']]
  conv_5_2 = [f['conv_5_2']['conv_5_2_W'], f['conv_5_2']['conv_5_2_b']]
  fc_6 = [f['dense_1']['dense_1_W'], f['dense_1']['dense_1_b']]
  fc_7 = [f['dense_2']['dense_2_W'], f['dense_2']['dense_2_b']]
  fc_8 = [f['dense_3']['dense_3_W'], f['dense_3']['dense_3_b']]

  # tensorflow model implementation (Alexnet convolution)
  X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
  radius = 5; alpha = 1e-4; beta = 0.75; bias = 2.0 # hyper parametes for lrn
  # Layer - 1 conv1
  W_conv_1 = tf.Variable(np.transpose(conv_1[0], (2, 3, 1, 0)))
  b_conv_1 = tf.Variable(np.reshape(conv_1[1], (96, )))
  y_conv_1 = tf.nn.conv2d(X, filter=W_conv_1, strides=[1, 4, 4, 1],
                          padding='SAME') + b_conv_1
  h_conv_1 = tf.nn.relu(y_conv_1, name='conv1')
  h_conv_1 = tf.nn.local_response_normalization(h_conv_1, depth_radius=radius,
                                                alpha=alpha, beta=beta, bias=bias)
  h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='VALID')
  h_pool_1_1, h_pool_1_2 = tf.split(axis=3, value=h_pool_1, num_or_size_splits=2)
  # Layer - 2 conv2
  W_conv_2_1 = tf.Variable(np.transpose(conv_2_1[0], (2, 3, 1, 0)))
  b_conv_2_1 = tf.Variable(np.reshape(conv_2_1[1], (128, )))
  y_conv_2_1 = tf.nn.conv2d(h_pool_1_1, filter=W_conv_2_1, strides=[1, 1, 1, 1],
                            padding='SAME') + b_conv_2_1
  h_conv_2_1 = tf.nn.relu(y_conv_2_1, name='conv2_1')
  h_conv_2_1 =\
    tf.nn.local_response_normalization(h_conv_2_1, depth_radius=radius,
                                       alpha=alpha, beta=beta, bias=bias)
  h_pool_2_1 = tf.nn.max_pool(h_conv_2_1, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
  W_conv_2_2 = tf.Variable(np.transpose(conv_2_2[0], (2, 3, 1, 0)))
  b_conv_2_2 = tf.Variable(np.reshape(conv_2_2[1], (128, )))
  y_conv_2_2 = tf.nn.conv2d(h_pool_1_2, filter=W_conv_2_2, strides=[1, 1, 1, 1],
                            padding='SAME')+b_conv_2_2
  h_conv_2_2 = tf.nn.relu(y_conv_2_2, name='conv2_2')
  h_conv_2_2 =\
    tf.nn.local_response_normalization(h_conv_2_2, depth_radius=radius,
                                       alpha=alpha, beta=beta, bias=bias)
  h_pool_2_2 = tf.nn.max_pool(h_conv_2_2, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
  h_pool_2 = tf.concat(axis=3, values=[h_pool_2_1, h_pool_2_2])
  # Layer - 3 conv3
  W_conv_3 = tf.Variable(np.transpose(conv_3[0], (2, 3, 1, 0)))
  b_conv_3 = tf.Variable(np.reshape(conv_3[1], (384, )))
  y_conv_3 = tf.nn.conv2d(h_pool_2, filter=W_conv_3, strides=[1, 1, 1, 1],
                          padding='SAME') + b_conv_3
  h_conv_3 = tf.nn.relu(y_conv_3, name='conv3')
  h_conv_3_1, h_conv_3_2 = tf.split(axis=3, value=h_conv_3, num_or_size_splits=2)
  # Layer - 4 conv4
  W_conv_4_1 = tf.Variable(np.transpose(conv_4_1[0], (2, 3, 1, 0)))
  b_conv_4_1 = tf.Variable(np.reshape(conv_4_1[1], (192, )))
  y_conv_4_1 = tf.nn.conv2d(h_conv_3_1, filter=W_conv_4_1,
                            strides=[1, 1, 1, 1], padding='SAME') + b_conv_4_1
  h_conv_4_1 = tf.nn.relu(y_conv_4_1, name='conv4_1')
  W_conv_4_2 = tf.Variable(np.transpose(conv_4_2[0], (2, 3, 1, 0)))
  b_conv_4_2 = tf.Variable(np.reshape(conv_4_2[1], (192, )))
  y_conv_4_2 = tf.nn.conv2d(h_conv_3_2, filter=W_conv_4_2,
                            strides=[1, 1, 1, 1], padding='SAME') + b_conv_4_2
  h_conv_4_2 = tf.nn.relu(y_conv_4_2, name='conv4_2')
  h_conv_4 = tf.concat(axis=3, values=[h_conv_4_1, h_conv_4_2])
  # Layer - 5 conv5
  W_conv_5_1 = tf.Variable(np.transpose(conv_5_1[0], (2, 3, 1, 0)))
  b_conv_5_1 = tf.Variable(np.reshape(conv_5_1[1], (128, )))
  y_conv_5_1 = tf.nn.conv2d(h_conv_4_1, filter=W_conv_5_1, strides=[1, 1, 1, 1],
                            padding='SAME') + b_conv_5_1
  h_conv_5_1 = tf.nn.relu(y_conv_5_1, name='conv5_1')
  h_conv_5_1 =\
    tf.nn.local_response_normalization(h_conv_5_1, depth_radius=radius,
                                       alpha=alpha, beta=beta, bias=bias)
  h_pool_5_1 = tf.nn.max_pool(h_conv_5_1, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
  W_conv_5_2 = tf.Variable(np.transpose(conv_5_2[0], (2, 3, 1, 0)))
  b_conv_5_2 = tf.Variable(np.reshape(conv_5_2[1], (128, )))
  y_conv_5_2 = tf.nn.conv2d(h_conv_4_2, filter=W_conv_5_2, strides=[1, 1, 1, 1],
                            padding='SAME') + b_conv_5_2
  h_conv_5_2 = tf.nn.relu(y_conv_5_2, name='conv5_2')
  h_conv_5_2 =\
    tf.nn.local_response_normalization(h_conv_5_2, depth_radius=radius,
                                       alpha=alpha, beta=beta, bias=bias)
  h_pool_5_2 = tf.nn.max_pool(h_conv_5_2, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
  h_pool_5 = tf.concat(axis=3, values=[h_pool_5_1, h_pool_5_2])
  dimensions = h_pool_5.get_shape().as_list()
  dim = dimensions[1] * dimensions[2] * dimensions[3]
  # # Part of Alexnet model which is not required for deconvolution
  h_flatten = tf.reshape(h_pool_5, shape=[-1, dim])
  # Layer - 6 fc6
  W_full_6 = tf.Variable(np.array(fc_6[0]))
  b_full_6 = tf.Variable(np.array(fc_6[1]))
  y_full_6 = tf.add(tf.matmul(h_flatten, W_full_6), b_full_6)
  h_full_6 = tf.nn.relu(y_full_6, name='fc6')
  h_dropout_6 = tf.nn.dropout(h_full_6, 0.5)
  # Layer - 7 fc7
  W_full_7 = tf.Variable(np.array(fc_7[0]))
  b_full_7 = tf.Variable(np.array(fc_7[1]))
  y_full_7 = tf.add(tf.matmul(h_dropout_6, W_full_7), b_full_7)
  h_full_7 = tf.nn.relu(y_full_7, name='fc7')
  h_dropout_7 = tf.nn.dropout(h_full_7, 0.5)
  # Layer - 8 fc8
  W_full_8 = tf.Variable(np.array(fc_8[0]))
  b_full_8 = tf.Variable(np.array(fc_8[1]))
  y_full_8 = tf.add(tf.matmul(h_dropout_7, W_full_8), b_full_8, name='fc8')
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
      orig_image = inputs[0]
      with tp.symbolic_functions.guided_relu():
        f = h5py.File('./alexnet_weights.h5','r')

        conv_1 = [f['conv_1']['conv_1_W'], f['conv_1']['conv_1_b']]
        conv_2_1 = [f['conv_2_1']['conv_2_1_W'], f['conv_2_1']['conv_2_1_b']]
        conv_2_2 = [f['conv_2_2']['conv_2_2_W'], f['conv_2_2']['conv_2_2_b']]
        conv_3 = [f['conv_3']['conv_3_W'], f['conv_3']['conv_3_b']]
        conv_4_1 = [f['conv_4_1']['conv_4_1_W'], f['conv_4_1']['conv_4_1_b']]
        conv_4_2 = [f['conv_4_2']['conv_4_2_W'], f['conv_4_2']['conv_4_2_b']]
        conv_5_1 = [f['conv_5_1']['conv_5_1_W'], f['conv_5_1']['conv_5_1_b']]
        conv_5_2 = [f['conv_5_2']['conv_5_2_W'], f['conv_5_2']['conv_5_2_b']]
        fc_6 = [f['dense_1']['dense_1_W'], f['dense_1']['dense_1_b']]
        fc_7 = [f['dense_2']['dense_2_W'], f['dense_2']['dense_2_b']]
        fc_8 = [f['dense_3']['dense_3_W'], f['dense_3']['dense_3_b']]

        # tensorflow model implementation (Alexnet convolution)
        radius = 5; alpha = 1e-4; beta = 0.75; bias = 2.0
        W_conv_1 = tf.Variable(np.transpose(conv_1[0], (2, 3, 1, 0)))
        b_conv_1 = tf.Variable(np.reshape(conv_1[1], (96, )))
        y_conv_1 = tf.nn.conv2d(inputs, filter=W_conv_1, strides=[1, 4, 4, 1],
                                padding='SAME') + b_conv_1
        h_conv_1 = tf.nn.relu(y_conv_1, name='conv1')
        h_conv_1 =\
          tf.nn.local_response_normalization(h_conv_1, depth_radius=radius,
                                             alpha=alpha, beta=beta, bias=bias)
        h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1], padding='VALID')
        h_pool_1_1, h_pool_1_2 =\
          tf.split(axis=3, value=h_pool_1, num_or_size_splits=2)
        W_conv_2_1 = tf.Variable(np.transpose(conv_2_1[0], (2, 3, 1, 0)))
        b_conv_2_1 = tf.Variable(np.reshape(conv_2_1[1], (128, )))
        y_conv_2_1 =\
          tf.nn.conv2d(h_pool_1_1, filter=W_conv_2_1, strides=[1, 1, 1, 1],
                       padding='SAME') + b_conv_2_1
        h_conv_2_1 = tf.nn.relu(y_conv_2_1, name='conv2_1')
        h_conv_2_1 =\
          tf.nn.local_response_normalization(h_conv_2_1, depth_radius=radius,
                                             alpha=alpha, beta=beta, bias=bias)
        h_pool_2_1 = tf.nn.max_pool(h_conv_2_1, ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1], padding='VALID')
        W_conv_2_2 = tf.Variable(np.transpose(conv_2_2[0], (2, 3, 1, 0)))
        b_conv_2_2 = tf.Variable(np.reshape(conv_2_2[1], (128, )))
        y_conv_2_2 =\
          tf.nn.conv2d(h_pool_1_2, filter=W_conv_2_2, strides=[1, 1, 1, 1],
                       padding='SAME')+b_conv_2_2
        h_conv_2_2 = tf.nn.relu(y_conv_2_2, name='conv2_2')
        h_conv_2_2 =\
          tf.nn.local_response_normalization(h_conv_2_2, depth_radius=radius,
                                             alpha=alpha, beta=beta, bias=bias)
        h_pool_2_2 = tf.nn.max_pool(h_conv_2_2, ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1], padding='VALID')
        h_pool_2 = tf.concat(axis=3, values=[h_pool_2_1, h_pool_2_2])
        W_conv_3 = tf.Variable(np.transpose(conv_3[0], (2, 3, 1, 0)))
        b_conv_3 = tf.Variable(np.reshape(conv_3[1], (384, )))
        y_conv_3 = tf.nn.conv2d(h_pool_2, filter=W_conv_3, strides=[1, 1, 1, 1],
                                padding='SAME') + b_conv_3
        h_conv_3 = tf.nn.relu(y_conv_3, name='conv3')
        h_conv_3_1, h_conv_3_2 = tf.split(axis=3, value=h_conv_3,
                                          num_or_size_splits=2)
        W_conv_4_1 = tf.Variable(np.transpose(conv_4_1[0], (2, 3, 1, 0)))
        b_conv_4_1 = tf.Variable(np.reshape(conv_4_1[1], (192, )))
        y_conv_4_1 =\
          tf.nn.conv2d(h_conv_3_1, filter=W_conv_4_1, strides=[1, 1, 1, 1],
                       padding='SAME')+b_conv_4_1
        h_conv_4_1 = tf.nn.relu(y_conv_4_1, name='conv4_1')
        W_conv_4_2 = tf.Variable(np.transpose(conv_4_2[0], (2, 3, 1, 0)))
        b_conv_4_2 = tf.Variable(np.reshape(conv_4_2[1], (192, )))
        y_conv_4_2 =\
          tf.nn.conv2d(h_conv_3_2, filter=W_conv_4_2, strides=[1, 1, 1, 1],
                       padding='SAME')+b_conv_4_2
        h_conv_4_2 = tf.nn.relu(y_conv_4_2, name='conv4_2')
        h_conv_4 = tf.concat(axis=3, values=[h_conv_4_1, h_conv_4_2])
        W_conv_5_1 = tf.Variable(np.transpose(conv_5_1[0], (2, 3, 1, 0)))
        b_conv_5_1 = tf.Variable(np.reshape(conv_5_1[1], (128, )))
        y_conv_5_1 =\
          tf.nn.conv2d(h_conv_4_1, filter=W_conv_5_1, strides=[1, 1, 1, 1],
                       padding='SAME')+b_conv_5_1
        h_conv_5_1 = tf.nn.relu(y_conv_5_1, name='conv5_1')
        h_conv_5_1 =\
          tf.nn.local_response_normalization(h_conv_5_1, depth_radius=radius,
                                             alpha=alpha, beta=beta, bias=bias)
        h_pool_5_1 = tf.nn.max_pool(h_conv_5_1, ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1], padding='VALID')
        W_conv_5_2 = tf.Variable(np.transpose(conv_5_2[0], (2, 3, 1, 0)))
        b_conv_5_2 = tf.Variable(np.reshape(conv_5_2[1], (128, )))
        y_conv_5_2 =\
          tf.nn.conv2d(h_conv_4_2, filter=W_conv_5_2,
                       strides=[1, 1, 1, 1], padding='SAME')+b_conv_5_2
        h_conv_5_2 = tf.nn.relu(y_conv_5_2, name='conv5_2')
        h_conv_5_2 =\
          tf.nn.local_response_normalization(h_conv_5_2, depth_radius=radius,
                                             alpha=alpha, beta=beta, bias=bias)
        h_pool_5_2 = tf.nn.max_pool(h_conv_5_2, ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1], padding='VALID')
        h_pool_5 = tf.concat(axis=3, values=[h_pool_5_1, h_pool_5_2])
        dimensions = h_pool_5.get_shape().as_list()
        dim = dimensions[1] * dimensions[2] * dimensions[3]
        h_flatten = tf.reshape(h_pool_5, shape=[-1, dim])
        W_full_6 = tf.Variable(np.array(fc_6[0]))
        b_full_6 = tf.Variable(np.array(fc_6[1]))
        y_full_6 = tf.add(tf.matmul(h_flatten, W_full_6), b_full_6)
        h_full_6 = tf.nn.relu(y_full_6, name='fc6')
        h_dropout_6 = tf.nn.dropout(h_full_6, 0.5)
        W_full_7 = tf.Variable(np.array(fc_7[0]))
        b_full_7 = tf.Variable(np.array(fc_7[1]))
        y_full_7 = tf.add(tf.matmul(h_dropout_6, W_full_7), b_full_7)
        h_full_7 = tf.nn.relu(y_full_7, name='fc7')
        h_dropout_7 = tf.nn.dropout(h_full_7, 0.5)
        W_full_8 = tf.Variable(np.array(fc_8[0]))
        b_full_8 = tf.Variable(np.array(fc_8[1]))
        logits = tf.add(tf.matmul(h_dropout_7, W_full_8), b_full_8, name='fc8')
        tp.symbolic_functions.saliency_map(logits, orig_image, name='saliency')

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
    cv2.imwrite(args.prefix+'_abs.jpg', abs_saliency)

    rsl = im * 0.2 + abs_saliency * 0.8
    cv2.imwrite(args.prefix+'_blended.jpg', rsl)

  run('model/tmp-model', args.image)
