#!/usr/bin/python3
import os, sys, time, copy, h5py, argparse, cv2
import numpy as np
from tf_cnnvis import *
import tensorflow as tf
from subprocess import call
from scipy.misc import imread, imresize
import tensorpack as tp
import tensorpack.utils.viz as viz
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception

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
im = np.expand_dims(imresize(imread(args.image), (224, 224)), axis = 0)
imagenet_mean = 117.0

if args.method <= 1:
  with tf.gfile.FastGFile('tensorflow_inception_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  t_input = tf.placeholder(np.float32, name='input') # define the input tensor
  t_preprocessed = t_input-imagenet_mean
  tf.import_graph_def(graph_def, {'input' : t_preprocessed})

  visual_func = [deconv_visualization, activation_visualization]
  is_success =\
    visual_func[args.method](graph_or_path=tf.get_default_graph(),
                             value_feed_dict={t_input : im}, layers=layers,
                             path_logdir=logdir, path_outdir=outdir)
else:
  IMAGE_SIZE = 224
  class Model(tp.ModelDesc):
    def _get_inputs(self):
      return [tp.InputDesc(tf.float32, (IMAGE_SIZE, IMAGE_SIZE, 3), 'image')]

    def _build_graph(self, inputs):
      orig_image = inputs[0]
      with tp.symbolic_functions.guided_relu():
        with slim.arg_scope(inception.inception_v1_arg_scope()):
          image = tf.expand_dims(((orig_image / 255) - 0.5) * 2, 0)
          logits, _ = inception.inception_v1(image, 1001, False)
        tp.symbolic_functions.saliency_map(logits, orig_image, name="saliency")

  def run(model_path, image_path):
    predictor = tp.OfflinePredictor(tp.PredictConfig(
      model=Model(),
      session_init=tp.get_model_loader(model_path),
      input_names=['image'],
      output_names=['saliency']))
    im = cv2.imread(image_path)
    assert im is not None and im.ndim == 3, image_path

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

  run('ckpts/inception_v1.ckpt', args.image)
