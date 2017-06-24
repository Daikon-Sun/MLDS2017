#!/usr/bin/python3
import os, sys, time, copy, h5py, argparse, cv2, re, six
import numpy as np
from tf_cnnvis import *
import tensorflow as tf
from subprocess import call
from scipy.misc import imread, imresize
import tensorpack as tp
from tensorpack import *
import tensorpack.utils.viz as viz
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.dataflow.dataset import ILSVRCMeta
from tensorpack.tfutils import get_tensors_by_names
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

MODEL_DEPTH = None

def get_inference_augmentor():
  meta = ILSVRCMeta()
  pp_mean = meta.get_per_pixel_mean()
  pp_mean_224 = pp_mean[16:-16, 16:-16, :]

  transformers = imgaug.AugmentorList([
    imgaug.ResizeShortestEdge(256),
    imgaug.CenterCrop((224, 224)),
    imgaug.MapImage(lambda x: x - pp_mean_224),
  ])
  return transformers

def name_conversion(caffe_layer_name):
  """ Convert a caffe parameter name to a tensorflow parameter name as
    defined in the above model """
  # beginning & end mapping
  NAME_MAP = {'bn_conv1/beta': 'conv0/bn/beta',
        'bn_conv1/gamma': 'conv0/bn/gamma',
        'bn_conv1/mean/EMA': 'conv0/bn/mean/EMA',
        'bn_conv1/variance/EMA': 'conv0/bn/variance/EMA',
        'conv1/W': 'conv0/W', 'conv1/b': 'conv0/b',
        'fc1000/W': 'fc1000/W', 'fc1000/b': 'fc1000/b'}
  if caffe_layer_name in NAME_MAP:
    return NAME_MAP[caffe_layer_name]

  s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
  if s is None:
    s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
    layer_block_part1 = s.group(3)
    layer_block_part2 = s.group(4)
    assert layer_block_part1 in ['a', 'b']
    layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
  else:
    layer_block = ord(s.group(3)) - ord('a')
  layer_type = s.group(1)
  layer_group = s.group(2)

  layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
  assert layer_branch in [1, 2]
  if layer_branch == 2:
    layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
    layer_id = ord(layer_id) - ord('a') + 1

  TYPE_DICT = {'res': 'conv', 'bn': 'bn'}

  tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
  layer_type = TYPE_DICT[layer_type] + \
    (str(layer_id) if layer_branch == 2 else 'shortcut')
  tf_name = 'group{}/block{}/{}'.format(
    int(layer_group) - 2, layer_block, layer_type) + tf_name
  return tf_name
parser = argparse.ArgumentParser()
parser.add_argument('--prefix', '-p', type=str, default='',
          help='prefix of log and output directory')
parser.add_argument('--layers', '-l', type=str, default='rpc',
          help='layers to be visualized')
parser.add_argument('--method', '-m', type=int, default=0,
          choices=(0, 1, 2), help='methods to visualize')
parser.add_argument('--image', '-i', type=str, default='NULL',
          help='input image')
parser.add_argument('--depth', '-d', help='resnet depth', required=True,
          type=int, choices=[50, 101, 152])
args = parser.parse_args()
MODEL_DEPTH = args.depth
layers = list(args.layers)
logdir = args.prefix+'_Log'
outdir = args.prefix+'_Output'
param = np.load('npy/ResNet'+str(args.depth)+'.npy', encoding='latin1').item()
params = {}
for k, v in six.iteritems(param):
  try:
    newname = name_conversion(k)
  except:
    logger.error("Exception when processing caffe layer {}".format(k))
    raise
  #logger.info("Name Transform: " + k + ' --> ' + newname)
  params[newname] = v

if args.method <= 1:
  class Model(ModelDesc):
    def _get_inputs(self):
      return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
          InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, input_vars):
      image, label = input_vars

      def shortcut(l, n_in, n_out, stride):
        if n_in != n_out:
          l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
          return BatchNorm('bnshortcut', l)
        else:
          return l

      def bottleneck(l, ch_out, stride, preact):
        ch_in = l.get_shape().as_list()[-1]
        input = l
        if preact == 'both_preact':
          l = tf.nn.relu(l, name='preact-relu')
          input = l
        l = Conv2D('conv1', l, ch_out, 1, stride=stride)
        l = BatchNorm('bn1', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv2', l, ch_out, 3)
        l = BatchNorm('bn2', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv3', l, ch_out * 4, 1)
        l = BatchNorm('bn3', l)  # put bn at the bottom
        return l + shortcut(input, ch_in, ch_out * 4, stride)

      def layer(l, layername, features, count, stride, first=False):
        with tf.variable_scope(layername):
          with tf.variable_scope('block0'):
            l = bottleneck(l, features, stride,
                     'no_preact' if first else 'both_preact')
          for i in range(1, count):
            with tf.variable_scope('block{}'.format(i)):
              l = bottleneck(l, features, 1, 'both_preact')
          return l

      cfg = {
        50: ([3, 4, 6, 3]),
        101: ([3, 4, 23, 3]),
        152: ([3, 8, 36, 3])
      }
      defs = cfg[MODEL_DEPTH]

      with argscope(Conv2D, nl=tf.identity, use_bias=False,
              W_init=variance_scaling_initializer(mode='FAN_OUT')):
        # tensorflow with padding=SAME will by default pad [2,3] here.
        # but caffe conv with stride will pad [3,3]
        image = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]])
        fc1000 = (LinearWrap(image)
             .Conv2D('conv0', 64,7, stride=2, nl=BNReLU, padding='VALID')
             .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
             .apply(layer, 'group0', 64, defs[0], 1, first=True)
             .apply(layer, 'group1', 128, defs[1], 2)
             .apply(layer, 'group2', 256, defs[2], 2)
             .apply(layer, 'group3', 512, defs[3], 2)
             .tf.nn.relu()
             .GlobalAvgPooling('gap')
             .FullyConnected('fc1000', 1000, nl=tf.identity)())
      prob = tf.nn.softmax(fc1000, name='prob')
      nr_wrong = prediction_incorrect(fc1000, label, name='wrong-top1')
      nr_wrong = prediction_incorrect(fc1000, label, 5, name='wrong-top5')
  pred_config = PredictConfig(
    model=Model(),
    session_init=DictRestore(params),
    input_names=['input'],
    output_names=['prob']
  )
  predict_func = OfflinePredictor(pred_config)

  prepro = get_inference_augmentor()
  im = cv2.imread(args.image).astype('float32')
  im = prepro.augment(im)
  im = np.reshape(im, (1, 224, 224, 3))

  visual_func = [deconv_visualization, activation_visualization]
  with predict_func.graph.as_default():
    input_tensor = get_tensors_by_names(pred_config.input_names)
    is_success =\
      visual_func[args.method](
      graph_or_path=tf.get_default_graph(),
      value_feed_dict={input_tensor[0] : im},
      layers=layers, path_logdir=logdir, path_outdir=outdir)
else:

  IMAGE_SIZE = 224

  class Model(tp.ModelDesc):
      def _get_inputs(self):
          return [tp.InputDesc(tf.float32, (IMAGE_SIZE, IMAGE_SIZE, 3), 'image')]

      def _build_graph(self, inputs):
          orig_image = inputs[0]
          mean = tf.get_variable('resnet_v1_'+str(args.depth)+'/mean_rgb', shape=[3])
          with tp.symbolic_functions.guided_relu():
              with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False)):
                  image = tf.expand_dims(orig_image - mean, 0)
                  if args.depth == 50:
                    logits, _ = resnet_v1.resnet_v1_50(image, 1000)
                  elif args.depth == 101:
                    logits, _ = resnet_v1.resnet_v1_101(image, 1000)
                  else:
                    logits, _ = resnet_v1.resnet_v1_152(image, 1000)
              tp.symbolic_functions.saliency_map(logits, orig_image, name="saliency")


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

      abs_saliency = viz.intensity_to_rgb(abs_saliency, normalize=True)[:, :, ::-1]  # bgr
      cv2.imwrite(args.prefix+'_abs-saliency.jpg', abs_saliency)

      rsl = im * 0.2 + abs_saliency * 0.8
      cv2.imwrite(args.prefix+'_blended.jpg', rsl)

  run('ckpts/resnet_v1_'+str(args.depth)+'.ckpt', args.image)
