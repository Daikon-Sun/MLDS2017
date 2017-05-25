#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--z_dim', type=int, default=100, help='Noise Dimension')
  parser.add_argument('--t_dim', type=int, default=256,
                      help='Text feature dimension')
  parser.add_argument('--image_size', '-is', type=int, default=64,
                      help='Image Size')
  parser.add_argument('--gf_dim', type=int, default=64,
                      help='Number of conv in the first layer gen.')
  parser.add_argument('--df_dim', type=int, default=64,
                      help='Number of conv in the first layer discr.')
  parser.add_argument('--gfc_dim', type=int, default=1024,
             help='Dimension of gen untis for for fully connected layer 1024')
  parser.add_argument('--caption_vector_length', '-cvl', type=int, default=2400,
                      help='Caption Vector Length')
  parser.add_argument('--data_set', '-ds', type=str, default='faces',
                      help='data directory')
  parser.add_argument('--method_dir', '-md', type=str, default='',
                      help='method directory')
  parser.add_argument('--model_path', '-mp', type=str,
                      default='latest_model_faces_temp.ckpt',
                      help='Trained Model Path')
  parser.add_argument('--n_images', '-ni', type=int, default=5,
                       help='Number of Images per Caption')
  parser.add_argument('--caption_vectors', '-cv', type=str,
                      default='test_caption_vectors.hdf5',
                      help='Caption Thought Vector File')
  parser.add_argument('--out_dir', '-od', type=str, default='samples',
                      help='output directory')

  args = parser.parse_args()
  model_options = {
    'z_dim' : args.z_dim,
    't_dim' : args.t_dim,
    'batch_size' : args.n_images,
    'image_size' : args.image_size,
    'gf_dim' : args.gf_dim,
    'df_dim' : args.df_dim,
    'gfc_dim' : args.gfc_dim,
    'caption_vector_length' : args.caption_vector_length
  }

  gan = model.GAN(model_options)
  _, _, _, _, _ = gan.build_model()
  sess = tf.Session()
  saver = tf.train.Saver()
  saver.restore(sess,
                join(args.data_set, args.method_dir, 'Models', args.model_path))

  input_tensors, outputs = gan.build_generator()

  h = h5py.File(join(args.data_set, args.method_dir, args.caption_vectors))
  caption_image_dic = {}

  for i, key in enumerate(h):
    caption_images = []
    z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])
    caption = np.array([h[key][0, :args.caption_vector_length]
                       for i in range(args.n_images)])

    [gen_image] =\
      sess.run([outputs['generator']],
               feed_dict = {input_tensors['t_real_caption'] : caption,
                            input_tensors['t_z'] : z_noise} )

    caption_image_dic[key] =\
      [gen_image[i, :, :, :] for i in range(0, args.n_images)]

  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
  for key in h:
    for i, im in enumerate(caption_image_dic[key]):
      scipy.misc.imsave(join(args.out_dir, 'sample_'+key+'_'+str(i)+'.jpg'), im)

if __name__ == '__main__':
  main()
