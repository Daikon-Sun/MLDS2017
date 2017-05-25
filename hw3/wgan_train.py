import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
import h5py
import skimage
import skimage.io
import scipy.misc
import random
import json
import os
import shutil
from os.path import join
from Utils import image_processing

save_cnt = 0
vector_name = ['uni_skip','bi_skip','combine_skip','one_hot','glove_50','glove_100','glove_200','glove_300']

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--z_dim', type=int, default=100,
             help='Noise dimension')

  parser.add_argument('--t_dim', type=int, default=256,
             help='Text feature dimension')

  parser.add_argument('--batch_size', type=int, default=64,
             help='Batch Size')

  parser.add_argument('--image_size', type=int, default=64,
             help='Image Size a, a x a')

  parser.add_argument('--gf_dim', type=int, default=64,
             help='Number of conv in the first layer gen.')

  parser.add_argument('--df_dim', type=int, default=64,
             help='Number of conv in the first layer discr.')

  parser.add_argument('--gfc_dim', type=int, default=1024,
             help='Dimension of gen untis for for fully connected layer 1024')

  parser.add_argument('--caption_vector_length', type=int, default=23,
             help='Caption Vector Length')

  parser.add_argument('--data_dir', type=str, default="hw3_data",
             help='Data Directory')

  parser.add_argument('--learning_rate', type=float, default=0.0002,
             help='Learning Rate')

  parser.add_argument('--beta1', type=float, default=0.7,
             help='Momentum for Adam Update')

  parser.add_argument('--epochs', type=int, default=800,
             help='Max number of epochs')

  parser.add_argument('--save_every', type=int, default=100,
             help='Save Model/Samples every x iterations over batches')

  parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

  parser.add_argument('--data_set', type=str, default="bonus",
                       help='Dat set: faces,bonus')
  parser.add_argument('--vector',type=int,default=3,help='method to encode captions, options: 1. uni-skip, 2. bi-skip, 3. combine-skip, 4. one-hot, 5. glove_50 , 6. glove_100 , 7. glove_200 , 8. glove_300')
  parser.add_argument('--update_rate',type=str, default='1_2',help='update rate between discrimminator and generator')
  parser.add_argument('--gan_type', type=int, default=0, help='GAN type: 0->DCGAN, 1->WGAN, 2->LSGAN, 3->BSGAN')

  args = parser.parse_args()
   #check if the caption vector length is correct:
  if args.vector==1:
    args.caption_vector_length=2400
  if args.vector==2:
    args.caption_vector_length=2400
  if args.vector==3:
    args.caption_vector_length=4800
  if args.vector==4:
    args.caption_vector_length=23
  if args.vector==5:
    args.caption_vector_length=100
  if args.vector==6:
    args.caption_vector_length=200
  if args.vector==7:
    args.caption_vector_length=400
  if args.vector==8:
    args.caption_vector_length=600
  
  print(args.caption_vector_length)
  model_options = {
    'z_dim' : args.z_dim,
    't_dim' : args.t_dim,
    'batch_size' : args.batch_size,
    'image_size' : args.image_size,
    'gf_dim' : args.gf_dim,
    'df_dim' : args.df_dim,
    'gfc_dim' : args.gfc_dim,
    'caption_vector_length' : args.caption_vector_length,
    'gan_type' : args.gan_type
  }
 
  #GAN model
  gan = model.GAN(model_options)
  input_tensors, variables, loss, outputs, checks = gan.build_model()
  with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    if args.gan_type == 1: # WGAN
      d_optim = tf.train.RMSPropOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
      g_optim = tf.train.RMSPropOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
    else:
      d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
      g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  sess = tf.InteractiveSession(config=config)
  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver(max_to_keep=None)
  if args.resume_model:
    saver.restore(sess, args.resume_model)

  loaded_data = load_training_data(args.data_dir, args.data_set, args.vector)

  for i in range(args.epochs):
    batch_no = 0
    while batch_no*args.batch_size < loaded_data['data_length']:
      real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size,
        args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, args.vector, loaded_data)

      # DISCR UPDATE ( 5 times for WGAN )
      check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
      if args.gan_type == 1: #WGAN
        _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator'], clip_updates['clip_updates1'],
                                               clip_updates['clip_updates2'], clip_updates['clip_updates3']] + check_ts,
        feed_dict = {
          input_tensors['t_real_image'] : real_images,
          input_tensors['t_wrong_image'] : wrong_images,
          input_tensors['t_real_caption'] : caption_vectors,
          input_tensors['t_z'] : z_noise,
        })

      else:
        _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,

        feed_dict = {
          input_tensors['t_real_image'] : real_images,
          input_tensors['t_wrong_image'] : wrong_images,
          input_tensors['t_real_caption'] : caption_vectors,
          input_tensors['t_z'] : z_noise,
        })

      print("d1", d1)
      print("d2", d2)
      print("d3", d3)
      print("D", d_loss)

      if args.gan_type == 1: #WAGN
        _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator'], clip_updates['clip_updates1'],
                                             clip_updates['clip_updates2'], clip_updates['clip_updates3']] + check_ts,
          feed_dict = {
            input_tensors['t_real_image'] : real_images,
            input_tensors['t_wrong_image'] : wrong_images,
            input_tensors['t_real_caption'] : caption_vectors,
            input_tensors['t_z'] : z_noise,
          })

        print("d1", d1)
        print("d2", d2)
        print("d3", d3)
        print("D", d_loss)

        _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator'], clip_updates['clip_updates1'],
                                             clip_updates['clip_updates2'], clip_updates['clip_updates3']] + check_ts,
          feed_dict = {
            input_tensors['t_real_image'] : real_images,
            input_tensors['t_wrong_image'] : wrong_images,
            input_tensors['t_real_caption'] : caption_vectors,
            input_tensors['t_z'] : z_noise,
          })

        print("d1", d1)
        print("d2", d2)
        print("d3", d3)
        print("D", d_loss)

        _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator'], clip_updates['clip_updates1'],
                                             clip_updates['clip_updates2'], clip_updates['clip_updates3']] + check_ts,
          feed_dict = {
            input_tensors['t_real_image'] : real_images,
            input_tensors['t_wrong_image'] : wrong_images,
            input_tensors['t_real_caption'] : caption_vectors,
            input_tensors['t_z'] : z_noise,
          })

        print("d1", d1)
        print("d2", d2)
        print("d3", d3)
        print("D", d_loss)

        _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator'], clip_updates['clip_updates1'],
                                             clip_updates['clip_updates2'], clip_updates['clip_updates3']] + check_ts,
          feed_dict = {
            input_tensors['t_real_image'] : real_images,
            input_tensors['t_wrong_image'] : wrong_images,
            input_tensors['t_real_caption'] : caption_vectors,
            input_tensors['t_z'] : z_noise,
          })

        print("d1", d1)
        print("d2", d2)
        print("d3", d3)
        print("D", d_loss)

      # GEN UPDATE
      _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
        feed_dict = {
          input_tensors['t_real_image'] : real_images,
          input_tensors['t_wrong_image'] : wrong_images,
          input_tensors['t_real_caption'] : caption_vectors,
          input_tensors['t_z'] : z_noise,
        })
      if args.update_rate=='1_2':
        
        # GEN UPDATE TWICE, to make sure d_loss does not go to 0
        _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
          feed_dict = {
            input_tensors['t_real_image'] : real_images,
            input_tensors['t_wrong_image'] : wrong_images,
            input_tensors['t_real_caption'] : caption_vectors,
            input_tensors['t_z'] : z_noise,
          })
      
      print("LOSSES", d_loss, g_loss, batch_no, i, len(loaded_data['image_list'])/ args.batch_size)
      batch_no += 1
      if (batch_no % args.save_every) == 0:
        #print("Saving Images, Model")
        save_for_vis(args.data_dir, real_images, gen, image_files,args.vector,args.update_rate)
        save_path = saver.save(sess, join(args.data_dir,vector_name[args.vector-1],args.update_rate,"Models/latest_model_{}_temp.ckpt".format(args.data_set)))
    if i%40 == 0:
      save_path = saver.save(sess, join(args.data_dir,vector_name[args.vector-1],args.update_rate,"Models/model_after_{}_epoch_{}.ckpt".format(args.data_set, i)))

def load_training_data(data_dir, data_set, vector):
  
  h = h5py.File(join(data_dir,vector_name[vector-1],'train_vector.hdf5'))
  flower_captions = {}
  for ds in h.items():
    flower_captions[ds[0]] = np.array(ds[1])
  image_list = [key for key in flower_captions]
  image_list.sort()

  img_75 = int(len(image_list)*0.75)
  training_image_list = image_list[0:img_75]
  random.shuffle(training_image_list)

  return {
    'image_list' : training_image_list,
    'captions' : flower_captions,
    'data_length' : len(training_image_list)
  }

def save_for_vis(data_dir, real_images, generated_images, image_files,vector,update_rate):

  global save_cnt
  target_dir = join(vector_name[vector-1],update_rate,'samples','samples{}'.format(save_cnt))
  save_cnt += 1
  #shutil.rmtree( join(data_dir, target_dir) )
  os.makedirs( join(data_dir, target_dir) )
  #print('image_files')
  #print(image_files)
  #print(('len(image_files)',len(image_files)))
  #print(('real_images.shape[0]',real_images.shape[0]))
  for i in range(0, real_images.shape[0]):
    real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
    real_images_255 = (real_images[i,:,:,:])
    scipy.misc.imsave( join(data_dir, target_dir,'{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)
    fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
    fake_images_255 = (generated_images[i,:,:,:])
    scipy.misc.imsave(join(data_dir, target_dir,'fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim,
  caption_vector_length, split, data_dir, data_set, vector,loaded_data = None):
  real_images = np.zeros((batch_size, 64, 64, 3))
  wrong_images = np.zeros((batch_size, 64, 64, 3))
  captions = np.zeros((batch_size, caption_vector_length))

  cnt = 0
  image_files = []
  for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
    idx = i % len(loaded_data['image_list'])
    image_file =  join(data_dir,data_set,loaded_data['image_list'][idx])
    image_array = image_processing.load_image_array(image_file, image_size)
    real_images[cnt,:,:,:] = image_array

    # Improve this selection of wrong image
    wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
    wrong_image_file =  join(data_dir,data_set,loaded_data['image_list'][wrong_image_id])
    wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
    wrong_images[cnt, :,:,:] = wrong_image_array
    if vector==1: 
      captions[cnt,:] =  loaded_data['captions'][ loaded_data['image_list'][idx] ][0][0:caption_vector_length]
    if vector==2:
      captions[cnt,:] =  loaded_data['captions'][ loaded_data['image_list'][idx] ][0][(caption_vector_length+1):]
    if vector>=3:
      captions[cnt,:] =  loaded_data['captions'][ loaded_data['image_list'][idx]][0]
    image_files.append( image_file )
    cnt += 1

  z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
  return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
  main()

