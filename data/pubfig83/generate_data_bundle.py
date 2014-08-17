import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import os
import sys
from os.path import join
from time import time
import pickle

import h5py

import caffe

from PIL import Image

caffe_root = '../../'
net_root = caffe_root + 'examples/cfw/'
data_root = caffe_root + 'data/cfw/'
subdirectory = '4layers_v5/'

model_file = net_root + 'cfw_deploy_v5.prototxt'
pretrained_file = net_root + subdirectory + 'cfw_iter_457000'
mean_file = data_root + 'mean.npy'

blob_name = 'prob'
blob_name = 'ip2'
blob_name = 'ip1'
blob_name = 'relu4'

flag_plot_pool3 = False

DEPLOY_FILE = 'cfw_deploy_v5.prototxt'
MEAN_FILE = 'mean.binaryproto'

SRC_DIR = './pubfig83_vj/'
DST_DIR = './pubfig83_vj.databundle/'


net = caffe.Classifier(model_file,
                       pretrained_file,
                       mean_file=mean_file,
                       channel_swap=(2, 1, 0),
                       input_scale=0.00390625)

net.set_phase_train()
net.set_mode_cpu()


def vis_filters_rect(data, n, padsize=1, padval=0):    
  # make a rectangle by dividing one dimension by the other
  m = data.shape[0] / n
  padding = ((0, n * m - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
  # tile the filters into an image
  data = data.reshape((n, m) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], m * data.shape[3]) + data.shape[4:])
  return data



def normalize_matrix(data, type='global'):
  if type == 'global':
    data -= data.min()
    data /= data.max()
  elif type == 'rowwise':
    for i in range(data.shape[0]):
      data[i, :] /= abs(data[i, :]).max()
  elif type == 'colwise':
    for i in range(data.shape[1]):
      data[:, i] /= abs(data[:, i]).max()
  else:
    print "norm type unknown"
  return data



def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i+n]

outdir_name = sys.argv[1] + "," + blob_name + ".databundle"
if not os.path.isdir(outdir_name):
  os.makedirs(outdir_name)

h = h5py.File('%s/features.hdf5' % outdir_name, 'w')

face_id_to_individual = dict()
person_id_to_trntst = dict()

CELEB_LIST = filter(lambda x: not x.startswith('.'), os.listdir(SRC_DIR))

#CELEB_LIST = ['Claudia_Schiffer', 'Jennifer_Love_Hewitt', 'Shakira']


for c in CELEB_LIST:
  IMAGE_LIST = filter(lambda x: not x.startswith('.'), os.listdir(SRC_DIR + c))

  input_images = [caffe.io.load_image(join(SRC_DIR, c, img)) for img in IMAGE_LIST]
  print len(input_images), "images of", c

  input_images_dict = dict(zip(IMAGE_LIST, input_images))

  feature_list = []

  for img_list in chunks(IMAGE_LIST, 64):

    # build list of 64 images
    input_images = [ input_images_dict[k] for k in img_list ]

    t0 = time()
    f = net.compute_features(input_images, blob_name=blob_name)
    print "took", time() - t0, "s", "to compute", len(img_list), "features"

    # truncate
    f = f[:len(img_list)]
    feature_list.append(f.copy())



    if flag_plot_pool3:
      # extract conv activations
      act_name = 'pool3'
      act_name = 'ip1'
      activations = net.blobs[act_name].data
      # truncate
      activations = activations[:len(img_list)]

      for i, k in enumerate(img_list):
        X = activations[i]
        X = vis_filters_rect(X, 5)
        X = normalize_matrix(X, type='global')
        X = np.uint8(X*255.0)
        filename = '%s_%s.png' % (k, act_name)
        Image.fromarray(X).save('%s/%s' % (outdir_name, filename))



    print np.argmax(f, axis=1)

    for i, k in enumerate(img_list):
      h.require_dataset(name=k, shape=f[i].shape, dtype=f[i].dtype, data=f[i])

  X = np.vstack(feature_list)

  if blob_name == "ip2":
    X = normalize_matrix(X, type='global')
  if blob_name == "relu4":
    X = normalize_matrix(X, type='global')
  if blob_name == "ip1":
    X = normalize_matrix(X, type='global')

  X = np.uint8(X*255.0)
  Image.fromarray(X).save('%s' % outdir_name + '/f_' + c + '.png')


  if False:
    plt.clf()
    plt.imshow(f)
    plt.savefig('%s/f_' % outdir_name + c + '.jpg', dpi=1000)

  # add this celeb to the face_id_to_individual map
  for k in IMAGE_LIST:
    face_id_to_individual[k] = c

  # create a trn/tst split for this celeb
  np.random.shuffle(IMAGE_LIST)
  person_id_to_trntst[c] = dict()
  person_id_to_trntst[c]['trn'] = IMAGE_LIST[:60]
  person_id_to_trntst[c]['tst'] = IMAGE_LIST[60:]
  


with open('%s/face_id_to_individual.pickle' % outdir_name, 'w') as fh:
  pickle.dump(face_id_to_individual, fh)

with open('%s/person_id_to_trntst.pickle' % outdir_name, 'w') as fh:
  pickle.dump(person_id_to_trntst, fh)

h.close()



#      for j in range(activations.shape[1]):
#        X = activations[i, j, :, :]
#        X = normalize_matrix(X, type='global')
#        X = np.uint8(X*255.0)
#        filename = '%s_%s_chan_%03d.png' % (k, act_name, j)
#        Image.fromarray(X).save('%s/%s' % (outdir_name, filename))


