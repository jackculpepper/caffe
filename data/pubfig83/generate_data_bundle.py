import numpy as np
import os
from os.path import join
from time import time
import pickle

import h5py

from iqface.bundle.feature import CaffeFeatureBundle

import caffe

caffe_root = '../../'
net_root = caffe_root + 'examples/cfw/'
data_root = caffe_root + 'data/cfw/'
subdirectory = '4layers_v5/'

model_file = net_root + 'cfw_deploy_v5.prototxt'
pretrained_file = net_root + subdirectory + 'cfw_iter_457000'
mean_file = data_root + 'mean.npy'


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



def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i+n]


h = h5py.File('features.hdf5', 'w')
face_id_to_individual = dict()
person_id_to_trntst = dict()

CELEB_LIST = filter(lambda x: not x.startswith('.'), os.listdir(SRC_DIR))

for c in CELEB_LIST:
  IMAGE_LIST = filter(lambda x: not x.startswith('.'), os.listdir(SRC_DIR + c))

  input_images = [caffe.io.load_image(join(SRC_DIR, c, img)) for img in IMAGE_LIST]
  print len(input_images), "images of", c

  input_images_dict = dict(zip(IMAGE_LIST, input_images))

  for img_list in chunks(IMAGE_LIST, 64):

    # build list of 64 images
    input_images = [ input_images_dict[k] for k in img_list ]

    t0 = time()
    f = net.compute_features(input_images, blob_name='prob')
    print "took", time() - t0, "s", "to compute", len(img_list), "features"

    # truncate
    f = f[:len(img_list)]

    print np.argmax(f, axis=1)

    for i, k in enumerate(img_list):
      h.require_dataset(name=k, shape=f[i].shape, dtype=f[i].dtype, data=f[i])

  # add this celeb to the face_id_to_individual map
  for k in IMAGE_LIST:
    face_id_to_individual[k] = c

  # create a trn/tst split for this celeb
  np.random.shuffle(IMAGE_LIST)
  person_id_to_trntst[c] = dict()
  person_id_to_trntst[c]['trn'] = IMAGE_LIST[:60]
  person_id_to_trntst[c]['tst'] = IMAGE_LIST[60:]
  


with open('face_id_to_individual.pickle', 'w') as fh:
  pickle.dump(face_id_to_individual, fh)

with open('person_id_to_trntst.pickle', 'w') as fh:
  pickle.dump(person_id_to_trntst, fh)

h.close()



