import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math

from collections import defaultdict

import numpy as np
import os
import sys
from os.path import join
from time import time
import pickle

import h5py

import caffe

from PIL import Image

indir_name = sys.argv[1]
print indir_name


face_id_to_individual_path = '%s/face_id_to_individual.pickle' % indir_name
person_id_to_trntst_path = '%s/person_id_to_trntst.pickle' % indir_name

print face_id_to_individual_path
print person_id_to_trntst_path

face_id_to_individual = pickle.loads(open(face_id_to_individual_path).read())
person_id_to_trntst = pickle.loads(open(person_id_to_trntst_path).read())



score_type = sys.argv[2]

outdir_name = indir_name + ',' + score_type

if not os.path.isdir(outdir_name):
  os.makedirs(outdir_name)


def compscore(v, w, type='normdot'):
  if type == 'dot':
    return v.dot(w)
  elif type == 'normdot':
    if np.max(v) > 0 and np.max(w) > 0:
      v /= math.sqrt((v**2).sum())
      w /= math.sqrt((w**2).sum())
      return v.dot(w)
    else:
      return 0.0
  elif type == 'histdot':  
    if np.max(v) > 0 and np.max(w) > 0:
      v /= v.sum()
      w /= w.sum()
      return v.dot(w)
    else:
      return 0.0
  elif type == 'histint':
    if np.max(v) > 0 and np.max(w) > 0:
      v /= v.sum()
      w /= w.sum()
      return np.minimum(v, w).sum()
    else:
      return 0.0
  elif type == 'int':
    if np.max(v) > 0 and np.max(w) > 0:
      return np.minimum(v, w).sum()
    else:
      return 0.0
  elif type == 'chi-square':
    sqdiff = (v-w)**2
    sum = v+w

    return -np.sum(sqdiff[sqdiff > 0] / sum[sqdiff > 0])
  else:
    print "type unknown"


def compconfusion(n_trn, score_type, person_id_to_trntst, h):

  person_ids = person_id_to_trntst.keys()

  # make a map from person_id to list of trn vecs
  person_id_to_trn_vecs = defaultdict(list)

  for person_id in person_ids:
    trn_keys = person_id_to_trntst[person_id]['trn'][:n_trn]
    for k in trn_keys:
      v = np.array(h[k])
      person_id_to_trn_vecs[person_id].append(v)




  # confusion matrix: confmat[ground_truth, prediction]
  confmat = np.zeros((len(person_id_to_trntst.keys()),
                      len(person_id_to_trntst.keys())))

  # iterate over tst vecs and print dot prod
  for i, person_id in enumerate(person_ids):
    print person_id,
    sys.stdout.flush()

    tst_keys = person_id_to_trntst[person_id]['tst']
    for k in tst_keys:
      # grab the test vector
      v = np.array(h[k])

      best_score = -np.infty
      best_model_id = 0

      # for each person in the trn set, compare it to each trn vec
      for j, model_id in enumerate(person_ids):
        for w in person_id_to_trn_vecs[model_id]:
          score = compscore(v, w, type=score_type)
          if score > best_score:
            best_score = score
            best_model_id = j

      confmat[i, best_model_id] += 1

  print

  for i, person_id in enumerate(person_ids):
    confmat[i] /= len(person_id_to_trntst[person_id]['tst'])

  mean_acc = np.diag(confmat).mean()
  print mean_acc

  return confmat, mean_acc


h = h5py.File('%s/features.hdf5' % indir_name, 'r')

n_trns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
mean_accs = list()

for n_trn in n_trns:
  confmat, mean_acc = compconfusion(n_trn, score_type, person_id_to_trntst, h)
  mean_accs.append(mean_acc)

h.close()

n_trn_vs_acc = dict(zip(n_trns, mean_accs))

with open('%s/n_trn_vs_acc.pickle' % outdir_name, 'w') as fh:
  pickle.dump(n_trn_vs_acc, fh)

