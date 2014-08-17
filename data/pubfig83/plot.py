import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from collections import defaultdict

import numpy as np
import os
import sys
from os.path import join
from time import time
import pickle


dir_names = [ 'out10,ip1.databundle',
              'out11,relu4.databundle' ]
labels = [ 'R11 + NN', 'R13 + NN' ]
n_trns = [1, 2, 3, 5, 10, 20]
mean_accs_list = list()

def ufl_perf():
  # ufl, may 2014, from my email "Here are our numbers at the moment:"
  n_trns = [1, 2, 3, 5, 10, 20, 30, 40, 50]
  mean_accs = [0.1019, 0.1743, 0.2246, 0.3307, 0.4914,
               0.6413, 0.7149, 0.7559, 0.7848]
  return "UFL + SVM", dict(zip(n_trns, mean_accs))

def extract_mean_accs(n_trns, n_trn_vs_acc):
  mean_accs = list()
  for n_trn in n_trns:
    mean_accs.append(n_trn_vs_acc[n_trn])
  return mean_accs


plt.hold(True)

# load & plot saved results
for dir_name in dir_names:
  n_trn_vs_acc_path = '%s/n_trn_vs_acc.pickle' % dir_name
  print n_trn_vs_acc_path

  n_trn_vs_acc = pickle.loads(open(n_trn_vs_acc_path).read())

  mean_accs = extract_mean_accs(n_trns, n_trn_vs_acc)

  mean_accs_list.append(mean_accs)

# load & plot ufl results
label, n_trn_vs_acc = ufl_perf()
labels.append(label)
mean_accs_list.append(extract_mean_accs(n_trns, n_trn_vs_acc))


for i, mean_accs in enumerate(mean_accs_list):
  plt.plot(n_trns, mean_accs, label=labels[i])


plt.axis([0, 21, 0, 1])
plt.xlabel('Number of examples per individual')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. number of examples per individual')
plt.legend(loc='best')
plt.grid(True)

plt.savefig('perf.jpg')



