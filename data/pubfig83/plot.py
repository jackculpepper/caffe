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

n_trns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
legend = [ 'R11', 'R13' ]


plt.hold(True)

for dir_name in dir_names:

  n_trn_vs_acc_path = '%s/n_trn_vs_acc.pickle' % dir_name

  print n_trn_vs_acc_path

  n_trn_vs_acc = pickle.loads(open(n_trn_vs_acc_path).read())

  mean_accs = list()
  for n_trn in n_trns:
    mean_accs.append(n_trn_vs_acc[n_trn])

  plt.plot(n_trns, mean_accs)


plt.axis([0, 21, 0, 1])
plt.xlabel('Number of examples per individual')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. number of examples per individual')
plt.legend(legend, loc='best')
plt.grid(True)

plt.savefig('perf.jpg')



