#!/usr/bin/env python

import os, sys

sys.path.append('/nfs/jack/github/spearmint/spearmint/spearmint')
sys.path.append('/nfs/jack/github/spearmint/spearmint')
sys.path.append('/nfs/jack/github/caffe/python')
from chooser.GPEIOptChooser import GPEIOptChooser
from hype import Drivr

if len(sys.argv) < 2:
    raise Exception('Args: experiment')
folder = sys.argv[1]
experiment = sys.argv[2]

def main():
    path = folder + '/' + experiment
    chooser = GPEIOptChooser(path);
    chooser.mcmc_iters = 10
    # chooser.noiseless = False
    # chooser.use_multiprocessing = False

    drivr = Drivr(chooser, folder, experiment)
    drivr.loop()

if __name__ == '__main__':
    main()
