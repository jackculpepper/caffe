#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/triplet/ \
  $TOOLS/caffe train --solver=examples/triplet/mnist_triplet_solver.prototxt

