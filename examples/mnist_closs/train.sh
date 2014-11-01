#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/mnist_closs/ $TOOLS/caffe train \
  --solver=examples/mnist_closs/solver.prototxt \
  --gpu=$1

#  --snapshot=snap/cfw_closs.kway/snap_iter_20000.solverstate \
