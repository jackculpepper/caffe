#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/cfw_local/ $TOOLS/caffe train \
  --solver=examples/cfw_local/solver.prototxt \
  --gpu=$1

#  --snapshot=snap/cfw_local/snap_iter_25000.solverstate \
