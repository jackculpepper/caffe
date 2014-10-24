#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/cfw_closs/ $TOOLS/caffe train \
  --solver=examples/cfw_closs/cfw_solver.prototxt \
  --gpu=$1

#  --snapshot=snap/cfw_closs/snap_iter_25000.solverstate \
