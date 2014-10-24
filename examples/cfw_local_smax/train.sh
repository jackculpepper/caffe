#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/cfw_local_smax/ $TOOLS/caffe train \
  --solver=examples/cfw_local_smax/cfw_solver.prototxt \
  --gpu=$1
