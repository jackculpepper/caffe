#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/cfw_closs/ $TOOLS/caffe train \
  --snapshot=snap/cfw_closs.kway/snap_iter_20000.solverstate \
  --solver=examples/cfw_closs/cfw_solver.prototxt \
  --gpu=$1

