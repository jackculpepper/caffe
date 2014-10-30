#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=examples/cfw_closs_top/ $TOOLS/caffe train \
  --snapshot=snap/cfw_closs.kway/snap_iter_20000.solverstate \
  --solver=examples/cfw_closs_top/cfw_solver.prototxt \
  --gpu=$1

