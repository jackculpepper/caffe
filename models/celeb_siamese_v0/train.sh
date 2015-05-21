#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/celeb_siamese_v0/ $TOOLS/caffe train \
  --solver=models/celeb_siamese_v0/solver.prototxt \
  --gpu=$1


