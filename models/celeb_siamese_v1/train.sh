#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/celeb_siamese_v1/ $TOOLS/caffe train \
  --solver=models/celeb_siamese_v1/solver.prototxt \
  --gpu=$1

#  --weights=models/celeb_siamese_v1/celeb_v17_iter_500000.caffemodel \
