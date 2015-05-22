#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/celeb_siamese_v0/ $TOOLS/caffe train \
  --solver=models/celeb_siamese_v0/solver.prototxt \
  --weights=models/celeb_siamese_v0/celeb_v1_iter_280000.caffemodel \
  --gpu=$1


#  --weights=models/celeb_siamese_v0/celeb_v3_iter_10000.caffemodel \
