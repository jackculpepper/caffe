#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/celeb_siamese_v0/ $TOOLS/caffe train \
  --solver=models/celeb_siamese_v0/solver.prototxt \
  --weights=models/celeb_siamese_v0/celeb_v17_iter_500000.caffemodel \
  --gpu=$1

#  --weights=models/celeb_siamese_v0/celeb_v18_iter_20000.caffemodel \
#  --weights=models/celeb_siamese_v0/celeb_v16_iter_1000.caffemodel \

#  --weights=models/celeb_siamese_v0/celeb_v1_iter_280000.caffemodel \
#  --weights=models/celeb_siamese_v0/celeb_v16_iter_90000.caffemodel \
#  --weights=models/celeb_siamese_v0/celeb_v14_iter_100000.caffemodel \
#  --weights=models/celeb_siamese_v0/celeb_v1_iter_280000.caffemodel \
#  --weights=models/celeb_siamese_v0/celeb_v7_iter_10000.caffemodel \
#  --weights=models/celeb_siamese_v0/celeb_v3_iter_10000.caffemodel \
