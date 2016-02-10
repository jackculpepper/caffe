#!/usr/bin/env sh

GLOG_logtostderr=0 GLOG_log_dir=examples/drsae/ ./build/tools/caffe train \
    --gpu=0,1 \
    --weights=examples/drsae/drsae_v00_iter_140000.caffemodel \
    --solver=examples/drsae/solver.prototxt

