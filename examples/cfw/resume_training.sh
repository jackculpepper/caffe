#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    cfw_solver.prototxt 4layers_v5/cfw_iter_100000.solverstate

echo "Done."
