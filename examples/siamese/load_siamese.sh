#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin siamese_solver.prototxt ../cfw/4layers_v3_init/cfw_iter_1.solverstate 1

echo "Done."
