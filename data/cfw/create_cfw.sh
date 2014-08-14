#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLES=../../examples/cfw
DATA=../../data/cfw/
TOOLS=../../build/tools

echo "Creating leveldb..."

head $DATA/cfw_all.txt

rm -rf cfw_leveldb && mkdir cfw_leveldb

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/ \
    $DATA/cfw_all.txt \
    cfw_leveldb 0

echo "Computing image mean..."

$TOOLS/compute_image_mean.bin ./cfw_leveldb mean_50.binaryproto

echo "Done."
