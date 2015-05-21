#!/usr/bin/env sh

OUTFILE=train_val.prototxt
echo >$OUTFILE

cat header.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE


OUTFILE=test.prototxt
echo >$OUTFILE

cat header.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE


OUTFILE=train_val_siamese.prototxt
echo >$OUTFILE

cat header.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat data.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat same_label.prototxt >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat contrast.prototxt >>$OUTFILE

OUTFILE=lfw_test_siamese.prototxt
echo >$OUTFILE

cat header_lfw.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat data.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat same_label.prototxt >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE
cat contrast.prototxt >>$OUTFILE

