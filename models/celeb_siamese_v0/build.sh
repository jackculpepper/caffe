#!/usr/bin/env sh

OUTFILE=train_val.prototxt
echo >$OUTFILE

cat header_train_val.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE


OUTFILE=test.prototxt
echo >$OUTFILE

cat header_test.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE


OUTFILE=train_val_siamese.prototxt
echo >$OUTFILE

cat header_train_val_siamese.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat data.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat same_label.prototxt >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE
#cat loss_kway.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

#cat contrast.prototxt >>$OUTFILE
cat contrast_contrastiveloss.prototxt >>$OUTFILE

OUTFILE=test_siamese.prototxt
echo >$OUTFILE

cat header_test_siamese.prototxt >>$OUTFILE
cat data_test.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat data_test.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat same_label.prototxt >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE
#cat contrast.prototxt >>$OUTFILE
cat contrast_contrastiveloss.prototxt >>$OUTFILE

OUTFILE=test_siamese_memorydata.prototxt
echo >$OUTFILE

cat header_test_siamese_memorydata.prototxt >>$OUTFILE
cat data_memory.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat data_memory.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat same_label.prototxt >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE
#cat contrast.prototxt >>$OUTFILE
cat contrast_contrastiveloss.prototxt >>$OUTFILE


OUTFILE=lfw_test_siamese.prototxt
echo >$OUTFILE

cat header_lfw_test_siamese.prototxt >>$OUTFILE
cat data_lfw.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat data_lfw.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE

cat same_label.prototxt >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared_contrast.prototxt | sed 's/_XXX/_b/g' >>$OUTFILE
#cat contrast.prototxt >>$OUTFILE
cat contrast_contrastiveloss.prototxt >>$OUTFILE


OUTFILE=test_feature.prototxt
echo >$OUTFILE

cat header_test_feature.prototxt >>$OUTFILE
cat data_memory.prototxt | sed 's/_XXX//g' >>$OUTFILE

cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE

cat shared_contrast.prototxt | sed 's/_XXX//g' >>$OUTFILE


