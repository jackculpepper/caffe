#!/usr/bin/env sh

OUTFILE=train_val.prototxt
echo >$OUTFILE

cat header_train_val.prototxt >>$OUTFILE
cat data.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE


