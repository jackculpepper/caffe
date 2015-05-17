#!/usr/bin/env sh

OUTFILE=train_val.prototxt
echo >$OUTFILE

cat header.prototxt >>$OUTFILE
cat data_memory.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE


OUTFILE=test.prototxt
echo >$OUTFILE

cat header.prototxt >>$OUTFILE
cat data_memory.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat shared.prototxt | sed 's/_XXX//g' >>$OUTFILE
cat loss_kway.prototxt | sed 's/_XXX//g' >>$OUTFILE

