#!/bin/sh

for FILE in `ls debian/tmp/usr/lib/python3*/*-packages/obspy/lib/lib*`
do
    mv $FILE `echo $FILE | sed 's#-[^-]*-linux-gnu.so#.so#'`
done
