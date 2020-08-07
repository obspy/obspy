#!/bin/sh

for FILE in `ls debian/tmp/usr/lib/python3*/*-packages/obspy/lib/lib*`
do
    FILENEW=`echo $FILE | sed 's#-[^-]*-linux-gnu.*.so#.so#'`
    if [ "$FILENEW" != "$FILE" ]
        then
        mv $FILE $FILENEW
    fi
done
