#!/bin/sh

for FILE in `ls debian/tmp/usr/lib/python2.7/*-packages/obspy/lib/lib*`
do
    # copy to original source directory, otherwise Help2Man exits with error
    # because command line script can't be executed (it is tried to be executed
    # in source directory it seems)
    cp $FILE obspy/lib/
done
for FILE in `ls debian/tmp/usr/lib/python3*/*-packages/obspy/lib/lib*`
do
    # copy to original source directory, otherwise Help2Man exits with error
    # because command line script can't be executed (it is tried to be executed
    # in source directory it seems)
    cp $FILE obspy/lib/
done
