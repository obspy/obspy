#!/bin/sh

for TESTSDIR in `find -type d -wholename 'debian/tmp/usr/lib/python2*/*-packages/obspy/*/tests'`
do
    for DIRTYPE in data images
    do
        SUFFIX=`echo $TESTSDIR | sed 's#.*-packages/##'`
        TARGET=usr/share/${SUFFIX}/${DIRTYPE}
        dh_installdirs -p python-obspy-dbg ${TARGET}
        dh_install -p python-obspy-dbg ${TESTSDIR}/${DIRTYPE}/* ${TARGET}
    done
done
