#!/bin/sh

for TESTSDIR in `find . -type d -regextype posix-egrep -regex './debian/tmp/usr/lib/python2.*/.*-packages/obspy/.*/tests/(data|images)'`
do
    SUFFIX=`echo $TESTSDIR | sed 's#.*-packages/##'`
    TARGET=usr/share/${SUFFIX}
    dh_installdirs -p python-obspy-dbg ${TARGET}
    dh_install -p python-obspy-dbg ${TESTSDIR}/* ${TARGET}
done
