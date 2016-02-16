#!/bin/sh

for TESTSDIR in `find . -type d -wholename './debian/tmp/usr/lib/python2*/*-packages/obspy/*/tests'`
do
    for DIRTYPE in data images
    do
        SUFFIX=`echo $TESTSDIR | sed 's#.*-packages/##'`
        TARGET=usr/share/${SUFFIX}/${DIRTYPE}
        # check if there's files that need copying, otherwise the build complains about the empty folder..
        if [ `ls ${TESTSDIR}/${DIRTYPE}/* &> /dev/null; echo $?` -eq 0 ]
        then
            dh_installdirs -p python-obspy-dbg ${TARGET}
            dh_install -p python-obspy-dbg ${TESTSDIR}/${DIRTYPE}/* ${TARGET}
        fi
    done
done
