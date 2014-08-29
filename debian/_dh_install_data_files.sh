#!/bin/bash

for DIRTYPE in data images
do
    for DIR in `ls -d debian/tmp/usr/lib/python2*/*-packages/obspy/*/tests/${DIRTYPE}`
    do
        MOD=`echo $DIR | sed 's#.*obspy/##' | sed 's#/.*##'`
        if [ "$CODENAME" = "squeeze" ]
        then
            TARGET=usr/share/pyshared/obspy/${MOD}/tests/${DIRTYPE}
        else
            TARGET=usr/share/obspy/${MOD}/tests/${DIRTYPE}
        fi
        dh_installdirs -p python-obspy-dbg ${TARGET}
        dh_install -p python-obspy-dbg ${DIR}/* ${TARGET}
    done
done
