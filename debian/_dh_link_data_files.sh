#!/bin/sh

rm -rf debian/python-obspy.links debian/python3-obspy.links
for DIRTYPE in data images
do
    for FILE in `find debian/tmp -type f -regex ".*/python2.*/*-packages/obspy/.*/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/obspy/#'`
        DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        echo $SOURCE $DESTINATION >> debian/python-obspy.links
    done
    for FILE in `find debian/tmp -type f -regex ".*/python3.*/*-packages/obspy/.*/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/obspy/#'`
        DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        echo $SOURCE $DESTINATION >> debian/python3-obspy.links
    done
done
