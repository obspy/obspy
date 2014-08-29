#!/bin/bash

CODENAME=`lsb_release -cs`

rm -rf debian/python-obspy.links debian/python3-obspy.links
for DIRTYPE in data images
do
    for FILE in `find debian/tmp -type f -regex ".*/python2.*/obspy/.*?/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/obspy/#'`
        if [ "$CODENAME" == "squeeze" ]
        then
            DESTINATION=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/pyshared/obspy/##'`
        else
            DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        fi
        echo $SOURCE $DESTINATION >> debian/python-obspy.links
    done
done
