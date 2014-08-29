#!/bin/bash

rm -rf debian/python-obspy.links debian/python3-obspy.links
for DIRTYPE in data images
do
    for FILE in `find debian/tmp -type f -regex ".*/python2.*/obspy/.*?/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/obspy/#'`
        DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        echo $SOURCE $DESTINATION >> debian/python-obspy.links
    done
    # for squeeze:
    for FILE in `find debian/tmp -type f -regex ".*/usr/share/pyshared/obspy/.*?/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*pyshared/obspy/#usr/share/obspy/#'`
        DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        echo $SOURCE $DESTINATION >> debian/python-obspy.links
    done
done
