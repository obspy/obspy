#!/bin/bash

#DIRS=`ls -d debian/python-obspy/usr/lib/python2*/*-packages/obspy/*/tests/data` `ls -d debian/python-obspy/usr/lib/python2*/*-packages/obspy/*/tests/images`
#for DIR in $DIRS
#do
#    SOURCEDIR=`echo $DIR | sed 's#debian/python-obspy/##'`
#    TARGETDIR=`echo $SOURCEDIR | sed 's#.*?-packages/obspy/#usr/share/obspy/#'`
#    for FILE in `ls $DIR`
#    do
#        FILENAME=`echo $FILE | sed 's#.*/##'`
#        echo "${TARGETDIR}/$FILENAME ${SOURCEDIR}/$FILENAME" >> python-obspy.links
#    done
#done
#

#for DIR in `ls debian/python-obspy/usr/lib/python2*/*-packages/obspy/*/tests/data debian/python-obspy/usr/lib/python2*/*-packages/obspy/*/tests/images`
#do
#    SOURCEFOLDER=`echo $DIR | sed 's#-*?-packages/obspy/#usr/share/obspy/#'`
#    LINKFOLDER=`echo $DIR | sed 's#debian/python-obspy/##'`
#    for FILE in `ls ${DIR}/*`
#    do
#        LINK=`echo $FILE | sed 's#debian/python-obspy-data/usr/share/obspy/##'`
#        FILE=`echo $FILE | sed 's#.*?-packages/obspy/##'`
#        echo "usr/share/obspy/$FILE `echo $DIR | sed 's#debian/python-obspy/##'`${FILE}" >> python-obspy.links
#    done
#done

rm -rf debian/python-obspy.links debian/python3-obspy.links
for DIRTYPE in data images
do
    for FILE in `find debian/tmp -type f -regex ".*/python2.*/obspy/.*?/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/obspy/#'`
        DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        echo $SOURCE $DESTINATION >> debian/python-obspy.links
    done
    for FILE in `find debian/tmp -type f -regex ".*/python3.*/obspy/.*?/tests/${DIRTYPE}/.*"`
    do
        SOURCE=`echo $FILE | sed 's#.*-packages/obspy/#usr/share/obspy/#'`
        DESTINATION=`echo $FILE | sed 's#debian/tmp/##'`
        echo $SOURCE $DESTINATION >> debian/python3-obspy.links
    done
done

#dh_link -p python-obspy $LINKFILES
#
#for DIR in DIRS=`ls -d debian/python3-obspy/usr/lib/python3*/*-packages/obspy/*/tests/*`
#do
#    SOURCEDIR=`echo $DIR | sed 's#debian/python3-obspy/##'`
#    TARGETDIR=`echo $SOURCEDIR | sed 's#.*?-packages/obspy/#usr/share/obspy/#'`
#    dh_link -p python-obspy ${TARGETDIR}/* ${SOURCEDIR}
#done

#rm -rf python3-obspy.links
#for FILE in `ls debian/python-obspy/usr/lib/python3*/*-packages/obspy/*/tests/data/* debian/python-obspy/usr/lib/python3*/*-packages/obspy/*/tests/images/*`
#do
#    LINK=`echo $FILE | sed 's#debian/python3-obspy/##'`
#    FILE=`echo $FILE | sed 's#.*?-packages/obspy/##'`
#    echo "usr/share/obspy/$FILE $LINK" >> python3-obspy.links
#done
