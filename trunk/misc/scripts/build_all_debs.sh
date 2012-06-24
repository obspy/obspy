#!/bin/sh
# Automated script to build all packages for all distributions.
# schroot environments have to be set up accordingly beforehand.

DIR=/tmp/debian
TAGSDIR=$DIR/tags
LOG=$DIR/build_all_debs.log

echo '#############' >> $LOG
echo `date` >> $LOG

rm -rf $DIR
mkdir $HOME/build_all_debs

svn checkout --non-interactive --trust-server-cert https://svn.obspy.org/trunk/misc/debian $DIR

for DIST in lenny squeeze lucid maverick natty oneiric precise; do
    for ARCH in i386 amd64; do
        DISTARCH=${DIST}_${ARCH}
        echo "$DISTARCH" >> $LOG
        rm -rf $TAGSDIR
        svn checkout --non-interactive --trust-server-cert https://svn.obspy.org/tags $TAGSDIR
        svn revert --non-interactive --trust-server-cert $DIR/control
        echo "cd $DIR; ./deb__build_debs.sh &>> $LOG" | schroot -c $DISTARCH
        mv $DIR/packages/* $HOME/build_all_debs/
    done
done

cp $LOG $HOME/
