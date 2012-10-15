#!/bin/sh
# Automated script to build all packages for all distributions.
# schroot environments have to be set up accordingly beforehand.

BASEDIR=/tmp/python-obspy_buildall
GITDIR=$BASEDIR/git
DEBSCRIPTDIR=$GITDIR/misc/debian
LOG=$BASEDIR/build_all_debs.log

BUILDDIR=/tmp/python-obspy_build
PACKAGEDIR=$BUILDDIR/packages

rm -rf $BASEDIR
mkdir -p $BASEDIR
echo '#############' >> $LOG
echo "#### `date`" >> $LOG

git clone https://github.com/obspy/obspy.git $GITDIR 2>&1 >> $LOG
cd $GITDIR 2>&1 >> $LOG

for DIST in squeeze lucid natty oneiric precise; do
    for ARCH in i386 amd64; do
        DISTARCH=${DIST}_${ARCH}
        echo "#### $DISTARCH" >> $LOG
        git clean -fxd 2>&1 >> $LOG
        echo "cd $DEBSCRIPTDIR; ./deb__build_debs.sh &>> $LOG" | schroot -c $DISTARCH 2>&1 >> $LOG
        mv $PACKAGEDIR/* $BASEDIR 2>&1 >> $LOG
    done
done
