#!/bin/bash
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

for DIST in squeeze wheezy lucid natty oneiric precise quantal; do
    for ARCH in i386 amd64; do
        DISTARCH=${DIST}_${ARCH}
        echo "#### $DISTARCH" >> $LOG
        git clean -fxd 2>&1 >> $LOG
        cd /tmp  # can make problems to enter schroot environment from a folder not present in the schroot
        COMMAND="cd $DEBSCRIPTDIR; ./deb__build_debs.sh &>> $LOG"
        if [[ "$DIST" == "quantal" ]]
        then
            COMMAND="export GIT_SSL_NO_VERIFY=true; $COMMAND"
        fi
        SCHROOT_SESSION=$(schroot --begin-session -c $DISTARCH)
        echo "$COMMAND" | schroot --run-session -c "$SCHROOT_SESSION" 2>&1 >> $LOG
        schroot -f --end-session -c "$SCHROOT_SESSION" 2>&1 >> $LOG
        mv $PACKAGEDIR/* $BASEDIR 2>&1 >> $LOG
    done
done
ln $BASEDIR/*.deb $PACKAGEDIR/ 2>&1 >> $LOG
