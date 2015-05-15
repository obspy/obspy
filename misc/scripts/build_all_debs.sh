#!/bin/bash
# Automated script to build all packages for all distributions.
# schroot environments have to be set up accordingly beforehand.

GITFORK=obspy
GITTARGET=master
# Process command line arguments
while getopts f:t: opt
do
   case "$opt" in
      f) GITFORK=$OPTARG;;
      t) GITTARGET=$OPTARG;;
   esac
done


BASEDIR=/tmp/python-obspy_buildall
GITDIR=$BASEDIR/git
DEBSCRIPTDIR=$GITDIR/misc/debian
LOG=$BASEDIR/build_all_debs.log

BUILDDIR=/tmp/python-obspy_build
PACKAGEDIR=$BUILDDIR/packages

rm -rf $BASEDIR
mkdir -p $BASEDIR
exec 2>&1 >> $LOG
echo '#############'
echo "#### `date`"

git clone git://github.com/${GITFORK}/obspy.git $GITDIR
cd $GITDIR
if [ "$GITFORK" != "obspy" ]
then
    git remote add upstream git://github.com/obspy/obspy.git
    git fetch upstream
fi
git clean -fxd
git fetch --all
git checkout -- .
if [ "$GITTARGET" != "master" ]
then
    git checkout -b $GITTARGET origin/$GITTARGET
fi
git clean -fxd

for DIST in squeeze wheezy jessie precise trusty utopic; do
    for ARCH in i386 amd64; do
        DISTARCH=${DIST}_${ARCH}
        echo "#### $DISTARCH"
        cd $GITDIR
        git clean -fxd
        cd /tmp  # can make problems to enter schroot environment from a folder not present in the schroot
        COMMAND="cd $DEBSCRIPTDIR; ./deb__build_debs.sh -f $GITFORK -t $GITTARGET &>> $LOG"
        SCHROOT_SESSION=$(schroot --begin-session -c $DISTARCH)
        echo "$COMMAND" | schroot --run-session -c "$SCHROOT_SESSION"
        schroot -f --end-session -c "$SCHROOT_SESSION"
        mv $PACKAGEDIR/* $BASEDIR
    done
done
ln $BASEDIR/*.deb $PACKAGEDIR/
