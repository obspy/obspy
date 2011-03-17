#!/bin/bash
#-------------------------------------------------------------------
# Filename: build_deb.sh
#  Purpose: Build Debian packages for ObsPy 
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 Moritz Beyreuther
#---------------------------------------------------------------------

# Must be executed in the scripts directory

#
# Setting PATH to correct python distribution, avoid to use virtualenv
#
export PATH=/usr/bin:/usr/sbin:/bin:/sbin


DEBDIR=`pwd`/packages
DIR=`pwd`/../..
rm -rf $DEBDIR
mkdir -p $DEBDIR


MODULES="\
obspy.events \
obspy.segy \
obspy.neries \
obspy.iris \
obspy.core \
obspy.mseed \
obspy.arclink \
obspy.db \
obspy.fissures \
obspy.gse2 \
obspy.imaging \
obspy.sac \
obspy.seisan \
obspy.seishub \
obspy.sh \
obspy.signal \
obspy.wav \
obspy.xseed"

# if first argument not empty
if [ -n "$1" ]; then
    MODULES=$1
    NOT_EQUIVS="True"
fi

#
# Build all ObsPy Packages
#
for MODULE in $MODULES; do
    cd $DIR/$MODULE
    # remove dependencies of distribute for obspy.core
    # distribute is not packed for python2.5 in Debain
    # Note: the space before distribute is essential
    if [ "$MODULE" = "obspy.core" ]; then
       ex setup.py <<< "g/ distribute_setup/d|:wq"
    fi
    # remove untracked files
    svn cleanup .
    # increase version number, the debian version
    # has to be increased manually. Uncomment only
    # on final build process
    VERSION=`cat ${MODULE/./\/}/VERSION.txt` 
    DEBVERSION=1
    #dch --newversion ${VERSION}-$DEBVERSION "New release" 
    # update also Standards-Version: 0.3.3
    ex debian/control <<< "g/Standards-Version/s/[0-9.]\+/$VERSION/|:wq"
    # build the package
    fakeroot ./debian/rules clean build binary
    mv ../python-${MODULE/./-}_*.deb $DEBDIR
    # revert changes made
    if [ "$MODULE" = "obspy.core" ]; then
       svn revert setup.py
    fi
done

#
# Build namespace package if NOT_EQUIVS is non zero
#
if [ -z "$NOT_EQUIVS" ]; then
    cd $DIR/misc
    equivs-build ./debian/control
    mv python-obspy_*.deb $DEBDIR
fi

#
# run lintian to verify the packages
#
for MODULE in $MODULES; do
    PACKAGE=`ls $DEBDIR/python-obspy-${MODULE#obspy.}_*.deb`
    echo $PACKAGE
    lintian -i $PACKAGE
done
