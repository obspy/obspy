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

CODENAME=`lsb_release -cs`

BASEDIR=`pwd`
PACKAGEDIR=$BASEDIR/packages
DIR=$BASEDIR/../..
TAGSDIR=$BASEDIR/tags

FTPHOST=obspy.org
FTPUSER=obspy

# deactivate, else each time all packages are removed
rm -rf $PACKAGEDIR $TAGSDIR
mkdir -p $PACKAGEDIR

# download tags
svn checkout --quiet https://svn.obspy.org/tags $TAGSDIR
if [ ! $? -eq 0 ]; then
    echo "Error during svn checkout, aborting"
    exit 1
fi

MODULES=`ls $TAGSDIR`

## if first argument not empty
#if [ -n "$1" ]; then
#    MODULES=$1
#    NOT_EQUIVS="True"
#fi

#
# Build all ObsPy Packages
#
for MODULE in $MODULES; do
    echo "#### Working on $MODULE"
    MODULEDIR=$TAGSDIR/$MODULE
    TAGS=`ls $MODULEDIR`
    for TAG in $TAGS; do
        echo "#### Working on $MODULE $TAG"
        TAGDIR=$MODULEDIR/$TAG
        cd $TAGDIR
        # remove dependencies of distribute for obspy.core
        # distribute is not packed for python2.5 in Debian
        # Note: the space before distribute is essential
        if [ "$MODULE" = "obspy.core" ]; then
           ex setup.py << EOL
g/ distribute_setup/d
wq
EOL
        fi
        # get version number, the debian version
        # has to be increased manually.
        VERSION=`cat ${MODULE/./\/}/VERSION.txt`
        DEBVERSION=1
        # the commented code shows how to update the changelog
        # information, however we do not do it as it hard to
        # automatize it for all packages in common
        # dch --newversion ${VERSION}-$DEBVERSION "New release" 
        # just write a changelog template with only updated version info
    cat >debian/changelog << EOF
python-${MODULE/./-} (${VERSION}-${DEBVERSION}) unstable; urgency=low

  * This changelog file is overwritten for every release, only the version
    is not kept up to date. Visit www.obspy.org for more information about
    the age and the contents of the version given above.

 -- ObsPy Development Team <devs@obspy.org>  Thu, 20 Oct 2011 10:07:58 +0200
EOF
        # update also Standards-Version: 0.3.3
        ex debian/control << EOF
g/Standards-Version/s/[0-9.]\+/$VERSION/
wq
EOF
        # build the package
        fakeroot ./debian/rules clean build binary
        mv ../python-${MODULE/./-}_*.deb $PACKAGEDIR
    done
done

#
# Build namespace package if NOT_EQUIVS is non zero
#
if [ -z "$NOT_EQUIVS" ]; then
    cd $DIR/misc
    equivs-build ./debian/control
    mv python-obspy_*.deb $PACKAGEDIR
fi

#
# run lintian to verify the packages
#
for MODULE in $MODULES; do
    PACKAGE=`ls $PACKAGEDIR/python-obspy-${MODULE#obspy.}_*.deb`
    echo $PACKAGE
    #lintian -i $PACKAGE # verbose output
    lintian $PACKAGE
done

#
# upload built packages
#
cd $PACKAGEDIR
echo -n "Give password for FTPUSER $FTPUSER and press [ENTER]: "
read FTPPASSWD
ftp -i -n -v $FTPHOST &> $BASEDIR/ftp.log << EOF
user $FTPUSER $FTPPASSWD
binary
cd debian/packages/$CODENAME
mput *.deb
bye
EOF
