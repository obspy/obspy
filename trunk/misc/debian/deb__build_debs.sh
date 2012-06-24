#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__build_debs.sh
#  Purpose: Build Debian packages for ObsPy 
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 ObsPy Development Team
#---------------------------------------------------------------------

# Must be executed in the misc/debian directory
# Tags are supposed to be checked out as "tags" subdirectory

METAPACKAGE_VERSION=0.7.0
METAPACKAGE_DEBVERSION=1
DEBVERSION=1
DATE=`date +"%a, %d %b %Y %H:%M:%S %z"`

# Setting PATH to correct python distribution, avoid to use virtualenv
export PATH=/usr/bin:/usr/sbin:/bin:/sbin

CODENAME=`lsb_release -cs`

BASEDIR=`pwd`
PACKAGEDIR=$BASEDIR/packages
TAGSDIR=$BASEDIR/tags

# deactivate, else each time all packages are removed
#rm -rf $PACKAGEDIR $TAGSDIR
mkdir -p $PACKAGEDIR

# download tags
#svn checkout --quiet https://svn.obspy.org/tags $TAGSDIR
#if [ ! $? -eq 0 ]; then
#    echo "Error during svn checkout, aborting"
#    exit 1
#fi

MODULES=`ls $TAGSDIR`

## if first argument not empty
#if [ -n "$1" ]; then
#    MODULES=$1
#    NOT_EQUIVS="True"
#fi

# Build all ObsPy Packages
for MODULE in $MODULES; do
    echo "#### Working on $MODULE"
    MODULEDIR=$TAGSDIR/$MODULE
    TAGS=`ls -1 $MODULEDIR | tail -1`
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
        # get version number from the tag, the debian version
        # has to be increased manually if necessary.
        VERSION=$TAG
        # the commented code shows how to update the changelog
        # information, however we do not do it as it hard to
        # automatize it for all packages in common
        # dch --newversion ${VERSION}-$DEBVERSION "New release" 
        # just write a changelog template with only updated version info
    cat > debian/changelog << EOF
python-${MODULE/./-} (${VERSION}-${DEBVERSION}~${CODENAME}) unstable; urgency=low

  * visit http://www.obspy.org for more information about the age 
    and the contents of this release
EOF
    if [ -f obspy/*/CHANGELOG.txt ]
        then
        echo "" >> debian/changelog
        sed "s/^/  /" obspy/*/CHANGELOG.txt >> debian/changelog
    fi
    cat >> debian/changelog << EOF

 -- ObsPy Development Team <devs@obspy.org>  $DATE
EOF
        # dh doesn't know option python2 in lucid
        if [ $CODENAME = "lucid" ]
            then
            ex ./debian/rules << EOL
%s/--with=python2/ /g
g/dh_numpy/d
wq
EOL
        fi
        # build the package
        fakeroot ./debian/rules clean build binary
        mv ../python-${MODULE/./-}_*.deb $PACKAGEDIR/
    done
done

# Build namespace package if NOT_EQUIVS is non zero
#if [ -z "$NOT_EQUIVS" ]; then
cd $BASEDIR
svn revert control
ex control << EOF
g/^Version: /s/ .*/ ${METAPACKAGE_VERSION}-${METAPACKAGE_DEBVERSION}\~${CODENAME}/
wq
EOF
equivs-build control
mv python-obspy_*.deb $PACKAGEDIR
#fi

# run lintian to verify the packages
for PACKAGE in `ls $PACKAGEDIR/*.deb`; do
    echo $PACKAGE
    #lintian -i $PACKAGE # verbose output
    lintian $PACKAGE
done
