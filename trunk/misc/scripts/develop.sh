#!/bin/bash

# go from here to ObsPy root directory
cd ../..
TRUNK=`pwd`

PACKAGES=$(find . -maxdepth 1 -name '*obspy.*' -type d | grep -v core)

echo $PACKAGES

# link all packages to python2.x/lib/site-packages/
for NAME in ./obspy.core $PACKAGES; do
    cd $TRUNK/$NAME
    rm -rf build
    python setup.py develop -N -U --verbose
done

# go back to scripts directory
cd $TRUNK/misc/scripts
