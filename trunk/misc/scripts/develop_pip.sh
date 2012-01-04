#!/bin/bash

# go from here to ObsPy root directory
cd ../..
TRUNKDIR=`pwd`

# get list of all package names from core.util.base.py
cd $TRUNKDIR/obspy.core/obspy/core/util/
PACKAGES=`python -c "from base import ALL_MODULES; print ' '.join(ALL_MODULES)"`
cd $TRUNKDIR

# link all packages to python2.x/lib/site-packages/
for NAME in $PACKAGES; do
    cd obspy.$NAME
    rm -rf build
    pip install -v -e ./
    cd ..
done

# go back to scripts directory
cd misc/scripts
