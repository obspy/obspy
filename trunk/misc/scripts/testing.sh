#!/bin/bash

# go from here to ObsPy root directory
cd ../..
TRUNKDIR=`pwd`

# get list of all package names from core.util.base.py
cd $TRUNKDIR/obspy.core/obspy/core/util/
PACKAGES=`python -c "from base import ALL_MODULES; print ' '.join(ALL_MODULES)"`
cd $TRUNKDIR

for NAME in $PACKAGES; do
    cd obspy.$NAME
    echo === obspy.$NAME ===
    python setup.py -q clean --all >/dev/null
    python setup.py -q build
    python setup.py -v test
    echo Hit enter to continue
    read # that's the pause
    python setup.py -q clean --all >/dev/null
    cd ..
done

cd misc/scripts
