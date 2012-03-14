#!/bin/bash

PACKAGES="core mseed arclink db earthworm gse2 imaging iris neries realtime sac seedlink seg2 \
segy seisan seishub sh signal taup wav xseed"

# go from here to ObsPy root directory
cd ../..

# link all packages to python2.x/lib/site-packages/
for NAME in $PACKAGES; do
    cd obspy.$NAME
    rm -rf build
    python setup.py develop -N -U --verbose
    cd ..
done

# go back to scripts directory
cd misc/scripts
