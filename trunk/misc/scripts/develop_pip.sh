#!/bin/bash

PACKAGES="core gse2 mseed sac seisan sh wav xseed signal imaging arclink \
seishub fissures db segy iris neries db taup"

# go from here to ObsPy root directory
cd ../..

# link all packages to python2.x/lib/site-packages/
for NAME in $PACKAGES; do
    cd obspy.$NAME
    rm -rf build
    pip install -v -e ./
    cd ..
done

# go back to scripts directory
cd misc/scripts
