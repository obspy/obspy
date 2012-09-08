#!/bin/bash

# go from here to ObsPy root directory
cd ../..

PACKAGES=$(ls | grep obspy | grep -v core)

echo $PACKAGES

# link all packages to python2.x/lib/site-packages/
for NAME in obspy.core $PACKAGES; do
    cd $NAME
    rm -rf build
    pip install --no-deps -v -e .
    cd ..
done

# go back to scripts directory
cd misc/scripts
