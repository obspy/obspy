#!/bin/bash

# go from here to ObsPy root directory
cd ../..

PACKAGES=$(ls | grep obspy)

echo $PACKAGES

# link all packages to python2.x/lib/site-packages/
for NAME in $PACKAGES; do
    cd $NAME
    rm -rf build
    pip install -v -e ./
    cd ..
done

# go back to scripts directory
cd misc/scripts
