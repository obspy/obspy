#!/bin/bash

# go from here to ObsPy root directory
cd ../..

# link to python2.x/lib/site-packages/
rm -rf build
pip install --no-deps -v -e .

# go back to scripts directory
cd misc/scripts
