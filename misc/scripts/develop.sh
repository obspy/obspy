#!/bin/bash

# go from here to ObsPy root directory
cd ../..

# link to python2.x/lib/site-packages/
rm -rf build
python setup.py develop -N -U --verbose

# go back to scripts directory
cd misc/scripts
