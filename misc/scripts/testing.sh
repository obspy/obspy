#!/bin/bash

PACKAGES="core arclink fissures gse2 imaging mseed sac seisan seishub \
signal wav xseed segy"

for NAME in $PACKAGES; do
    cd ../..
    cd obspy.$NAME/trunk
    echo === obspy.$NAME ===
    python setup.py -q clean --all >/dev/null
    python setup.py -q build
    python setup.py -v test
    echo Hit enter to continue
    read # that's the pause
    python setup.py -q clean --all >/dev/null
done

cd ../..
cd misc/scripts
