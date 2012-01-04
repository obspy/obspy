#!/bin/bash

PACKAGES="core mseed arclink db earthworm gse2 imaging iris neries sac seg2 \
segy seisan seishub sh signal taup wav xseed"

cd ../..

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
