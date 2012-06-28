#!/usr/bin/env sh
PREFIX=/Applications/ObsPy.app/Contents/MacOS
BINS=$PREFIX/bin

# Install latest ObsPy version.
$BINS/pip uninstall -y obspy.arclink
$BINS/pip uninstall -y obspy.core
$BINS/pip uninstall -y obspy.datamark
$BINS/pip uninstall -y obspy.db
$BINS/pip uninstall -y obspy.earthworm
$BINS/pip uninstall -y obspy.fissures
$BINS/pip uninstall -y obspy.gse2
$BINS/pip uninstall -y obspy.imaging
$BINS/pip uninstall -y obspy.iris
$BINS/pip uninstall -y obspy.mseed
$BINS/pip uninstall -y obspy.neries
$BINS/pip uninstall -y obspy.realtime
$BINS/pip uninstall -y obspy.sac
$BINS/pip uninstall -y obspy.seg2
$BINS/pip uninstall -y obspy.segy
$BINS/pip uninstall -y obspy.seisan
$BINS/pip uninstall -y obspy.seishub
$BINS/pip uninstall -y obspy.sh
$BINS/pip uninstall -y obspy.signal
$BINS/pip uninstall -y obspy.taup
$BINS/pip uninstall -y obspy.wav
$BINS/pip uninstall -y obspy.xseed

# Install latest ObsPy version.
$BINS/pip install --no-deps obspy.arclink==0.7.1
$BINS/pip install --no-deps obspy.core==0.7.1
$BINS/pip install --no-deps obspy.datamark==0.1.0
$BINS/pip install --no-deps obspy.db==0.7.0
$BINS/pip install --no-deps obspy.earthworm==0.1.0
$BINS/pip install --no-deps obspy.gse2==0.7.0
$BINS/pip install --no-deps obspy.imaging==0.7.0
$BINS/pip install --no-deps obspy.iris==0.7.0
$BINS/pip install --no-deps obspy.mseed==0.7.0
$BINS/pip install --no-deps obspy.neries==0.7.0
$BINS/pip install --no-deps obspy.realtime==0.1.0
$BINS/pip install --no-deps obspy.sac==0.7.0
$BINS/pip install --no-deps obspy.seg2==0.7.0
$BINS/pip install --no-deps obspy.segy==0.5.2
$BINS/pip install --no-deps obspy.seisan==0.5.1
$BINS/pip install --no-deps obspy.seishub==0.5.1
$BINS/pip install --no-deps obspy.sh==0.5.2
$BINS/pip install --no-deps obspy.signal==0.7.0
$BINS/pip install --no-deps obspy.taup==0.7.0
$BINS/pip install --no-deps obspy.wav==0.5.1
$BINS/pip install --no-deps obspy.xseed==0.7.0
