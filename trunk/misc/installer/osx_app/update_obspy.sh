#!/usr/bin/env sh

PREFIX=/Applications/ObsPy.app/Contents/MacOS
BINS=$PREFIX/bin

# Uninstall old ipython version and install new version
$BINS/pip uninstall -y ipython
$BINS/pip install ipython==0.12

# Uninstall old ObsPy version
$BINS/pip uninstall -y obspy.core
$BINS/pip uninstall -y obspy.arclink
$BINS/pip uninstall -y obspy.db
$BINS/pip uninstall -y obspy.earthworm
$BINS/pip uninstall -y obspy.gse2
$BINS/pip uninstall -y obspy.imaging
$BINS/pip uninstall -y obspy.iris
$BINS/pip uninstall -y obspy.mseed
$BINS/pip uninstall -y obspy.neries
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
$BINS/pip install obspy.core==0.6.2
$BINS/pip install obspy.arclink==0.5.1
$BINS/pip install obspy.db==0.5.1
$BINS/pip install obspy.earthworm==0.1.0
$BINS/pip install obspy.gse2==0.5.1
$BINS/pip install obspy.imaging==0.5.1
$BINS/pip install obspy.iris==0.5.1
$BINS/pip install obspy.mseed==0.6.1
$BINS/pip install obspy.neries==0.5.1
$BINS/pip install obspy.sac==0.5.1
$BINS/pip install obspy.seg2==0.1.1
$BINS/pip install obspy.segy==0.5.2
$BINS/pip install obspy.seisan==0.5.1
$BINS/pip install obspy.seishub==0.5.1
$BINS/pip install obspy.sh==0.5.2
$BINS/pip install obspy.signal==0.6.1
$BINS/pip install obspy.taup==0.5.1
$BINS/pip install obspy.wav==0.5.1
$BINS/pip install obspy.xseed==0.5.2
