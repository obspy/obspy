#!/bin/bash

cd ../../..
cd obspy.core/trunk
python setup.py develop -N -U

cd ../..
cd obspy.imaging/trunk
python setup.py develop -N -U

cd ../..
cd obspy.gse2/trunk
python setup.py develop -N -U

cd ../..
cd obspy.mseed/trunk
python setup.py develop -N -U

cd ../..
cd obspy.sac/trunk
python setup.py develop -N -U

cd ../..
cd obspy.wav/trunk
python setup.py develop -N -U

cd ../..
cd obspy.seisan/trunk
python setup.py develop -N -U

cd ../..
cd obspy.arclink/trunk
python setup.py develop -N -U

cd ../..
cd obspy.seishub/trunk
python setup.py develop -N -U

cd ../..
cd obspy.fissures/trunk
python setup.py develop -N -U

cd ../..
cd obspy.xseed/trunk
python setup.py develop -N -U

cd ../..
cd obspy.signal/trunk
python setup.py develop -N -U

cd ../..
cd obspy/trunk
python setup.py develop -N -U
