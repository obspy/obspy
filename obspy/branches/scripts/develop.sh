#!/bin/bash

cd ../../..
cd obspy.core/trunk
python setup.py develop

cd ../..
cd obspy.imaging/trunk
python setup.py develop

cd ../..
cd obspy.gse2/trunk
python setup.py develop

cd ../..
cd obspy.mseed/trunk
python setup.py develop

cd ../..
cd obspy.sac/trunk
python setup.py develop

cd ../..
cd obspy.wav/trunk
python setup.py develop

cd ../..
cd obspy.seisan/trunk
python setup.py develop

cd ../..
cd obspy.arclink/trunk
python setup.py develop

cd ../..
cd obspy.seishub/trunk
python setup.py develop

cd ../..
cd obspy.fissures/trunk
python setup.py develop

cd ../..
cd obspy.xseed/trunk
python setup.py develop

cd ../..
cd obspy.signal/trunk
python setup.py develop
