@echo off
SETLOCAL

set PYTHON="python"

cd ..

FOR %%M IN (core arclink fissures gse2 imaging mseed sac seisan seishub signal wav xseed sh) DO (
cd ..\..
cd obspy.%%M/trunk
echo === obspy.%%M ===
%PYTHON% setup.py -q clean --all >NUL
%PYTHON% setup.py -q build develop
%PYTHON% setup.py -q clean --all >NUL
echo OK
)

cd ..\..
cd obspy\branches\scripts
