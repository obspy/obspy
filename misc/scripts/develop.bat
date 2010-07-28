@echo off
SETLOCAL

REM set PYTHON="C:\Program Files (x86)\Python2.6\python"
set PYTHON="C:\Program Files\Python26\python"

cd ..

FOR %%M IN (core arclink fissures gse2 imaging mseed sac seisan seishub signal wav xseed sh) DO (
cd ..\..
cd obspy.%%M/trunk
echo === obspy.%%M ===
%PYTHON% setup.py -q clean --all >NUL
%PYTHON% setup.py -q build -c mingw32 develop
%PYTHON% setup.py -q clean --all >NUL
echo OK
)

cd ..\..
cd obspy\branches\scripts
