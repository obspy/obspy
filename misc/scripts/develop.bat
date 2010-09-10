@echo off
SETLOCAL

REM
REM Run vcvars32.bat in VS2008 command line for 32 bit!
REM Run vcvars64.bat in VS2008 command line for 64 bit!
REM

set PYTHON="C:\Program Files (x86)\Python2.6\python"
REM set PYTHON="C:\Program Files\Python26\python"

FOR %%M IN (core mseed gse2 signal imaging arclink fissures sac seisan seishub wav xseed sh) DO (
cd ..\..
cd obspy.%%M/trunk
echo === obspy.%%M ===
%PYTHON% setup.py -q clean --all >NUL
%PYTHON% setup.py -q build develop
%PYTHON% setup.py -q clean --all >NUL
echo OK
)

cd ..\..
cd misc\scripts
