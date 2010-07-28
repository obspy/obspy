@echo off
SETLOCAL

REM
REM Run vcvars32.bat in VS2008 command line for 32 bit!
REM Run vcvars64.bat in VS2008 command line for 64 bit!
REM

REM set PYTHON="C:\Program Files (x86)\Python2.6\python"
set PYTHON="C:\Program Files\Python26\python"

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
cd misc\scripts
