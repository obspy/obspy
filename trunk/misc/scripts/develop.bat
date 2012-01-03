@echo off
SETLOCAL

REM
REM Run vcvars32.bat in VS2008 command line for 32 bit!
REM Run vcvars64.bat in VS2008 command line for 64 bit!
REM

REM
REM Set your correct python interpreter here or just use virtualenv.
set PYTHON=python
REM

cd ..\..

FOR %%M IN (core mseed arclink db earthworm gse2 imaging iris neries sac seg2 segy seisan seishub sh signal taup wav xseed) DO (
cd obspy.%%M
echo === obspy.%%M ===
%PYTHON% setup.py -q clean --all >NUL 2>NUL
IF [%1]==[] (
  %PYTHON% setup.py -q build develop
) ELSE (
  %PYTHON% setup.py -q build -c mingw32 develop
)
%PYTHON% setup.py -q clean --all >NUL 2>NUL
echo OK
cd ..
)

cd misc\scripts
