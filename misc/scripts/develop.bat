@echo off
SETLOCAL

IF [%1]==[] (
    set MODE=MS Visual C++ 2008 and MinGW
) else (
    set MODE=MinGW only
)
echo Build Mode: %MODE%

if defined PYTHON (
    echo PYTHON=%PYTHON%
) else (
    set PYTHON=python.exe
    echo PYTHON=python.exe
)

if defined BUILD_FLAGS (
    echo BUILD_FLAGS=%BUILD_FLAGS%
) else (
    set BUILD_FLAGS=-q
    echo BUILD_FLAGS=-q
)

REM add build mode
IF [%1]==[] (
    set BUILD_FLAGS=%BUILD_FLAGS% build
) else (
    set BUILD_FLAGS=%BUILD_FLAGS% build -c mingw32
)

REM
REM Run vcvars32.bat in VS2008 command line for 32 bit!
REM Run vcvars64.bat in VS2008 command line for 64 bit!
REM

if defined VCINSTALLDIR (
    echo Microsoft Visual C++ 2008 found
    echo * VSINSTALLDIR="%VSINSTALLDIR%"
    echo * VCINSTALLDIR="%VCINSTALLDIR%"
) else (
    echo MS Visual C++ 2008 not found
)

REM
REM Add mingw to path in order to find those executables
REM

echo Check for MinGW executables
<nul (set/p z="* gcc: ")
WHERE /F gcc.exe

echo.

cd ..\..


%PYTHON% setup.py %BUILD_FLAGS% clean --all >NUL 2>NUL
%PYTHON% setup.py %BUILD_FLAGS% develop
%PYTHON% setup.py %BUILD_FLAGS% clean --all >NUL 2>NUL

cd misc\scripts
