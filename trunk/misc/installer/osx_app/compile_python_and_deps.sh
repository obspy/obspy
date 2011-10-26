#!/usr/bin/env sh

# Script to compile Python and a larger number of modules including the latest
# ObsPy version to /Applications/ObsPy.app/Contents/MacOS
#
# An eventually existing folder /Applications/ObsPy.app will be deleted in the
# process.
#
# Requirements to run this script:
#  * XCode including developer tools
#  * gfortran compiler
#  * git
#  * active internet connection.

echo "This will delete /Applications/ObsPy.app. Continue? [y/n]"
while true; do
read answer
case $answer in
    [Y/y] ) break;;
    [N/n] ) echo "Cancelling..."; exit 1;;
    * ) echo "Please answer Y/y or N/n.";;
esac
done

# Make the directories first
rm -rf /Applications/ObsPy.app
mkdir /Applications/ObsPy.app
mkdir /Applications/ObsPy.app/Contents
mkdir /Applications/ObsPy.app/Contents/MacOS
mkdir /Applications/ObsPy.app/Contents/MacOS/bin

rm -rf temp_build_dir
mkdir temp_build_dir
cd temp_build_dir

PREFIX=/Applications/ObsPy.app/Contents/MacOS

# Set some flags to always use the same compilers.
export CC=gcc-4.2
export CXX=g++-4.2
export FFLAGS=-ff2c

# Download and install python
curl -O http://www.python.org/ftp/python/2.7.2/Python-2.7.2.tgz
tar -xvvzf Python-2.7.2.tgz
cd Python-2.7.2
./configure --prefix=$PREFIX --enable-universalsdk --with-universal-archs=intel
make
make install
cd ..

BINS=$PREFIX/bin

# Install distribute and use it to install pip
curl -O http://python-distribute.org/distribute_setup.py
$BINS/python distribute_setup.py
$BINS/easy_install pip

# Install other stuff.
$BINS/pip install ipython
# Readline requires easy_install for some reason.
$BINS/easy_install readline
$BINS/pip install virtualenv
$BINS/pip install nose
$BINS/pip install numpy
$BINS/pip install lxml
$BINS/pip install scipy

# Install and older scipy version as that version seems to work in Snow Leopard and Lion.
git clone https://github.com/scipy/scipy.git scipy_git
cd scipy_git
git checkout v0.9.0
$BINS/pip install .
cd ..

# Install certain matplotlib tag. The latest version does not work with ObsPy.
git clone git://github.com/matplotlib/matplotlib.git matplotlib_git
cd matplotlib_git
git checkout v1.1.0-rc1
# Also fetch and build all dependencies directly to the prefix folder to resolve
# any potential dependencies.
make PREFIX=$PREFIX PYTHON=$BINS/python -f make.osx fetch deps mpl_install
cd ..

# Install latest ObsPy version.
$BINS/pip install obspy.core==dev
$BINS/pip install obspy.arclink==dev
$BINS/pip install obspy.fissures==dev
$BINS/pip install obspy.gse2==dev
$BINS/pip install obspy.imaging==dev
$BINS/pip install obspy.iris==dev
$BINS/pip install obspy.mseed==dev
$BINS/pip install obspy.neries==dev
$BINS/pip install obspy.sac==dev
$BINS/pip install obspy.segy==dev
$BINS/pip install obspy.seisan==dev
$BINS/pip install obspy.seishub==dev
$BINS/pip install obspy.sh==dev
$BINS/pip install obspy.signal==dev
$BINS/pip install obspy.taup==dev
$BINS/pip install obspy.wav==dev
$BINS/pip install obspy.xseed==dev
