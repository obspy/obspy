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
$BINS/pip install ipython==0.12
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
