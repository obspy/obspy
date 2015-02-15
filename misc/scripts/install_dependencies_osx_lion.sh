#!/bin/sh
echo "Script last updated on August 24th, 2011\n"

# Check OSX version
version=`sw_vers -productVersion`
if [ "$version" != 10.7 ]; then
    echo "This script has only been tested on OSX Lion (10.7). It might work on other OSX Versions, too.\nContinue? (Y/N)"
    while true; do
	read answer
        case $answer in
            [Y/y] ) break;;
            [N/n] ) echo "Cancelling..."; exit 1;;
            * ) echo "Please answer Y/y or N/n.";;
        esac
    done
fi

echo "\nObsPy and dependencies install script for OSX Lion"
echo "--------------------------------------------------"
echo "This script needs an already installed Python. It furthermore requires"
echo "the following Python modules: NumPy, setuptools/distribute and the pip"
echo "installer.\n"
echo "It is intended to be used with the Python/NumPy combination that ships"
echo "with OSX Lion.\n"


echo "It is recommended to use an ACTIVATED virtual environment"
echo "(www.virtualenv.org) to not mess with the system wide installation."
echo "Newer versions will also install setuptools/distribute and the pip"
echo "installer.\n"

echo "This script will install the following Python modules"
echo " * Python readline"
echo " * IPython"
echo " * lxml"
echo " * SciPy"
echo " * matplotlib"
echo " * The current development versions of all ObsPy modules (optional)\n"

echo "For further help see\nhttp://obspy.org/wiki/InstallingDependenciesMac\n"


echo "Make sure XCode and pkg-config are installed. Continue? (Y/N)"
    while true; do
	read answer
        case $answer in
            [Y/y] ) break;;
            [N/n] ) echo "Cancelling..."; exit 1;;
            * ) echo "Please answer Y/y or N/n.";;
        esac
    done


# Install some dependencies.
easy_install readline # for some reason the version installed with pip does not work properly ...
pip install ipython
pip install lxml


# Currently only the repository version of scipy works. Also the llvm-gcc of OSX Lion does currently
# not work with scipy, so just use the old one.
export CC=gcc-4.2
export CXX=g++-4.2
export FFLAGS=-ff2c
git clone https://github.com/scipy/scipy.git scipy_git
cd scipy_git
pip install .
cd ..
rm -rf scipy_git


# Matplotlib also needs the development version.
git clone git://github.com/matplotlib/matplotlib.git matplotlib_git
cd matplotlib_git
pip install .
cd ..
rm -rf matplotlib_git


# Install ObsPy or dont.
echo "\n\nDo you want to install the latest development version of ObsPy? (Y/N)"
while true; do
    read answer
    case $answer in
        [Y/y] ) break;;
        [N/n] ) echo "Cancelling..."; exit 1;;
        * ) echo "Please answer Y/y or N/n.";;
    esac
done
pip install obspy.core==dev
pip install obspy.imaging==dev
pip install obspy.signal==dev
pip install obspy.xseed==dev
pip install obspy.gse2==dev
pip install obspy.sac==dev
pip install obspy.mseed==dev
pip install obspy.wav==dev
pip install obspy.sh==dev
pip install obspy.seisan==dev
pip install obspy.segy==dev
pip install obspy.seishub==dev
pip install obspy.arclink==dev
pip install obspy.fissures==dev
pip install obspy.iris==dev
pip install obspy.neries==dev
pip install obspy.taup==dev
