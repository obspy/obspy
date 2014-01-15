#!/usr/bin/env bash

# Script to compile Python and a larger number of modules including the latest
# ObsPy version to /Applications/ObsPy.app/Contents/MacOS
#
#
# Requirements to run this script:
#  * XCode including developer tools
#  * gfortran compiler installed with Homebrew
#  * git
#  * active internet connection.

# Stop on first error.
set -o errexit

BUILD_DIR=/tmp/obspy_app_build_dir
PREFIX=/Applications/ObsPy.app/Contents/MacOS
INCLUDE=$PREFIX/include
LIB=$PREFIX/lib
FRAMEWORK_PATH=$PREFIX/Python.framework/Versions/2.7
FRAMEWORK_BIN=$FRAMEWORK_PATH/bin
PYTHON=$FRAMEWORK_BIN/python
EASY_INSTALL=$FRAMEWORK_BIN/easy_install
PIP=$FRAMEWORK_BIN/pip
SITE_PACKAGES=$FRAMEWORK_PATH/lib/python2.7/site-packages

# Set some compiler flags. That should direct many build processes to the
# correct folders.
export PKG_CONFIG_PATH=$LIB/pkgconfig
# We have no desire for the freetype that ships with X11.
export PATH=$(echo ${PATH} | awk -v RS=: -v ORS=: '/X11/ {next} {print}')

mkdir -p $PREFIX/bin
mkdir -p $PREFIX/lib
mkdir -p $PREFIX/man/man1
mkdir -p $BUILD_DIR
mkdir -p $FRAMEWORK_BIN

LAUNCH_IPYTHON="#!/usr/bin/env bash
$FRAMEWORK_BIN/ipython notebook --pylab=inline"
echo "$LAUNCH_IPYTHON" > $FRAMEWORK_BIN/launch_ipython_notebook
chmod a+x $FRAMEWORK_BIN/launch_ipython_notebook

################################################################################
# Use the homebrew gfortran and include its libs
GFORTRAN_DIR=/usr/local/Cellar/gfortran/4.8.2/gfortran/lib
# Exclude Mach-O stub files.
GFORTRAN_FILES=$(ls $GFORTRAN_DIR/*.dylib | grep -v 10.4 | grep -v 10.5)

# Copy files.
for brew_name in $GFORTRAN_FILES
do
    cp $brew_name $LIB/$(basename $brew_name)
done

# Change the dynamic library names.
for brew_name in $GFORTRAN_FILES
do
    new_name=$LIB/$(basename $brew_name)
    install_name_tool -change $brew_name $new_name $new_name
done
################################################################################


################################################################################
# libz is required for different parts and does not appear to be compatible
# between OSX versions.
if [ ! -f $LIB/libz.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f zlib-1.2.8.tar.gz ]
    then
        wget http://zlib.net/zlib-1.2.8.tar.gz
    fi
    rm -rf zlib-1.2.8
    tar -xzf zlib-1.2.8.tar.gz
    cd zlib-1.2.8
    ./configure --prefix=$PREFIX
    make
    make install
fi
################################################################################


################################################################################
# Latest Python as a framework build.
if [ ! -f $PYTHON ]
then
    cd $BUILD_DIR
    if [ ! -f Python-2.7.6.tgz ]
    then
        wget http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tgz
    fi
    rm -rf Python-2.7.6
    tar -xzf Python-2.7.6.tgz
    cd Python-2.7.6
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX  --enable-framework=$PREFIX
    make
    make install
fi
################################################################################


################################################################################
# Installation utilities.
if [ ! -f $EASY_INSTALL ]
then
    cd $BUILD_DIR
    wget http://python-distribute.org/distribute_setup.py
    $PYTHON distribute_setup.py
fi

if [ ! -f $PIP ]
then
    cd $BUILD_DIR
    $EASY_INSTALL pip
fi
################################################################################


################################################################################
# Misc utilties
if [ ! -f $FRAMEWORK_BIN/virtualenv ]
then
    $PIP install virtualenv
fi

if [ ! -d $SITE_PACKAGES/nose ]
then
    $PIP install nose
fi

if [ ! -f $SITE_PACKAGES/mock.py ]
then
    $PIP install mock
fi
################################################################################


################################################################################
# IPython including all dependencies for the notebook.
if [ ! -f $LIB/libzmq.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f zeromq-4.0.3.tar.gz ]
    then
        wget http://download.zeromq.org/zeromq-4.0.3.tar.gz
    fi
    rm -rf zeromq-4.0.3
    tar -xzf zeromq-4.0.3.tar.gz
    cd zeromq-4.0.3
    ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -d $SITE_PACKAGES/zmq ]
then
    $PIP install pyzmq
fi

if [ ! -d $SITE_PACKAGES/tornado ]
then
    $PIP install tornado
fi

if [ ! -d $SITE_PACKAGES/jinja2 ]
then
    $PIP install jinja2
fi

if [ ! -d $SITE_PACKAGES/ipython ]
then
    $PIP install ipython
fi

################################################################################


NUMPY_AND_SCIPY_CONFIG="[DEFAULT]
search_static_first = True
library_dirs = /Applications/ObsPy.app/Contents/MacOS/lib
include_dirs = /Applications/ObsPy.app/Contents/MacOS/include

[openblas]
libraries = openblas
library_dirs = /Applications/ObsPy.app/Contents/MacOS/lib
include_dirs = /Applications/ObsPy.app/Contents/MacOS/include

[amd]
amd_libs = amd

[umfpack]
umfpack_libs = umfpack "


################################################################################
# NumPy and dependencies
# OpenBLAS
if [ ! -f $LIB/libopenblas.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f v0.2.8.tar.gz ]
    then
        wget "https://github.com/xianyi/OpenBLAS/archive/v0.2.8.tar.gz"
    fi
    rm -rf OpenBLAS-0.2.8
    tar -xzf v0.2.8.tar.gz
    cd OpenBLAS-0.2.8
    make CC=clang FC=gfortran libs netlib shared
    make PREFIX=$PREFIX install
    cd ..
fi

if [ ! -d $SITE_PACKAGES/numpy ]
then
    cd $BUILD_DIR

    if [ ! -d numpy ]
    then
        git clone git@github.com:numpy/numpy.git
    fi

    cd numpy
    git checkout master
    git reset --hard origin/master
    git checkout v1.8.0
    echo "$NUMPY_AND_SCIPY_CONFIG" > site.cfg
    $PYTHON setup.py install
fi
################################################################################


################################################################################
# SciPy and dependencies
# UMFPACK: Requires AMD, and UFconfig to be in the same parent
# directory.
if [ ! -f $LIB/libumfpack.a ]
then
    cd $BUILD_DIR
    if [ ! -f UMFPACK.tar.gz ]
    then
        wget http://www.cise.ufl.edu/research/sparse/umfpack/current/UMFPACK.tar.gz
    fi

    if [ ! -f AMD-2.3.1.tar.gz ]
    then
        wget http://www.cise.ufl.edu/research/sparse/amd/AMD-2.3.1.tar.gz
    fi

    if [ ! -f SuiteSparse_config-4.2.1.tar.gz ]
    then
        wget http://www.cise.ufl.edu/research/sparse/UFconfig/SuiteSparse_config-4.2.1.tar.gz
    fi
    rm -rf UMFPACK
    rm -rf AMD
    rm -rf SuiteSparse_config

    tar -xzf UMFPACK.tar.gz
    tar -xzf AMD-2.3.1.tar.gz
    tar -xzf SuiteSparse_config-4.2.1.tar.gz

    # Configure all three.
    SUITE_SPARSE_CONFIG="CF = \$(CFLAGS) \$(CPPFLAGS) \$(TARGET_ARCH) -O3 -fexceptions -fPIC
RANLIB = ranlib
ARCHIVE = \$(AR) \$(ARFLAGS)
CP = cp -f
MV = mv -f
F77 = gfortran
F77FLAGS = \$(FFLAGS) -O
F77LIB =
LIB = -lm
INSTALL_LIB = $LIB
INSTALL_INCLUDE = $INCLUDE
BLAS = -lopenblas -lgfortran -L$LIB -I$INCLUDE
LAPACK = -lopenblas -lgfortran -L$LIB -I$INCLUDE
UMFPACK_CONFIG = -DNCHOLMOD
CF = \$(CFLAGS) -O3 -fno-common -fexceptions -DNTIMER
LIB = -lm
CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d *.gcda *.gcno"
    echo "$SUITE_SPARSE_CONFIG" > $BUILD_DIR/SuiteSparse_config/SuiteSparse_config.mk

    cd $BUILD_DIR/UMFPACK
    make
    make install
    # The AMD and SuiteSparse headers are also needed.
    cd $BUILD_DIR/AMD
    make install
    cd $BUILD_DIR/SuiteSparse_config
    make install
    cd $BUILD_DIR
fi

# SciPy
if [ ! -d $SITE_PACKAGES/scipy ]
then
    cd $BUILD_DIR

    if [ ! -d scipy ]
    then
        git clone git@github.com:scipy/scipy.git
    fi

    cd scipy
    git checkout master
    git reset --hard origin/master
    git checkout v0.13.2
    echo "$NUMPY_AND_SCIPY_CONFIG" > site.cfg
    $PYTHON setup.py install
fi
################################################################################


#################################################################################
## lxml dependencies. The statically compiled version does not seem to work.
if [ ! -f $LIB/liblzma.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f xz-5.0.5.tar.gz ]
    then
        wget ...
    fi
    rm -rf xz-5.0.5
    tar -xzf xz-5.0.5.tar.gz
    cd xz-5.0.5
    ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -f $LIB/libxml2.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f libxml2-2.9.1.tar.gz ]
    then
        wget ...
    fi
    rm -rf libxml2-2.9.1
    tar -xzf libxml2-2.9.1.tar.gz
    cd libxml2-2.9.1
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -f $LIB/libxslt.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f libxslt-1.1.28.tar.gz ]
    then
        wget ...
    fi
    rm -rf libxslt-1.1.28
    tar -xzf libxslt-1.1.28.tar.gz
    cd libxslt-1.1.28
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -d $SITE_PACKAGES/lxml ]
then
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE $PIP install lxml
fi
#################################################################################


#################################################################################
## Matplotlib, basemap and dependencies
if [ ! -f $LIB/libpng.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f libpng-1.6.8.tar.gz ]
    then
        wget "http://download.sourceforge.net/libpng/libpng-1.6.8.tar.gz"
    fi
    rm -rf libpng-1.6.8
    tar -xzf libpng-1.6.8.tar.gz
    cd libpng-1.6.8
    CPPFLAGS="-I$INCLUDE" LDFLAGS="-L$LIB -lz" ./configure --prefix=$PREFIX
    make
    make install
    cd ..
fi

if [ ! -f $LIB/libfreetype.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f freetype-2.5.2.tar.gz ]
    then
        wget http://download.savannah.gnu.org/releases/freetype/freetype-2.5.2.tar.gz
    fi
    rm -rf freetype-2.5.2
    tar -xzf freetype-2.5.2.tar.gz
    cd freetype-2.5.2
    # The configure scripts needs libpng-config which is the bin directory...
    PATH=$PREFIX/bin:$PATH LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX
    make
    make install
    cd ..
fi

if [ ! -d $SITE_PACKAGES/matplotlib ]
then
    cd $BUILD_DIR
    if [ ! -f matplotlib-1.3.1.tar.gz ]
    then
        wget https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.1/matplotlib-1.3.1.tar.gz
    fi
    # A simple pip install will install an older matplotlib version as they
    # messed their pypi packaging up a little bit.
    LDFLAGS="-L$LIB" CPPFLAGS="-I$INCLUDE -I$INCLUDE/freetype2" $PIP install matplotlib-1.3.1.tar.gz
fi

# GEOS is statically compiled. Here it will use the homebrew version.
if [ ! -d $SITE_PACKAGES/mpl_toolkits/basemap ]
then
    GEOS_DIR=/usr/local/lib $PIP install --allow-external basemap basemap --allow-unverified basemap
fi
#################################################################################


if [ ! -d $SITE_PACKAGES/suds ]
then
    $PIP install suds
fi


if [ ! -d $SITE_PACKAGES/sqlalchemy ]
then
    $PIP install sqlalchemy
fi


if [ ! -d $SITE_PACKAGES/flake8 ]
then
    $PIP install flake8
fi


if [ ! -d $SITE_PACKAGES/flake8 ]
then
    $PIP install flake8
fi


if [ ! -d $SITE_PACKAGES/pandas ]
then
    $PIP install pandas
fi

if [ ! -d $SITE_PACKAGES/colorama ]
then
    $PIP install colorama
fi


if [ ! -f $LIB/libjpeg.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f jpegsrc.v9.tar.gz ]
    then
        wget http://www.ijg.org/files/jpegsrc.v9.tar.gz
    fi
    rm -rf jpeg-9
    tar -xzf jpegsrc.v9.tar.gz
    cd jpeg-9
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -d $SITE_PACKAGES/PIL ]
then
    $PIP install pil --allow-external pil --allow-unverified pil
fi


#################################################################################
# pyproj including dependencies
if [ ! -f $LIB/libproj.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f proj-4.8.0.tar.gz ]
    then
        wget http://download.osgeo.org/proj/proj-4.8.0.tar.gz
    fi
    rm -rf proj-4.8.0
    tar -xzf proj-4.8.0.tar.gz
    cd proj-4.8.0
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -f $SITE_PACKAGES/_pyproj.so ]
then
    PROJ_DIR=$PREFIX $PIP install pyproj
fi
#################################################################################


#################################################################################
# mlpy including dependencies
if [ ! -f $LIB/libgsl.dylib ]
then
    cd $BUILD_DIR
    if [ ! -f gsl-1.16.tar.gz ]
    then
        wget http://ftp.halifax.rwth-aachen.de/gnu/gsl/gsl-1.16.tar.gz
    fi
    rm -rf gsl-1.16
    tar -xzf gsl-1.16.tar.gz
    cd gsl-1.16
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE ./configure --prefix=$PREFIX
    make
    make install
fi

if [ ! -d $SITE_PACKAGES/mlpy ]
then
    cd $BUILD_DIR
    if [ ! -f mlpy-3.5.0.tar.gz ]
    then
        wget http://sourceforge.net/projects/mlpy/files/mlpy%203.5.0/mlpy-3.5.0.tar.gz/download -O mlpy-3.5.0.tar.gz
    fi
    rm -rf mlpy-3.5.0
    tar -xzf mlpy-3.5.0.tar.gz
    cd mlpy-3.5.0
    LDFLAGS=-L$LIB CPPFLAGS=-I$INCLUDE $PIP install .
fi
#################################################################################



if [ ! -d $SITE_PACKAGES/obspy ]
then
    $PIP install obspy
fi

# Adjust the links to all gfortan libraries so they point to the one in the app folder.
for file_to_be_modified in $(find $PREFIX -type f -name \*.dylib -o -name \*.so)
do
    for brew_name in $GFORTRAN_FILES
    do
        new_name=$LIB/$(basename $brew_name)
        install_name_tool -change $brew_name $new_name $file_to_be_modified
    done
done
