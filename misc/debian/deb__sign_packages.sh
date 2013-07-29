#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__sign_packages.sh
#  Purpose: Sign built packages
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2013 ObsPy Development Team
#---------------------------------------------------------------------

BUILDDIR=/tmp/python-obspy_build
PACKAGEDIR=$BUILDDIR/packages
export GNUPGHOME=$HOME/.gnupg-obspy
GPGKEY=34811F05

for FILE in `ls $PACKAGEDIR/python-obspy_*.changes`; do
    dpkg-sig -k $GPGKEY --sign builder $FILE
done
