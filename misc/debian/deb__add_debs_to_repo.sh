#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__add_debs_to_repo.sh
#  Purpose: Add packages to Debian repository
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011-2012 ObsPy Development Team
#---------------------------------------------------------------------

BUILDDIR=/tmp/python-obspy_build
DEBDIR=$BUILDDIR/deb
PACKAGEDIR=$BUILDDIR/packages
export GNUPGHOME=$HOME/.gnupg-obspy

eval $(gpg-agent --daemon)
for FILE in `ls $PACKAGEDIR/python-obspy_*.deb`; do
    CODENAME=`echo $FILE | sed -e "s#.*~##" -e "s#_.*##"`
    reprepro --component 'main' -Vb $DEBDIR includedeb $CODENAME $FILE
done
killall gpg-agent

##if [ $? != 0 ]; then Error when running reprepro, exiting now; fi
## do reprepro check + checkpool! get return status
chmod -R a+rX $DEBDIR
