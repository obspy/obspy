#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__add_debs_to_repo.sh
#  Purpose: Add packages to Debian repository
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 ObsPy Development Team
#---------------------------------------------------------------------

# Must be executed in the misc/debian directory
BASEDIR=`pwd`
DEBDIR=$BASEDIR/deb
PACKAGEDIR=$BASEDIR/packages
export GNUPGHOME=$HOME/.gnupg-obspy

for FILE in `ls $PACKAGEDIR/*.deb`; do
    CODENAME=`echo $FILE | sed -e "s#.*~##" -e "s#_.*##"`
    reprepro --ask-passphrase --component 'main' -Vb $DEBDIR includedeb $CODENAME $FILE
done
##if [ $? != 0 ]; then Error when running reprepro, exiting now; fi
## do reprepro check + checkpool! get return status
chmod -R a+rX $DEBDIR
