#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__upload_repo.sh
#  Purpose: Upload Debian repository to deb.obspy.org
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011-2012 ObsPy Development Team
#---------------------------------------------------------------------

BUILDDIR=/tmp/python-obspy_build
DEBDIR=$BUILDDIR/deb

FTPHOST=obspy.org
FTPUSER=obspy

# upload complete repo
read -s -p "Give password for FTPUSER $FTPUSER and press [ENTER]: " FTPPASSWD
echo
lftp << EOF
set ftp:ssl-allow 0
open $FTPUSER:$FTPPASSWD@$FTPHOST
mirror --reverse --delete-first --verbose $DEBDIR /debian/deb
bye
EOF
