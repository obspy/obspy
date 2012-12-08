#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__download_repo.sh
#  Purpose: Download Debian repository from deb.obspy.org
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011-2012 ObsPy Development Team
#---------------------------------------------------------------------

BUILDDIR=/tmp/python-obspy_build
DEBDIR=$BUILDDIR/deb

FTPHOST=obspy.org
FTPUSER=obspy

# remove old repo structure if present
rm -rf $DEBDIR

# download deb repository from server
read -s -p "Give password for FTPUSER $FTPUSER and press [ENTER]: " FTPPASSWD
echo
lftp << EOF
set ftp:ssl-allow 0
open $FTPUSER:$FTPPASSWD@$FTPHOST
mirror --delete-first --verbose /debian/deb $DEBDIR
bye
EOF
