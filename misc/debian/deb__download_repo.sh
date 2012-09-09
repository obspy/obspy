#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__download_repo.sh
#  Purpose: Download Debian repository from deb.obspy.org
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 ObsPy Development Team
#---------------------------------------------------------------------

# Must be executed in the misc/debian directory
BASEDIR=`pwd`
DEBDIR=$BASEDIR/deb

FTPHOST=obspy.org
FTPUSER=obspy

# remove old repo structure if present
rm -rf $DEBDIR

# download deb repository from server
echo -n "Give password for FTPUSER $FTPUSER and press [ENTER]: "
read FTPPASSWD
lftp << EOF
set ftp:ssl-allow 0
open $FTPUSER:$FTPPASSWD@$FTPHOST
mirror --delete-first --verbose /debian/deb $DEBDIR
bye
EOF
