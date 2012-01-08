#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__upload_repo.sh
#  Purpose: Upload Debian repository to deb.obspy.org
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

# upload complete repo
echo -n "Give password for FTPUSER $FTPUSER and press [ENTER]: "
read FTPPASSWD
lftp << EOF
set ftp:ssl-allow 0
open $FTPUSER:$FTPPASSWD@$FTPHOST
mirror --reverse --delete-first --verbose $DEBDIR /debian/deb
bye
EOF
