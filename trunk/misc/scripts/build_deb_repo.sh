#!/bin/bash
#-------------------------------------------------------------------
# Filename: update_deb_repo.sh
#  Purpose: Update Debian repository on deb.obspy.org 
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 ObsPy Development Team
#---------------------------------------------------------------------

###CODENAME=`lsb_release -cs`

BASEDIR=`pwd`
DEBDIR=$BASEDIR/deb
###PACKAGEDIR=$BASEDIR/packages

# Must be executed in the scripts directory
###FTPHOST=obspy.org
###FTPUSER=obspy
###export GNUPGHOME=$HOME/.gnupg-obspy

#
# generating repo structure
#
rm -rf $DEBDIR
mkdir -p $DEBDIR/conf $DEBDIR/dists

#### download all packages from deb archive on server
###echo -n "Give password for FTPUSER $FTPUSER and press [ENTER]: "
###read FTPPASSWD
###lftp << EOF
###set ftp:ssl-allow 0
###open $FTPUSER:$FTPPASSWD@$FTPHOST
###mirror --delete-first --verbose /debian/packages $PACKAGEDIR
###bye
###EOF

# build repo structure
cat > $DEBDIR/conf/distributions << EOF
Origin: ObsPy Developer Team
Label: ObsPy Apt Repository
Suite: stable
Codename: squeeze
Version: 6.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Developer Team
Label: ObsPy Apt Repository
Suite: oldstable
Codename: lenny
Version: 5.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Developer Team
Label: ObsPy Apt Repository
Codename: lucid
Version: 10.04 LTS
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Developer Team
Label: ObsPy Apt Repository
Codename: maverick
Version: 10.10
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Developer Team
Label: ObsPy Apt Repository
Codename: natty
Version: 11.04
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Developer Team
Label: ObsPy Apt Repository
Codename: oneiric
Version: 11.10
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz
EOF
###for CNAME in `ls $PACKAGEDIR`; do
###    reprepro --ask-passphrase --component 'main' -Vb $DEBDIR includedeb $CNAME $PACKAGEDIR/$CNAME/*.deb
###done
###if [ $? != 0 ]; then Error when running reprepro, exiting now; fi
###chmod -R a+rX $DEBDIR
###
#### upload complete repo
####echo -n "Give password for FTPUSER $FTPUSER and press [ENTER]: "
####read FTPPASSWD
###lftp << EOF
###set ftp:ssl-allow 0
###open $FTPUSER:$FTPPASSWD@$FTPHOST
###mirror --reverse --delete-first --verbose $DEBDIR /debian/deb
###bye
###EOF
