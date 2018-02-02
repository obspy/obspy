#!/bin/bash
#-------------------------------------------------------------------
# Filename: deb__build_repo.sh
#  Purpose: Build basic Debian repository structure from scratch
#   Author: Moritz Beyreuther, Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011-2012 ObsPy Development Team
#---------------------------------------------------------------------

BUILDDIR=/tmp/python-obspy_build
DEBDIR=$BUILDDIR/deb

rm -rf $DEBDIR
mkdir -p $DEBDIR/conf $DEBDIR/dists $DEBDIR/incoming

cat > $DEBDIR/conf/distributions << EOF
Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Suite: stable
Codename: stretch
Version: 9.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Suite: oldstable
Codename: jessie
Version: 8.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Suite: oldoldstable
Codename: wheezy
Version: 7.0
Architectures: amd64 i386 armhf
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: precise
Version: 12.04 LTS
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: trusty
Version: 14.04 LTS
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: xenial
Version: 16.04 LTS
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: zesty
Version: 17.04
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: artful
Version: 17.10
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz
EOF
