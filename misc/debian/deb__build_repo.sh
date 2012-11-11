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
mkdir -p $DEBDIR/conf $DEBDIR/dists

cat > $DEBDIR/conf/distributions << EOF
Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Suite: testing
Codename: wheezy
Version: 7.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Suite: stable
Codename: squeeze
Version: 6.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Suite: oldstable
Codename: lenny
Version: 5.0
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: lucid
Version: 10.04 LTS
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: maverick
Version: 10.10
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: natty
Version: 11.04
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz

Origin: ObsPy Development Team
Label: ObsPy Apt Repository
Codename: oneiric
Version: 11.10
Architectures: amd64 i386
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
Codename: quantal
Version: 12.10
Architectures: amd64 i386
Components: main
Description: ObsPy Apt Repository
SignWith: 34811F05
Contents: . .gz
EOF
