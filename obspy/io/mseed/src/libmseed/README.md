
# libmseed - The Mini-SEED library

The Mini-SEED library provides a framework for manipulation of SEED
data records including the unpacking and packing of data records.
Functionality is also included for managing waveform data as
continuous traces.

All structures of SEED 2.4 data records are supported with the
following exceptions: Blockette 2000 opaque data which has an unknown
data structure by definition and Blockette 405 which depends on full
SEED (SEED including full ASCII headers) for a full data description.

The library should work in Linux, BSD (and derivatives like macOS),
Solaris and Win32 environments.

## Documentation

The [Wiki](https://github.com/iris-edu/libmseed/wiki) provides an
overview of using the library. For function level documentation,
man pages are included in the [doc](doc) directory.

## Downloading and installing

The [releases](https://github.com/iris-edu/libmseed/releases) area
contains release versions.

For installation instructions see the [INSTALL](INSTALL.md) file.
For further information regarding the library interface see the
documentation in the 'doc' directory.  For example uses of libmseed
see the source code in the 'examples' directory.

## License

Copyright (C) 2016 Chad Trabant, IRIS Data Management Center

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License (GNU-LGPL) for more details.

You should have received a copy of the GNU Lesser General Public
License along with this software.
If not, see <https://www.gnu.org/licenses/>.

## Acknowlegements

Numerous improvements have been incorporated based on feedback and
patches submitted by others.  Individual acknowlegements are included
in the ChangeLog.

This library also uses code bits published in the public domain and
acknowledgement is included in the code whenever possible.

With software provided by http://2038bug.com/ (site offline, checked Oct. 2017)
