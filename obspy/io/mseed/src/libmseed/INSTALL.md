
The library requires that C99 integer types are available on the
target computer.  Specifically the int8_t, int16_t, int32_t, int64_t
and their unsigned counterpart types.

## Unix, Linux, macOS

A simple 'make' on most Unix-like systems should build the library.

The included Makefile should work for most Unix-like environments and
most make variants. It is know to work with GNU make, which, if not the
default, is sometimes installed as gmake.

The CC, CFLAGS, LDFLAGS and CPPFLAGS environment variables can be set
to control the build.

By default a statically linked version of the library is built: 'libmseed.a'.

With GCC, clang or compatible build tools it is possible to build a shared
library with 'make shared'.

A simple install method for the shared library can be invoked with
'make install'.  By default the installation destination is /usr/local.
The install destination may be specified using the PREFIX variable, for
example:

make install PREFIX=/path/to/install/

## Windows (Win32)

On a WIN32 platform the library can be compiled by using the
Nmake compatible Makefile.win (e.g. 'nmake -f Makefile.win') or Open
Watcom's Wmake with Makefile.wat (e.g. 'wmake -f Makefile.wat'). The
default target is a static library 'libmseed.lib'.  The library has
been tested with Open Watcom 1.8.  A libmseed.def file is included
for use building and linking a DLL.
