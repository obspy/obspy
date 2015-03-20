#
#
# Wmake File For libmseed - For Watcom's wmake
# Use 'wmake -f Makefile.wat'

.BEFORE
	@set INCLUDE=.;$(%watcom)\H;$(%watcom)\H\NT
	@set LIB=.;$(%watcom)\LIB386

cc     = wcc386
cflags = -zq
lflags = OPT quiet OPT map
cvars  = $+$(cvars)$- -DWIN32

# To build a DLL uncomment the following two lines
#cflags = -zq -bd
#lflags = OPT quiet OPT map SYS nt_dll

LIB = libmseed.lib
DLL = libmseed.dll

INCS = -I.

OBJS=	fileutils.obj	&
	genutils.obj	&
	gswap.obj	&
	lmplatform.obj	&
	lookup.obj	&
	msrutils.obj	&
	pack.obj	&
	packdata.obj	&
	traceutils.obj	&
	tracelist.obj	&
	parseutils.obj	&
	unpack.obj	&
	unpackdata.obj  &
	selection.obj	&
	logging.obj

all: lib

lib:	$(OBJS) .SYMBOLIC
	wlib -b -n -c -q $(LIB) +$(OBJS)

dll:	$(OBJS) .SYMBOLIC
	wlink $(lflags) name libmseed file {$(OBJS)}

# Source dependencies:
fileutils.obj:	fileutils.c libmseed.h
genutils.obj:	genutils.c libmseed.h
gswap.obj:	gswap.c lmplatform.h
lmplatform.obj:	lmplatform.c libmseed.h lmplatform.h
lookup.obj:	lookup.c libmseed.h
msrutils.obj:	msrutils.c libmseed.h
pack.obj:	pack.c libmseed.h packdata.h steimdata.h
packdata.obj:	packdata.c libmseed.h packdata.h steimdata.h
traceutils.obj:	traceutils.c libmseed.h
tracelist.obj:	tracelist.c libmseed.h
parseutils.obj:	parseutils.c libmseed.h
unpack.obj:	unpack.c libmseed.h unpackdata.h steimdata.h
unpackdata.obj:	unpackdata.c libmseed.h unpackdata.h steimdata.h
logging.obj:	logging.c libmseed.h

# How to compile sources:
.c.obj:
	$(cc) $(cflags) $(cvars) $(INCS) $[@ -fo=$@

# Clean-up directives:
clean:	.SYMBOLIC
	del *.obj *.map
	del $(LIB) $(DLL)
