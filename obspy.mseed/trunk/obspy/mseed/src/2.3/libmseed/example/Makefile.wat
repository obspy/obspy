#
# Wmake file - For Watcom's wmake
# Use 'wmake -f Makefile.wat'

.BEFORE
	@set INCLUDE=.;$(%watcom)\H;$(%watcom)\H\NT
	@set LIB=.;$(%watcom)\LIB386

cc     = wcc386
cflags = -zq
lflags = OPT quiet OPT map LIBRARY ..\libmseed.lib
cvars  = $+$(cvars)$- -DWIN32

BINS = msrepack.exe msview.exe

INCS = -I..

all: $(BINS)

msrepack.exe:	msrepack.obj
	wlink $(lflags) name msrepack file {msrepack.obj}

msview.exe:	msview.obj
	wlink $(lflags) name msview file {msview.obj}

# Source dependencies:
msrepack.obj:	msrepack.c
msview.obj:	msview.c

# How to compile sources:
.c.obj:
	$(cc) $(cflags) $(cvars) $(INCS) $[@ -fo=$@

# Clean-up directives:
clean:	.SYMBOLIC
	del *.obj *.map
	del $(BINS)
