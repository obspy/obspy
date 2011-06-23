# evalresp makefile for use with Microsoft Visual C++ 6.0, 8/28/2001 -- [ET]
#  10/20/2005 -- [ET]  Added "evr_spline.c".

NODEBUG=1
#PROFILE=1

!include <ntwin32.mak>

OBJS = file_ops.obj evresp.obj alloc_fctns.obj resp_fctns.obj regsub.obj \
    regexp.obj regerror.obj print_fctns.obj parse_fctns.obj evalresp.obj \
    error_fctns.obj calc_fctns.obj string_fctns.obj evr_spline.obj

all: evalresp.exe

evalresp.exe: $(OBJS)
   $(link) $(conlflags) $(ldebug) $(conlibsmt) $(OBJS) -out:evalresp.exe

.c.obj:
   $(cc) $(cflags) $(cdebug) $(cvarsmt) $(tflags) -DDEBUG $<


# Clean-up directives
clean:
	del *.obj

clean_bin:
	del evalresp.exe
