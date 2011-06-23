## Process this file with automake to create Makefile.in

AUTOMAKE_OPTIONS = foreign

bin_PROGRAMS = evalresp

evalresp_SOURCE = evalresp.c

evalresp_LDADD =  -L. -levresp  -lm


#evalresp_SOURCES = regexp.c regerror.c regsub.c calc_fctns.c resp_fctns.c \
#file_ops.c evresp.c evalresp.c alloc_fctns.c parse_fctns.c print_fctns.c \
#string_fctns.c error_fctns.c evr_spline.c


lib_LTLIBRARIES = libevresp.la
libevresp_la_SOURCES = regexp.c regerror.c regsub.c calc_fctns.c resp_fctns.c \
        file_ops.c evresp.c alloc_fctns.c parse_fctns.c print_fctns.c \
        string_fctns.c error_fctns.c evr_spline.c


include_HEADERS =  evresp.h   regmagic.h  regexp.h  evr_spline.h
man_MANS = evalresp.1

EXTRA_DIST =  evresp.h   regmagic.h  regexp.h  \
        evalresp.ide makefile.bor  makefile.ms evtest.f depcomp evalresp.1

#libevresp_la_LDFLAGS = -version-info 2:3:40
