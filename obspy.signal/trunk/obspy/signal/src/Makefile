# Makefile for coordtrans

INCLUDES= \
	-I .

DEPENDS = \
	coordtrans.o \
	xcorr.o \
	recstalta.o \
	pk_mbaer.o \
	filt_util.o \
	arpicker.o

CFLAGS = -g -O3 
CC = gcc

signal.so: $(DEPENDS)
	$(CC) $(CFLAGS) -shared -lm -o signal.so $(DEPENDS)

install: signal.so
	cp signal.so ../lib/libsignal.so

clean:
	find ./ -name "*.o" -exec rm {} \;
	rm -f *.so
