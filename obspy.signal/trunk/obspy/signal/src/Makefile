# Makefile for coordtrans

CC = gcc
CFLAGS = -g -O1 -fno-inline -Wall

DEPS = \
	coordtrans.o \
	xcorr.o \
	recstalta.o \
	pk_mbaer.o \
	filt_util.o \
	arpicker.o \
	bbfk.o

FFT = $(shell python -c "import numpy, os; print os.path.dirname(numpy.fft.__file__)")


signal.so: $(DEPS)
	$(CC) -shared $(CFLAGS) -L$(FFT) -l:fftpack_lite.so -lm -o signal.so $(DEPS)

install: signal.so
	cp signal.so ../lib/libsignal.so

clean:
	find ./ -name "*.o" -exec rm {} \;
	rm -f *.so
