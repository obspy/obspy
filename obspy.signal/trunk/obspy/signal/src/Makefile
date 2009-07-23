# Makefile for C extensions of obspy
#

INCLUDES= \
	-I .

CFLAGS = -g -O3
CC = gcc
LD = ld

recstalta.so: recstalta.c
	$(CC) $(CFLAGS) -c recstalta.c
	$(CC) $(CFLAGS) -shared -lm -o recstalta.so recstalta.o

clean:
	rm *.so *.o
