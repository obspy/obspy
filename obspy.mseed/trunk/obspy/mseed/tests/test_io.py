#!/usr/bin/python

import sys, os
from numpy import *
from obspy.mseed.libmseed import libmseed

try:
    file = sys.argv[1]
except:
    file = os.path.join("data","test.mseed")

S = libmseed()
h,d,n = S.read_ms(file)
S.write_ms(h,d,outfile='t.mseed')

T = libmseed()
h1,d1,n1 = T.read_ms("t.mseed")
#
print "len d:",len(d),"len d1:",len(d1)
print array(d1)
print array(d[0:len(d1)])
print h1
print h
#
#
#import pdb;pdb.set_trace()
