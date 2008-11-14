#!/usr/bin/python

import obspy
import ext_gse
from numpy import *

rand = array(1000*random.random(1e7),dtype='l')

file = "leak.gse"

obspy.writegse({}, rand, file)
print rand[0:10]
print rand[-11:-1]

del rand

while True:
	(header,data) = obspy.readgse(file)
	print len(data)
	print data[0:10]
	print data[-11:-1]
	del header, data

os.remove(file)
