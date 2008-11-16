#!/usr/bin/python
from numpy import *
from pylab import plot, show
import ext_mseed
#header,data=ext_mseed.read("test.mseed",-1.0,-1.0)
#header,data=ext_mseed.read("bayern.mseed",-1.0,-1.0)
header,data=ext_mseed.read("bayern.mseed",10.,10.)
print header
plot(data)
show()
