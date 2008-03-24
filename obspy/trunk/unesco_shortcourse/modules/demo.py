#!/usr/bin/python

from pylab import *
from numpy import *
from trigger import *

sample_rate=200
a=fromfile("loc_RJOB20020325181100.ascii",sep="\n",dtype=float32)
t=arange(len(a),dtype=double)/sample_rate


ax = subplot(511)
plot(t,a)

subplot(512, sharex=ax)
plot(t,classicStaLta(a,50,500))

subplot(513, sharex=ax)
plot(t,delayedStaLta(a,50,500))

subplot(514, sharex=ax)
plot(t,recursiveStaLta(a,50,500))

subplot(515, sharex=ax)
plot(t,zdetect(a,50,500))

show()
