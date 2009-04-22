#!/usr/bin/python

from pylab import *
from numpy import *
import obspy, obspy.filter, os, inspect
from scipy.signal import remez,convolve,get_window,firwin

file = os.path.join(os.path.dirname(inspect.getfile(obspy)),"gse2",
                    "tests","data","loc_RNON20040609200559.z")
g = obspy.Trace()
g.read(file,format='GSE2')

data = array(g.data,dtype='f')
newdata = data[0.45*1e4:0.59*1e4]

fmin = 5.
fmax = 20.

tworunzph = obspy.filter.lowpassZPHSH(newdata,fmin)
olifir = obspy.filter.lowpassFIR(newdata,fmin)

clf()
plot(newdata,'r',linewidth='1',label='Original Data')
plot(tworunzph,'b',linewidth='2',label='2 Run Zero Phase Butterworth Lowpass Filtered')
plot(olifir,'g',linewidth='2',label='FIR lowpass Filtered')
title("Compare Zero Phase Lowpass Methods, Lowpass @ 5Hz")
legend()
show()
