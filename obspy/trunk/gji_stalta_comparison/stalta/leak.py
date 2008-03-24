#!/usr/bin/env python2.5

import array, glob, os, time
import ext_recstalta
from numpy import *

def test():    
	samp_rate = 200
	sta_winlen=int(.25 * samp_rate)
	lta_winlen=int(2.5 * samp_rate)

	for i in xrange( int(1e9) ):
		a = random.random(72000000)
		stalta = zeros(len(a))
		ext_recstalta.rec_stalta(a,stalta,sta_winlen,lta_winlen)
		del a, stalta
	
test()
