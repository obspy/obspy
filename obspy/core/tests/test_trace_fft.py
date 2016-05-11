# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

# from obspy.core.trace import FrequencyDomainTrace
import math
import os
import unittest
from copy import deepcopy
import warnings

import numpy as np
import scipy as sp

from obspy import Stream, Trace, UTCDateTime, __version__, read, read_inventory
from obspy.core.compatibility import mock
from obspy.core.util.testing import ImageComparison
from obspy.io.xseed import Parser
from obspy.core.trace import FrequencyDomainTrace, BaseTrace

import matplotlib.pyplot as plt

class TraceTestCase(unittest.TestCase):

    def test_fft_circle(self):
        # Trace
        st = read()
        tr = st[0]
        # added a sinus-wave
        #x = np.linspace(0,4*(np.pi), 201)
        #y = np.sin(x)
        #tr.data = y

#########
        #test for fft and ifft       
        # Spectrum after fft
        tr_f = tr.fft()
        # Trace after ifft
        tr2 = tr_f.ifft()
        # control of the output
        np.testing.assert_allclose(tr.data, tr2.data, atol=10e-8)

############
        # test for polar coordinates
        #print (tr_f.data[2])
        #r, deg = tr_f.polar()
        
        #test für r 
        #np.testing.assert_allclose(r, (np.sqrt(((tr_f.data.imag)**2)+
        #(tr_f.data.real)**2)), atol = 10e-9)        

        #test für deg
        # (np.testing.assert_allclose(deg[5], (np.arctan((tr_f.data[5].imag)/(tr_f.data[5].real)))         , atol = 10e-9))        

############
     
        # plots
        #plt.figure(1)
        # plt.subplot(311)
        # plt.plot(tr.data)
        #plt.subplot(312)
        #plt.plot(tr_f.data)
        # plt.subplot(313)
        # plt.plot(tr2.data)
        #plt.show()

############
     
        #tr_f.plot_psd_trace()


############

        tr2 = tr_f
        co = tr_f.cross_correlation(tr2)
       
        #plt.plot(co)
        #plt.show()

###########
        freq = tr_f.frequencies()
        plt.plot(freq)
        plt.show()

###########
def suite():
    return unittest.makeSuite(TraceTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
