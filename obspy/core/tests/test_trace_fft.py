## -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

#from obspy.core.trace import FrequencyDomainTrace
import math
import os
import unittest
from copy import deepcopy
import warnings

import numpy as np
import numpy.ma as ma
import scipy as sp

from obspy import Stream, Trace, UTCDateTime, __version__, read, read_inventory
from obspy.core.compatibility import mock
from obspy.core.util.testing import ImageComparison
from obspy.io.xseed import Parser
from obspy.core.trace import FrequencyDomainTrace, BaseTrace

import matplotlib.pyplot as plt

class TraceTestCase(unittest.TestCase):

    def test_fft_circle(self):
        
        #Trace
        st=read()
        tr=st[0]

        #added a sinus-wave
        x = np.linspace(-np.pi,np.pi,201)
        y = np.sin(x)   
        tr.data = y

#########

        #Spectrum after fft
        tr_f=tr.fft()

        #Trace after ifft
        tr2 = tr_f.ifft()
        
        #control of the output
        print (np.testing.assert_allclose(tr.data, tr2.data, atol=1))
        print (tr.stats==tr_f.stats==tr2.stats)

###########

        #plots
        plt.figure(1)
        plt.subplot(311)
        plt.plot(tr.data)
        plt.subplot(312)
        plt.plot(tr_f.data)
        plt.subplot(313)
        plt.plot(tr2.data)
        plt.show()

############
               
def suite():
    return unittest.makeSuite(TraceTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
