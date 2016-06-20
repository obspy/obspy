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
from obspy.core.trace import BaseStats, TimeSeriesStats, Stats, TimeSeriesTrace
from obspy.core.frequency_domain_trace import FrequencyDomainTrace, TimeSeriesFrequencyDomainTrace

import matplotlib.pyplot as plt

class TraceTestCase(unittest.TestCase):

    def test_fft_circle_Trace(self):
        """
        Tests the fft and ifft Method circle of an Trace/FrequencyDomainTrace class
        """
        st = read()
        tr = st[0]
        #Test of the fft Method
        tr_f = tr.fft()
        self.assertTrue(isinstance(tr_f, FrequencyDomainTrace))
        #Test of the ifft Method
        tr2 = tr_f.ifft()        
        self.assertTrue(isinstance(tr2, Trace))        
        #Test of the consistency of the data 
        if (np.testing.assert_allclose(tr.data, tr2.data, atol=10e-6)) is not None:
            msg = "data in Trace tr and Trace (tr.fft).ifft are not the same"
            raise Exception(msg) 
        else:
            pass
        #Test of the consistency of the stats information
        self.assertTrue(tr.stats.network==tr2.stats.network)


    def test_fft_circle_TimeSeriesTrace(self): 
        """
        Tests the fft and ifft Method circle of an TimeSeriesTrace/
        TimeSeriesFrequencyDomainTrace class
        """
        st = read()
        tr = st[0]
        tr_t = TimeSeriesTrace()
        tr_t.data = tr.data
        tr_t.stats = tr.stats

        #Test of the fft Method
        tr_f = tr_t.fft()
        self.assertTrue(isinstance(tr_f, TimeSeriesFrequencyDomainTrace))
        #Test of the ifft Method
        tr_t2 = tr_f.ifft()        
        self.assertTrue(isinstance(tr_t2, TimeSeriesTrace))        
        #Test of the consistency of the data 
        if (np.testing.assert_allclose(tr_t.data, tr_t2.data, atol=10e-6)) is not None:
            msg = "data in TimeSeriesTrace tr and Trace (tr.fft).ifft are not the same"
            raise Exception(msg) 
        else:
            pass
        #Test of the consistency of the stats information
        self.assertTrue(tr_t.stats.network==tr_t2.stats.network)      


    def test_amplitude(self):
        """
        Tests the polar method of the FrequencyDomainTrace
        """
        st = read()
        tr = st[0]
        tr_f = tr.fft()
        r = tr_f.amplitude

        np.testing.assert_allclose(r, (np.sqrt(((tr_f.data.imag)**2)+
        (tr_f.data.real)**2)), atol = 10e-9)        

    def test_phase(self):
        """
        Tests the phase method of the FrequencyDomainTrace
        """
        st = read()
        tr = st[0]
        tr_f = tr.fft()
        deg = tr_f.phase

        np.testing.assert_allclose(deg[5], (np.arctan((tr_f.data[5].imag)/
        (tr_f.data[5].real))) , atol = 10e-9)        

def suite():
    return unittest.makeSuite(TraceTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
