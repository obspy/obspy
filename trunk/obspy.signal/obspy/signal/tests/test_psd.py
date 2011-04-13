#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The PSD test suite.
"""

from __future__ import with_statement
import os
import unittest
import numpy as np
from obspy.core import read
from obspy.signal import PPSD


class PSDTestCase(unittest.TestCase):
    """
    Test cases for PSD.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_PPSD(self):
        """
        Test PPSD routine with some real data. Data was downsampled to 100Hz
        so the ppsd is a bit distorted which does not matter for the purpose
        of testing.
        """
        # load test file
        file_mseed = os.path.join(self.path,
                'BW.KW1..EHZ.D.2011.090_downsampled')
        file_histogram = os.path.join(self.path,
                'BW.KW1..EHZ.D.2011.090_downsampled__ppsd_hist_stack.npy')
        file_binning = os.path.join(self.path,
                'BW.KW1..EHZ.D.2011.090_downsampled__ppsd_mixed.npz')
        # parameters for the test
        st = read(file_mseed)
        tr = st[0]
        paz = {'gain': 60077000.0,
               'poles': [(-0.037004+0.037016j), (-0.037004-0.037016j),
                         (-251.33+0j), (-131.04-467.29j), (-131.04+467.29j)],
               'sensitivity': 2516778400.0,
               'zeros': [0j, 0j]}
        ppsd = PPSD(tr.stats, paz)
        ppsd.add(st)
        # read results and compare
        result_hist = np.load(file_histogram)
        self.assertEqual(len(ppsd.times), 4)
        self.assertEqual(ppsd.nfft, 524288)
        self.assertEqual(ppsd.nlap, 393216)
        np.testing.assert_array_equal(ppsd.hist_stack, result_hist)
        # add the same data a second time (which should do nothing at all) and
        # test again
        ppsd.add(st)
        np.testing.assert_array_equal(ppsd.hist_stack, result_hist)
        # test the binning arrays
        binning = np.load(file_binning)
        np.testing.assert_array_equal(ppsd.spec_bins, binning['spec_bins'])
        np.testing.assert_array_equal(ppsd.period_bins, binning['period_bins'])


def suite():
    return unittest.makeSuite(PSDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
