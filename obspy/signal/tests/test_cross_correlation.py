# -*- coding: utf-8 -*-
"""
The cross correlation test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy import UTCDateTime, read
from obspy.core.util.testing import ImageComparison
from obspy.signal.cross_correlation import xcorr_pick_correction


class CrossCorrelationTestCase(unittest.TestCase):
    """
    Cross corrrelation test case
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        self.path_images = os.path.join(os.path.dirname(__file__), 'images')

    def test_xcorrPickCorrection(self):
        """
        Test cross correlation pick correction on a set of two small local
        earthquakes.
        """
        st1 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.a.slist.gz'))
        st2 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.b.slist.gz'))

        tr1 = st1.select(component="Z")[0]
        tr2 = st2.select(component="Z")[0]
        tr1_copy = tr1.copy()
        tr2_copy = tr2.copy()
        t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
        t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

        dt, coeff = xcorr_pick_correction(t1, tr1, t2, tr2, 0.05, 0.2, 0.1)
        self.assertAlmostEqual(dt, -0.014459080288833711)
        self.assertAlmostEqual(coeff, 0.91542878457939791)
        dt, coeff = xcorr_pick_correction(t2, tr2, t1, tr1, 0.05, 0.2, 0.1)
        self.assertAlmostEqual(dt, 0.014459080288833711)
        self.assertAlmostEqual(coeff, 0.91542878457939791)
        dt, coeff = xcorr_pick_correction(
            t1, tr1, t2, tr2, 0.05, 0.2, 0.1, filter="bandpass",
            filter_options={'freqmin': 1, 'freqmax': 10})
        self.assertAlmostEqual(dt, -0.013025086360067755)
        self.assertAlmostEqual(coeff, 0.98279277273758803)
        self.assertEqual(tr1, tr1_copy)
        self.assertEqual(tr2, tr2_copy)

    def test_xcorrPickCorrection_images(self):
        """
        Test cross correlation pick correction on a set of two small local
        earthquakes.
        """
        st1 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.a.slist.gz'))
        st2 = read(os.path.join(self.path,
                                'BW.UH1._.EHZ.D.2010.147.b.slist.gz'))

        tr1 = st1.select(component="Z")[0]
        tr2 = st2.select(component="Z")[0]
        t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
        t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

        with ImageComparison(self.path_images, 'xcorr_pick_corr.png') as ic:
            dt, coeff = xcorr_pick_correction(
                t1, tr1, t2, tr2, 0.05, 0.2, 0.1, plot=True, filename=ic.name)


def suite():
    return unittest.makeSuite(CrossCorrelationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
