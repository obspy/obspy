# -*- coding: utf-8 -*-
"""
The obspy.signal.trigger test suite.
"""

from ctypes import ArgumentError
from obspy.signal import recStalta, recStaltaPy, triggerOnset, pkBaer, arPick
from obspy.signal.util import lib
import numpy as np
import unittest
import os
import inspect
import gzip


class TriggerTestCase(unittest.TestCase):
    """
    Test cases for obspy.trigger
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
        # random seed data
        np.random.seed(815)
        self.data = np.random.randn(int(1e5))
        pass

    def tearDown(self):
        pass

    def test_recStaltaC(self):
        """
        Test case for ctypes version of recStalta
        """
        nsta, nlta = 5, 10
        c1 = recStalta(self.data, nsta, nlta)
        self.assertAlmostEquals(c1[99], 0.80810165)
        self.assertAlmostEquals(c1[100], 0.75939449)
        self.assertAlmostEquals(c1[101], 0.91763978)
        self.assertAlmostEquals(c1[102], 0.97465004)

    def test_recStaltaPy(self):
        """
        Test case for python version of recStalta
        """
        nsta, nlta = 5, 10
        c2 = recStaltaPy(self.data, nsta, nlta)
        self.assertAlmostEquals(c2[99], 0.80810165)
        self.assertAlmostEquals(c2[100], 0.75939449)
        self.assertAlmostEquals(c2[101], 0.91763978)
        self.assertAlmostEquals(c2[102], 0.97465004)

    def test_recStaltaRaise(self):
        """
        Type checking recStalta
        """
        ndat = 1
        charfct = np.empty(ndat, dtype='float64')
        self.assertRaises(ArgumentError, lib.recstalta, [1], charfct,
                          ndat, 5, 10)
        self.assertRaises(ArgumentError, lib.recstalta,
                          np.array([1], dtype='int32'), charfct, ndat, 5, 10)

    def test_pkBaer(self):
        """
        Test pkBaer against implementation for UNESCO short course
        """
        file = os.path.join(self.path, 'manz_waldk.a01.gz')
        data = np.loadtxt(gzip.open(file), dtype='float32')
        df, ntdownmax, ntupevent, thr1, thr2, npreset_len, np_dur = \
            (200.0, 20, 60, 7.0, 12.0, 100, 100)
        nptime, pfm = pkBaer(data, df, ntdownmax, ntupevent,
                             thr1, thr2, npreset_len, np_dur)
        self.assertEquals(nptime, 17545)
        self.assertEquals(pfm, 'IPU0')

    def test_arPick(self):
        """
        Test arPick against implementation for UNESCO short course
        """
        data = []
        for channel in ['z', 'n', 'e']:
            file = os.path.join(self.path,
                                'loc_RJOB20050801145719850.' + channel)
            data.append(np.loadtxt(file, dtype='float32'))
        # some default arguments
        samp_rate, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s = \
                200.0, 1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2
        ptime, stime = arPick(data[0], data[1], data[2], samp_rate, f1, f2,
                              lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s)
        self.assertAlmostEquals(ptime, 30.6350002289)
        # seems to be strongly machine dependent, go for int for 64 bit
        #self.assertAlmostEquals(stime, 31.2800006866)
        self.assertEquals(int(stime + 0.5), 31)

    def test_triggerOnset(self):
        """
        Test trigger onset function
        """
        on_of = np.array([[6.0, 31], [69, 94], [131, 181], [215, 265],
                          [278, 315], [480, 505], [543, 568], [605, 631]])
        cft = np.concatenate((np.sin(np.arange(0, 5 * np.pi, 0.1)) + 1,
                              np.sin(np.arange(0, 5 * np.pi, 0.1)) + 2.1,
                              np.sin(np.arange(0, 5 * np.pi, 0.1)) + 0.4,
                              np.sin(np.arange(0, 5 * np.pi, 0.1)) + 1))
        picks = triggerOnset(cft, 1.5, 1.0, max_len=50)
        np.testing.assert_array_equal(picks, on_of)
        # check that max_len_delete drops the picks
        picks_del = triggerOnset(cft, 1.5, 1.0, max_len=50, max_len_delete=True)
        np.testing.assert_array_equal(picks_del, on_of[np.array([0, 1, 5, 6])])
        #
        # set True for visual understanding the tests
        if False:
            import matplotlib.pyplot as plt
            plt.plot(cft)
            plt.hlines([1.5, 1.0], 0, len(cft))
            on_of = np.array(on_of)
            plt.vlines(picks[:, 0], 1.0, 2.0, color='g', linewidth=2,
                       label="ON max_len")
            plt.vlines(picks[:, 1], 0.5, 1.5, color='r', linewidth=2,
                       label="OF max_len")
            plt.vlines(picks_del[:, 0] + 2, 1.0, 2.0, color='y', linewidth=2,
                       label="ON max_len_delete")
            plt.vlines(picks_del[:, 1] + 2, 0.5, 1.5, color='b', linewidth=2,
                       label="OF max_len_delete")
            plt.legend()
            plt.show()


def suite():
    return unittest.makeSuite(TriggerTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
