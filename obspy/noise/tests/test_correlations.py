#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import numpy as np
import os
import unittest

from obspy.noise import correlation_functions


class CorrelationsTestCase(unittest.TestCase):
    """
    Test suite for obspy.correlations
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_phase_autocorrelation(self):
        """
        Test auto-phase-correlation of sine wave.
        """
        sine1 = np.sin(np.linspace(0, 100*np.pi, 10000))
        pcc = correlation_functions.phase_xcorr(sine1, sine1, 300)

        # Is the length of the array correct
        self.assertEqual(len(pcc), 601)

        # Is the maximum value of the autocorrelation at lag 0
        self.assertEqual(pcc[300], pcc.max())

        # Is the maximum value at lag zero correct
        self.assertTrue(abs(pcc[300]-1.0) < 1e-9)
        self.assertTrue(abs(pcc[200]+1.0) < 0.01)
        self.assertTrue(abs(pcc[400]+1.0) < 0.01)

        # Is the autocorrelation symmetric to lag 0
        # XXX: still missing


def suite():
    return unittest.makeSuite(CorrelationsTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
