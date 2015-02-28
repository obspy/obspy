# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy.y.core import isY, readY


class CoreTestCase(unittest.TestCase):
    """
    Nanometrics Y file test suite.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_isYFile(self):
        """
        Testing Y file format.
        """
        testfile = os.path.join(self.path, 'data', 'YAYT_BHZ_20021223.124800')
        self.assertEqual(isY(testfile), True)
        self.assertEqual(isY("/path/to/slist.ascii"), False)
        self.assertEqual(isY("/path/to/tspair.ascii"), False)

    def test_readYFile(self):
        """
        Testing reading Y file format.
        """
        testfile = os.path.join(self.path, 'data', 'YAYT_BHZ_20021223.124800')
        st = readY(testfile)
        self.assertEqual(len(st), 1)
        tr = st[0]
        self.assertEqual(len(tr), 18000)
        self.assertEqual(tr.stats.sampling_rate, 100.0)
        self.assertEqual(tr.stats.station, 'AYT')
        self.assertEqual(tr.stats.channel, 'BHZ')
        self.assertEqual(tr.stats.location, '')
        self.assertEqual(tr.stats.network, '')
        self.assertEqual(max(tr.data),
                         tr.stats.y.tag_series_info.max_amplitude)
        self.assertEqual(min(tr.data),
                         tr.stats.y.tag_series_info.min_amplitude)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
