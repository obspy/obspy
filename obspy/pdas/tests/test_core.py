# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest
import os
import numpy as np
from obspy import Stream, UTCDateTime
from obspy.pdas import readPDAS, isPDAS


class PDASTestCase(unittest.TestCase):
    """
    Test suite for pdas.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.testfile = os.path.join(self.path, 'data', 'p1246001.108')

    def test_readPDAS(self):
        """
        Tests the readPDAS function.
        """
        st = readPDAS(self.testfile)
        self.assertTrue(isinstance(st, Stream))
        self.assertTrue(len(st) == 1)
        tr = st[0]
        expected = [('COMMENT', 'GAINRANGED'),
                    ('DATASET', 'P1246001108'),
                    ('FILE_TYPE', 'LONG'),
                    ('HORZ_UNITS', 'Sec'),
                    ('SIGNAL', 'Channel1'),
                    ('VERSION', 'next'),
                    ('VERT_UNITS', 'Counts')]
        self.assertTrue(sorted(tr.stats.pop("pdas").items()) == expected)
        expected = [('_format', 'PDAS'),
                    (u'calib', 1.0),
                    (u'channel', u''),
                    (u'delta', 0.005),
                    (u'endtime', UTCDateTime(1994, 4, 18, 0, 0, 2, 495000)),
                    (u'location', u''),
                    (u'network', u''),
                    (u'npts', 500),
                    (u'sampling_rate', 200.0),
                    (u'starttime', UTCDateTime(1994, 4, 18, 0, 0)),
                    (u'station', u'')]
        self.assertTrue(sorted(tr.stats.items()) == expected)
        expected = np.array([895, 867, 747, 591, 359, -129, -185, 3, 115, 243],
                            dtype=np.int16)
        np.testing.assert_array_equal(tr.data[:10], expected)

    def test_isPDAS(self):
        """
        Tests the readPDAS function.
        """
        self.assertTrue(isPDAS(self.testfile))


def suite():
    return unittest.makeSuite(PDASTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
