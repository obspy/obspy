# -*- coding: utf-8 -*-
"""
The seisan.core test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np

from obspy.core.utcdatetime import UTCDateTime
from obspy.seisan.core import _getVersion, isSEISAN, readSEISAN


class CoreTestCase(unittest.TestCase):
    """
    Test cases for SEISAN core interfaces.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_getVersion(self):
        """
        Tests resulting version strings of SEISAN file.
        """
        # 1 - big endian, 32 bit
        file = os.path.join(self.path, '1996-06-03-1917-52S.TEST__002')
        with open(file, 'rb') as fp:
            data = fp.read(80 * 12)
        self.assertEqual(_getVersion(data), ('>', 32, 7))
        # 2 - little endian, 32 bit
        file = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        with open(file, 'rb') as fp:
            data = fp.read(80 * 12)
        self.assertEqual(_getVersion(data), ('<', 32, 7))

    def test_isSEISAN(self):
        """
        Tests SEISAN file check.
        """
        # 1 - big endian, 32 bit
        file = os.path.join(self.path, '1996-06-03-1917-52S.TEST__002')
        self.assertTrue(isSEISAN(file))
        # 2 - little endian, 32 bit
        file = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        self.assertTrue(isSEISAN(file))

    def test_readSEISAN(self):
        """
        Test SEISAN file reader.
        """
        # 1 - big endian, 32 bit
        file = os.path.join(self.path, '9701-30-1048-54S.MVO_21_1')
        st1 = readSEISAN(file)
        st1.verify()
        self.assertEqual(len(st1), 21)
        self.assertEqual(st1[20].stats.network, '')
        self.assertEqual(st1[20].stats.station, 'MBGB')
        self.assertEqual(st1[20].stats.location, 'J')
        self.assertEqual(st1[20].stats.channel, 'SBE')
        self.assertEqual(st1[20].stats.starttime,
                         UTCDateTime('1997-01-30T10:48:54.040000Z'))
        self.assertEqual(st1[20].stats.endtime,
                         UTCDateTime('1997-01-30T10:49:42.902881Z'))
        self.assertAlmostEqual(st1[20].stats.sampling_rate, 75.2, 1)
        self.assertEqual(st1[20].stats.npts, 3675)
        self.assertAlmostEqual(st1[20].stats.delta, 0.0133, 4)
        datafile = os.path.join(self.path, 'MBGBSBJE')
        # compare with ASCII values of trace
        # XXX: extracted ASCII file contains less values than the original
        # Seisan file!
        self.assertEqual(list(st1[20].data[0:3665]),
                         np.loadtxt(datafile, dtype=np.int32).tolist())
        # 2 - little endian, 32 bit
        file = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        st2 = readSEISAN(file)
        st2.verify()
        self.assertEqual(len(st2), 4)
        self.assertEqual(list(st2[0].data[0:3]), [492, 519, 542])

    def test_readSEISANHeadOnly(self):
        """
        Test SEISAN file reader with headonly flag.
        """
        # 1 - big endian, 32 bit
        file = os.path.join(self.path, '9701-30-1048-54S.MVO_21_1')
        st1 = readSEISAN(file, headonly=True)
        self.assertEqual(len(st1), 21)
        self.assertEqual(st1[0].stats.network, '')
        self.assertEqual(st1[0].stats.station, 'MBGA')
        self.assertEqual(st1[0].stats.location, 'J')
        self.assertEqual(st1[0].stats.channel, 'SBZ')
        self.assertEqual(st1[0].stats.starttime,
                         UTCDateTime('1997-01-30T10:48:54.040000Z'))
        self.assertEqual(st1[0].stats.endtime,
                         UTCDateTime('1997-01-30T10:49:42.902881Z'))
        self.assertAlmostEqual(st1[0].stats.sampling_rate, 75.2, 1)
        self.assertEqual(st1[0].stats.npts, 3675)
        self.assertAlmostEqual(st1[20].stats.delta, 0.0133, 4)
        self.assertEqual(list(st1[0].data), [])  # no data


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
