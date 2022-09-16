# -*- coding: utf-8 -*-
"""
The seisan.core test suite.
"""
import os
import unittest

import numpy as np

from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.seisan.core import _get_version, _is_seisan, _read_seisan


class CoreTestCase(unittest.TestCase):
    """
    Test cases for SEISAN core interfaces.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_get_version(self):
        """
        Tests resulting version strings of SEISAN file.
        """
        # 1 - big endian, 32 bit, version 7
        fn = os.path.join(self.path, '1996-06-03-1917-52S.TEST__002')
        with open(fn, 'rb') as fp:
            data = fp.read(80 * 12)
        self.assertEqual(_get_version(data), ('>', 32, 7))

        # 2 - little endian, 32 bit, version 7
        fn = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        with open(fn, 'rb') as fp:
            data = fp.read(80 * 12)
        self.assertEqual(_get_version(data), ('<', 32, 7))

        # 3 - little endian, 32 bit, version 6
        fn = os.path.join(self.path, '2005-07-23-1452-04S.CER___030')
        with open(fn, 'rb') as fp:
            data = fp.read(80 * 12)
        self.assertEqual(_get_version(data), ('<', 32, 6))

    def test_is_seisan(self):
        """
        Tests SEISAN file check.
        """
        # 1 - big endian, 32 bit, version 7
        fn = os.path.join(self.path, '1996-06-03-1917-52S.TEST__002')
        self.assertTrue(_is_seisan(fn))

        # 2 - little endian, 32 bit, version 7
        fn = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        self.assertTrue(_is_seisan(fn))

        # 3 - little endian, 32 bit, version 6
        fn = os.path.join(self.path, '2005-07-23-1452-04S.CER___030')
        self.assertTrue(_is_seisan(fn))

    def test_read_seisan(self):
        """
        Test SEISAN file reader.
        """
        # 1 - big endian, 32 bit, version 7
        fn = os.path.join(self.path, '9701-30-1048-54S.MVO_21_1')
        st = _read_seisan(fn)
        st.verify()
        self.assertEqual(len(st), 21)
        self.assertEqual(st[20].stats.network, '')
        self.assertEqual(st[20].stats.station, 'MBGB')
        self.assertEqual(st[20].stats.location, 'J')
        self.assertEqual(st[20].stats.channel, 'SBE')
        self.assertEqual(st[20].stats.starttime,
                         UTCDateTime('1997-01-30T10:48:54.040000Z'))
        self.assertEqual(st[20].stats.endtime,
                         UTCDateTime('1997-01-30T10:49:42.902881Z'))
        self.assertAlmostEqual(st[20].stats.sampling_rate, 75.2, 1)
        self.assertEqual(st[20].stats.npts, 3675)
        self.assertAlmostEqual(st[20].stats.delta, 0.0133, 4)
        datafile = os.path.join(self.path, '9701-30-1048-54S.MVO_21_1.ascii')
        # compare with ASCII values of trace (extracted ASCII file contains
        # less values than the original Seisan file!)
        self.assertEqual(st[20].stats.npts, 3675)
        self.assertEqual(list(st[20].data[1:3666]),
                         np.loadtxt(datafile, dtype=np.int32).tolist())

        # 2 - little endian, 32 bit, version 7
        fn = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        st = _read_seisan(fn)
        st.verify()
        self.assertEqual(len(st), 4)
        self.assertEqual(st[0].stats.npts, 6000)
        self.assertEqual(list(st[0].data[0:5]), [464, 492, 519, 542, 565])

        # 3 - little endian, 32 bit, version 6, 1 channel
        fn = os.path.join(self.path, 'D1360930.203')
        st = _read_seisan(fn)
        st.verify()
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.npts, 12000)
        self.assertEqual(list(st[0].data[0:5]),
                         [24, 64, 139, 123, 99])

        # 4 - little endian, 32 bit, version 6, 3 channels
        fn = os.path.join(self.path, '2005-07-23-1452-04S.CER___030')
        st = _read_seisan(fn)
        st.verify()
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.npts, 10650)
        self.assertEqual(list(st[0].data[0:5]),
                         [7520, 7484, 7482, 7480, 7478])

    def test_read_seisan_head_only(self):
        """
        Test SEISAN file reader with headonly flag.
        """
        # 1 - big endian, 32 bit, version 7
        fn = os.path.join(self.path, '9701-30-1048-54S.MVO_21_1')
        st = _read_seisan(fn, headonly=True)
        self.assertEqual(len(st), 21)
        self.assertEqual(st[0].stats.network, '')
        self.assertEqual(st[0].stats.station, 'MBGA')
        self.assertEqual(st[0].stats.location, 'J')
        self.assertEqual(st[0].stats.channel, 'SBZ')
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime('1997-01-30T10:48:54.040000Z'))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime('1997-01-30T10:49:42.902881Z'))
        self.assertAlmostEqual(st[0].stats.sampling_rate, 75.2, 1)
        self.assertEqual(st[0].stats.npts, 3675)
        self.assertAlmostEqual(st[20].stats.delta, 0.0133, 4)
        self.assertEqual(list(st[0].data), [])  # no data

        # 2 - little endian, 32 bit, version 7
        fn = os.path.join(self.path, '2001-01-13-1742-24S.KONO__004')
        st = _read_seisan(fn, headonly=True)
        self.assertEqual(len(st), 4)
        self.assertEqual(st[0].stats.network, '')
        self.assertEqual(st[0].stats.station, 'KONO')
        self.assertEqual(st[0].stats.location, '0')
        self.assertEqual(st[0].stats.channel, 'B0Z')
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime(2001, 1, 13, 17, 45, 1, 999000))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime(2001, 1, 13, 17, 50, 1, 949000))
        self.assertEqual(st[0].stats.sampling_rate, 20.0)
        self.assertEqual(st[0].stats.npts, 6000)
        self.assertEqual(list(st[0].data), [])  # no data

        # 3 - little endian, 32 bit, version 6, 1 channel
        fn = os.path.join(self.path, 'D1360930.203')
        st = _read_seisan(fn, headonly=True)
        self.assertEqual(len(st), 1)
        self.assertEqual(st[0].stats.network, '')
        self.assertEqual(st[0].stats.station, 'mart')
        self.assertEqual(st[0].stats.location, '1')
        self.assertEqual(st[0].stats.channel, 'cp')
        self.assertEqual(st[0].stats.starttime,
                         UTCDateTime(2017, 7, 22, 9, 30))
        self.assertEqual(st[0].stats.endtime,
                         UTCDateTime(2017, 7, 22, 9, 31, 59, 990000))
        self.assertEqual(st[0].stats.sampling_rate, 100.0)
        self.assertEqual(st[0].stats.npts, 12000)
        self.assertEqual(list(st[0].data), [])

        # 4 - little endian, 32 bit, version 6, 3 channels
        fn = os.path.join(self.path, '2005-07-23-1452-04S.CER___030')
        st = _read_seisan(fn, headonly=True)
        self.assertEqual(len(st), 3)
        self.assertEqual(st[0].stats.channel, 'BHZ')
        self.assertEqual(st[1].stats.channel, 'BHN')
        self.assertEqual(st[2].stats.channel, 'BHE')
        for i in range(0, 3):
            self.assertEqual(st[i].stats.network, '')
            self.assertEqual(st[i].stats.station, 'CER')
            self.assertEqual(st[i].stats.location, '')
            self.assertEqual(st[i].stats.starttime,
                             UTCDateTime('2005-07-23T14:52:04.000000Z'))
            self.assertEqual(st[i].stats.endtime,
                             UTCDateTime('2005-07-23T14:53:14.993333Z'))
            self.assertEqual(st[i].stats.sampling_rate, 150.0)
            self.assertEqual(st[i].stats.npts, 10650)
            self.assertEqual(list(st[i].data), [])

    def test_read_obspy(self):
        """
        Test ObsPy read function and compare against given MiniSEED files.
        """
        # 1 - little endian, 32 bit, version 7
        st1 = read(os.path.join(self.path,
                                '2011-09-06-1311-36S.A1032_001BH_Z'))
        st2 = read(os.path.join(self.path,
                                '2011-09-06-1311-36S.A1032_001BH_Z.mseed'))
        self.assertEqual(len(st1), len(st2))
        self.assertTrue(np.allclose(st1[0].data, st2[0].data))

        # 2 - little endian, 32 bit, version 6, 1 channel
        st1 = read(os.path.join(self.path, 'D1360930.203'))
        st2 = read(os.path.join(self.path, 'D1360930.203.mseed'))
        self.assertEqual(len(st1), len(st2))
        self.assertTrue(np.allclose(st1[0].data, st2[0].data))

        # 3 - little endian, 32 bit, version 6, 3 channels
        st1 = read(os.path.join(self.path, '2005-07-23-1452-04S.CER___030'))
        st2 = read(os.path.join(self.path,
                                '2005-07-23-1452-04S.CER___030.mseed'))
        self.assertEqual(len(st1), len(st2))
        self.assertTrue(np.allclose(st1[0].data, st2[0].data))
        self.assertTrue(np.allclose(st1[1].data, st2[1].data))
        self.assertTrue(np.allclose(st1[2].data, st2[2].data))
