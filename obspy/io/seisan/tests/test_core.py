# -*- coding: utf-8 -*-
"""
The seisan.core test suite.
"""
import numpy as np

from obspy.core import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.seisan.core import _get_version, _is_seisan, _read_seisan


class TestCore():
    """
    Test cases for SEISAN core interfaces.
    """
    def test_get_version(self, testdata):
        """
        Tests resulting version strings of SEISAN file.
        """
        # 1 - big endian, 32 bit, version 7
        fn = testdata['1996-06-03-1917-52S.TEST__002']
        with open(fn, 'rb') as fp:
            data = fp.read(80 * 12)
        assert _get_version(data) == ('>', 32, 7)

        # 2 - little endian, 32 bit, version 7
        fn = testdata['2001-01-13-1742-24S.KONO__004']
        with open(fn, 'rb') as fp:
            data = fp.read(80 * 12)
        assert _get_version(data) == ('<', 32, 7)

        # 3 - little endian, 32 bit, version 6
        fn = testdata['2005-07-23-1452-04S.CER___030']
        with open(fn, 'rb') as fp:
            data = fp.read(80 * 12)
        assert _get_version(data) == ('<', 32, 6)

    def test_is_seisan(self, testdata):
        """
        Tests SEISAN file check.
        """
        # 1 - big endian, 32 bit, version 7
        fn = testdata['1996-06-03-1917-52S.TEST__002']
        assert _is_seisan(fn)

        # 2 - little endian, 32 bit, version 7
        fn = testdata['2001-01-13-1742-24S.KONO__004']
        assert _is_seisan(fn)

        # 3 - little endian, 32 bit, version 6
        fn = testdata['2005-07-23-1452-04S.CER___030']
        assert _is_seisan(fn)

    def test_read_seisan(self, testdata):
        """
        Test SEISAN file reader.
        """
        # 1 - big endian, 32 bit, version 7
        fn = testdata['9701-30-1048-54S.MVO_21_1']
        st = _read_seisan(fn)
        st.verify()
        assert len(st) == 21
        assert st[20].stats.network == ''
        assert st[20].stats.station == 'MBGB'
        assert st[20].stats.location == 'J'
        assert st[20].stats.channel == 'SBE'
        assert st[20].stats.starttime == \
            UTCDateTime('1997-01-30T10:48:54.040000Z')
        assert st[20].stats.endtime == \
            UTCDateTime('1997-01-30T10:49:42.902881Z')
        assert round(abs(st[20].stats.sampling_rate-75.2), 1) == 0
        assert st[20].stats.npts == 3675
        assert round(abs(st[20].stats.delta-0.0133), 4) == 0
        datafile = testdata['9701-30-1048-54S.MVO_21_1.ascii']
        # compare with ASCII values of trace (extracted ASCII file contains
        # less values than the original Seisan file!)
        assert st[20].stats.npts == 3675
        assert list(st[20].data[1:3666]) == \
            np.loadtxt(datafile, dtype=np.int32).tolist()

        # 2 - little endian, 32 bit, version 7
        fn = testdata['2001-01-13-1742-24S.KONO__004']
        st = _read_seisan(fn)
        st.verify()
        assert len(st) == 4
        assert st[0].stats.npts == 6000
        assert list(st[0].data[0:5]) == [464, 492, 519, 542, 565]

        # 3 - little endian, 32 bit, version 6, 1 channel
        fn = testdata['D1360930.203']
        st = _read_seisan(fn)
        st.verify()
        assert len(st) == 1
        assert st[0].stats.npts == 12000
        assert list(st[0].data[0:5]) == [24, 64, 139, 123, 99]

        # 4 - little endian, 32 bit, version 6, 3 channels
        fn = testdata['2005-07-23-1452-04S.CER___030']
        st = _read_seisan(fn)
        st.verify()
        assert len(st) == 3
        assert st[0].stats.npts == 10650
        assert list(st[0].data[0:5]) == [7520, 7484, 7482, 7480, 7478]

    def test_read_seisan_head_only(self, testdata):
        """
        Test SEISAN file reader with headonly flag.
        """
        # 1 - big endian, 32 bit, version 7
        fn = testdata['9701-30-1048-54S.MVO_21_1']
        st = _read_seisan(fn, headonly=True)
        assert len(st) == 21
        assert st[0].stats.network == ''
        assert st[0].stats.station == 'MBGA'
        assert st[0].stats.location == 'J'
        assert st[0].stats.channel == 'SBZ'
        assert st[0].stats.starttime == \
            UTCDateTime('1997-01-30T10:48:54.040000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('1997-01-30T10:49:42.902881Z')
        assert round(abs(st[0].stats.sampling_rate-75.2), 1) == 0
        assert st[0].stats.npts == 3675
        assert round(abs(st[20].stats.delta-0.0133), 4) == 0
        assert list(st[0].data) == []  # no data

        # 2 - little endian, 32 bit, version 7
        fn = testdata['2001-01-13-1742-24S.KONO__004']
        st = _read_seisan(fn, headonly=True)
        assert len(st) == 4
        assert st[0].stats.network == ''
        assert st[0].stats.station == 'KONO'
        assert st[0].stats.location == '0'
        assert st[0].stats.channel == 'B0Z'
        assert st[0].stats.starttime == \
            UTCDateTime(2001, 1, 13, 17, 45, 1, 999000)
        assert st[0].stats.endtime == \
            UTCDateTime(2001, 1, 13, 17, 50, 1, 949000)
        assert st[0].stats.sampling_rate == 20.0
        assert st[0].stats.npts == 6000
        assert list(st[0].data) == []  # no data

        # 3 - little endian, 32 bit, version 6, 1 channel
        fn = testdata['D1360930.203']
        st = _read_seisan(fn, headonly=True)
        assert len(st) == 1
        assert st[0].stats.network == ''
        assert st[0].stats.station == 'mart'
        assert st[0].stats.location == '1'
        assert st[0].stats.channel == 'cp'
        assert st[0].stats.starttime == UTCDateTime(2017, 7, 22, 9, 30)
        assert st[0].stats.endtime == \
            UTCDateTime(2017, 7, 22, 9, 31, 59, 990000)
        assert st[0].stats.sampling_rate == 100.0
        assert st[0].stats.npts == 12000
        assert list(st[0].data) == []

        # 4 - little endian, 32 bit, version 6, 3 channels
        fn = testdata['2005-07-23-1452-04S.CER___030']
        st = _read_seisan(fn, headonly=True)
        assert len(st) == 3
        assert st[0].stats.channel == 'BHZ'
        assert st[1].stats.channel == 'BHN'
        assert st[2].stats.channel == 'BHE'
        for i in range(0, 3):
            assert st[i].stats.network == ''
            assert st[i].stats.station == 'CER'
            assert st[i].stats.location == ''
            assert st[i].stats.starttime == \
                UTCDateTime('2005-07-23T14:52:04.000000Z')
            assert st[i].stats.endtime == \
                UTCDateTime('2005-07-23T14:53:14.993333Z')
            assert st[i].stats.sampling_rate == 150.0
            assert st[i].stats.npts == 10650
            assert list(st[i].data) == []

    def test_read_obspy(self, testdata):
        """
        Test ObsPy read function and compare against given MiniSEED files.
        """
        # 1 - little endian, 32 bit, version 7
        st1 = read(testdata['2011-09-06-1311-36S.A1032_001BH_Z'])
        st2 = read(testdata['2011-09-06-1311-36S.A1032_001BH_Z.mseed'])
        assert len(st1) == len(st2)
        assert np.allclose(st1[0].data, st2[0].data)

        # 2 - little endian, 32 bit, version 6, 1 channel
        st1 = read(testdata['D1360930.203'])
        st2 = read(testdata['D1360930.203.mseed'])
        assert len(st1) == len(st2)
        assert np.allclose(st1[0].data, st2[0].data)

        # 3 - little endian, 32 bit, version 6, 3 channels
        st1 = read(testdata['2005-07-23-1452-04S.CER___030'])
        st2 = read(testdata['2005-07-23-1452-04S.CER___030.mseed'])
        assert len(st1) == len(st2)
        assert np.allclose(st1[0].data, st2[0].data)
        assert np.allclose(st1[1].data, st2[1].data)
        assert np.allclose(st1[2].data, st2[2].data)
