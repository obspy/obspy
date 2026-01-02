#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.win.core test suite.
"""
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.win.core import _read_win


class TestCore():
    """
    Test cases for win core interface
    """
    def test_read_via_obspy(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['10030302.00']
        # 1
        st = read(filename)
        st.verify()
        st.sort(keys=['channel'])
        assert len(st) == 2
        assert st[0].stats.starttime == \
            UTCDateTime('2010-03-03T02:00:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2010-03-03T02:00:59.990000Z')
        assert st[0].stats.starttime == \
            UTCDateTime('2010-03-03T02:00:00.000000Z')
        assert len(st[0]) == 6000
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'a100'

    def test_read_via_module(self, testdata):
        """
        Read files via obspy.io.win.core._read_win function.
        """
        filename = testdata['10030302.00']
        # 1
        st = _read_win(filename)
        st.verify()
        st.sort(keys=['channel'])
        assert len(st) == 2
        assert st[0].stats.starttime == \
            UTCDateTime('2010-03-03T02:00:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2010-03-03T02:00:59.990000Z')
        assert st[0].stats.starttime == \
            UTCDateTime('2010-03-03T02:00:00.000000Z')
        assert len(st[0]) == 6000
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'a100'

    def test_read_05_byte_data(self, testdata):
        """
        Read a file included 0.5 byte datawide
        """
        filename = testdata['1070533011_1701260003.win']

        st = _read_win(filename)
        st.sort(keys=['channel'])
        assert len(st[0]) == 6000
        assert len(st[1]) == 6000
        assert len(st[2]) == 6000
        assert st[0].stats.starttime == \
            UTCDateTime('2017-01-26T00:03:00.000000Z')
        assert st[0].stats.endtime == \
            UTCDateTime('2017-01-26T00:03:59.990000Z')
        assert st[1].stats.starttime == \
            UTCDateTime('2017-01-26T00:03:00.000000Z')
        assert st[1].stats.endtime == \
            UTCDateTime('2017-01-26T00:03:59.990000Z')
        assert st[2].stats.starttime == \
            UTCDateTime('2017-01-26T00:03:00.000000Z')
        assert st[2].stats.endtime == \
            UTCDateTime('2017-01-26T00:03:59.990000Z')

    def test_read_1kHz_data(self, testdata):
        """ Reads a file with 1kHz sampling rate
        Was an issue with header parsing, see #3641"""
        filename = testdata['25112616.10']
        st = _read_win(filename)
        assert st[0].stats.sampling_rate == 1000.0

    def test_read_24bit_data(self, testdata):
        """ Reads a file with 24bit data """
        filename = testdata['25112618.24bits']
        st = read(filename)
        assert st[0].stats.sampling_rate == 200.0

        expected_stds = {
            "...0000": 62491.2874698,
            "...0001": 44598.5112877,
            "...0002": 64268.6162495,
            "...0003": 44025.9400495,
            "...0004": 63977.9475183,
            "...0005": 44582.1592993}

        for tr in st:
            assert round(expected_stds[tr.id], 4) == round(tr.data.std(), 4)
