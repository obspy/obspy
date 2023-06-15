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
