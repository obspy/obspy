#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.io.dmx.core test suite.
"""
import numpy as np

from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.io.dmx.core import _read_dmx


class TestCore():
    """
    Test cases for dmx core interface
    """
    def test_read_via_obspy(self, testdata):
        """
        Read files via obspy.core.stream.read function.
        """
        filename = testdata['131114_090600.dmx']
        # 1
        st = read(filename)
        st.verify()
        assert len(st) == 2
        assert st[0].stats.starttime == UTCDateTime(2013, 11, 14, 9, 6)
        assert st[0].stats.endtime == \
            UTCDateTime(2013, 11, 14, 9, 6, 59, 990000)
        assert "dmx" in st[0].stats
        assert len(st[0]) == 6000
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'Z'
        assert st[0].id == 'ETNA.EMFO..Z'

    def test_read_via_module(self, testdata):
        """
        Read files via obspy.io.mdx.core._read_dmx function directly.
        """
        filename = testdata['131114_090600.dmx']
        # 1
        st = _read_dmx(filename)
        st.verify()
        assert len(st) == 2
        assert st[0].stats.starttime == UTCDateTime(2013, 11, 14, 9, 6)
        assert st[0].stats.endtime == \
            UTCDateTime(2013, 11, 14, 9, 6, 59, 990000)
        assert "dmx" in st[0].stats
        assert len(st[0]) == 6000
        assert round(abs(st[0].stats.sampling_rate-100.0), 7) == 0
        assert st[0].stats.channel == 'Z'
        assert st[0].id == 'ETNA.EMFO..Z'

    def test_read_with_station(self, testdata):
        """
        Read files and passing a station keyword argument.
        """
        filename = testdata['131114_090600.dmx']
        # 1
        st = read(filename, station='EMPL')
        st.verify()

        assert len(st) == 1
        assert st[0].id == "ETNA.EMPL..Z"
        for tr in st:
            assert tr.stats.station == "EMPL"

    def test_check_data_content_sum(self, testdata):
        """
        Read files and passing a station keyword argument.
        """
        filename = testdata['131114_090600.dmx']
        # 1
        st = read(filename)
        st.verify()
        assert st[0].data.sum() == -90928.0
        np.testing.assert_array_equal(
            st[0].data[:5], [148., 238., 133., 9., -92.])

        assert st[1].data.sum() == 3003120.0
        np.testing.assert_array_equal(
            st[1].data[:5], [537., 721., 844., 924., 977.])
