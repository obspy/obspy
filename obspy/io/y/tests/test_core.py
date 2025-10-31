# -*- coding: utf-8 -*-
import warnings

from obspy.io.y.core import _is_y, _read_y


class TestCore():
    """
    Nanometrics Y file test suite.
    """
    def test_is_y_file(self, testdata):
        """
        Testing Y file format.
        """
        testfile = testdata['YAYT_BHZ_20021223.124800']
        assert _is_y(testfile)
        assert not _is_y("/path/to/slist.ascii")
        assert not _is_y("/path/to/tspair.ascii")

    def test_read_y_file(self, testdata):
        """
        Testing reading Y file format.
        """
        testfile = testdata['YAYT_BHZ_20021223.124800']
        st = _read_y(testfile)
        assert len(st) == 1
        tr = st[0]
        assert len(tr) == 18000
        assert tr.stats.sampling_rate == 100.0
        assert tr.stats.station == 'AYT'
        assert tr.stats.channel == 'BHZ'
        assert tr.stats.location == ''
        assert tr.stats.network == ''
        assert max(tr.data) == tr.stats.y.tag_series_info.max_amplitude
        assert min(tr.data) == tr.stats.y.tag_series_info.min_amplitude

    def test_ignore_non_ascii_tag_station_info(self, testdata):
        """
        Test faulty Y file containing non ASCII chars in TAG_STATION_INFO.
        """
        testfile = testdata['YAZRSPE.20100119.060433']
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            st = _read_y(testfile)
        assert len(w) == 1
        assert 'Invalid' in str(w[0])
        assert len(st) == 1
        tr = st[0]
        assert len(tr) == 16976
        assert tr.stats.sampling_rate == 50.0
        assert tr.stats.station == 'AZR'
        assert tr.stats.channel == 'E'
        assert tr.stats.location == 'SP'
        assert tr.stats.network == ''
