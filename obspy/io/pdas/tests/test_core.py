# -*- coding: utf-8 -*-
import numpy as np

from obspy import Stream, UTCDateTime
from obspy.io.pdas import _is_pdas, _read_pdas


class TestPDAS():
    """
    Test suite for pdas.
    """
    def test_read_pdas(self, testdata):
        """
        Tests the _read_pdas function.
        """
        st = _read_pdas(testdata['p1246001.108'])
        assert isinstance(st, Stream)
        assert len(st) == 1
        tr = st[0]
        expected = [('COMMENT', 'GAINRANGED'),
                    ('DATASET', 'P1246001108'),
                    ('FILE_TYPE', 'LONG'),
                    ('HORZ_UNITS', 'Sec'),
                    ('SIGNAL', 'Channel1'),
                    ('VERSION', 'next'),
                    ('VERT_UNITS', 'Counts')]
        assert sorted(tr.stats.pop("pdas").items()) == expected
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
        assert sorted(tr.stats.items()) == expected
        expected = np.array([895, 867, 747, 591, 359, -129, -185, 3, 115, 243],
                            dtype=np.int16)
        np.testing.assert_array_equal(tr.data[:10], expected)

    def test_is_pdas(self, testdata):
        """
        Tests the _read_pdas function.
        """
        assert _is_pdas(testdata['p1246001.108'])
