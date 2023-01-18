# -*- coding: utf-8 -*-
from obspy.io.alsep.core import (_is_pse, _is_wtn, _is_wth,
                                 _read_pse, _read_wtn, _read_wth)


class TestAlsep():
    def test_is_pse(self, testdata):
        """
        Testing ALSEP PSE file format.
        """
        testfile = testdata['pse.a15.1.2.mini']
        assert _is_pse(testfile)
        testfile = testdata['wtn.1.2.mini']
        assert not _is_pse(testfile)
        testfile = testdata['wth.1.5.mini']
        assert not _is_pse(testfile)

    def test_is_wtn(self, testdata):
        """
        Testing ALSEP WTN file format.
        """
        testfile = testdata['pse.a15.1.2.mini']
        assert not _is_wtn(testfile)
        testfile = testdata['wtn.1.2.mini']
        assert _is_wtn(testfile)
        testfile = testdata['wth.1.5.mini']
        assert not _is_wtn(testfile)

    def test_is_wth(self, testdata):
        """
        Testing ALSEP WTH file format.
        """
        testfile = testdata['pse.a15.1.2.mini']
        assert not _is_wth(testfile)
        testfile = testdata['wtn.1.2.mini']
        assert not _is_wth(testfile)
        testfile = testdata['wth.1.5.mini']
        assert _is_wth(testfile)

    def test_read_alsep_pse_file(self, testdata):
        """
        Read ALSEP PSE file test via obspy.core.alsep._read.
        """
        testfile = testdata['pse.a15.1.2.mini']
        stream = _read_pse(testfile)
        assert 10 == len(stream.traces)

    def test_read_alsep_pse_file_with_ignore_error(self, testdata):
        """
        Read ALSEP PSE file test via obspy.core.alsep._read.
        """
        testfile = testdata['pse.a15.1.2.mini']
        stream = _read_pse(testfile, ignore_error=True)
        assert 4654 == len(stream.traces)

    def test_read_alsep_wtn_file(self, testdata):
        """
        Read ALSEP WTN file test via obspy.core.alsep._read.
        """
        testfile = testdata['wtn.1.2.mini']
        stream = _read_wtn(testfile)
        assert 27 == len(stream.traces)

    def test_read_alsep_wth_file(self, testdata):
        """
        Read ALSEP WTH file test via obspy.core.alsep._read.
        """
        testfile = testdata['wth.1.5.mini']
        stream = _read_wth(testfile)
        assert 12 == len(stream.traces)

    def test_single_header_wtn(self, testdata):
        """
        Read single header WTN file test
        """
        testfile = testdata['wtn.6.30.mini']
        stream = _read_wtn(testfile)
        assert 18 == len(stream.traces)

    def test_single_header_wth(self, testdata):
        """
        Read single header WTH file test
        """
        testfile = testdata['wth.5.6.mini']
        stream = _read_wth(testfile)
        st_geophone1 = stream.select(id='XA.S17..GP1')
        assert 3 == len(st_geophone1)

    def test_pse_new_format(self, testdata):
        """
        Read PSE new format which does not have Apollo 12 SPZ
        """
        testfile = testdata['pse.a12.6.117.mini']
        stream = _read_pse(testfile)
        st_spz = stream.select(id='XA.S12..SPZ')
        assert 0 == len(st_spz)

    def test_frame_loss(self, testdata):
        """
        Check frame with many time skipping
        """
        testfile = testdata['pse.a14.4.171.mini']
        stream = _read_pse(testfile)
        st_lpx = stream.select(id='XA.S14..LPX')
        assert 1 == len(st_lpx)

    def test_pse_read_year_option(self, testdata):
        """
        Read pse data with year option to overwrite year
        """
        testfile = testdata['pse.a12.10.91.mini']
        stream = _read_pse(testfile)
        st_lpx = stream.select(id='XA.S12..LPX')
        assert 1976 == st_lpx[0].times("utcdatetime")[0].year

        stream = _read_pse(testfile, year=1975)
        st_lpx = stream.select(id='XA.S12..LPX')
        assert 1975 == st_lpx[0].times("utcdatetime")[0].year
