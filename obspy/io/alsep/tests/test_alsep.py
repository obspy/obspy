# -*- coding: utf-8 -*-
import os
import unittest

from obspy.io.alsep.core import (_is_pse, _is_wtn, _is_wth,
                                 _read_pse, _read_wtn, _read_wth)


class AlsepTestCase(unittest.TestCase):

    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_is_pse(self):
        """
        Testing ALSEP PSE file format.
        """
        testfile = os.path.join(self.path, 'data', 'pse.a15.1.2.mini')
        self.assertEqual(_is_pse(testfile), True)
        testfile = os.path.join(self.path, 'data', 'wtn.1.2.mini')
        self.assertEqual(_is_pse(testfile), False)
        testfile = os.path.join(self.path, 'data', 'wth.1.5.mini')
        self.assertEqual(_is_pse(testfile), False)

    def test_is_wtn(self):
        """
        Testing ALSEP WTN file format.
        """
        testfile = os.path.join(self.path, 'data', 'pse.a15.1.2.mini')
        self.assertEqual(_is_wtn(testfile), False)
        testfile = os.path.join(self.path, 'data', 'wtn.1.2.mini')
        self.assertEqual(_is_wtn(testfile), True)
        testfile = os.path.join(self.path, 'data', 'wth.1.5.mini')
        self.assertEqual(_is_wtn(testfile), False)

    def test_is_wth(self):
        """
        Testing ALSEP WTH file format.
        """
        testfile = os.path.join(self.path, 'data', 'pse.a15.1.2.mini')
        self.assertEqual(_is_wth(testfile), False)
        testfile = os.path.join(self.path, 'data', 'wtn.1.2.mini')
        self.assertEqual(_is_wth(testfile), False)
        testfile = os.path.join(self.path, 'data', 'wth.1.5.mini')
        self.assertEqual(_is_wth(testfile), True)

    def test_read_alsep_pse_file(self):
        """
        Read ALSEP PSE file test via obspy.core.alsep._read.
        """
        testfile = os.path.join(self.path, 'data', 'pse.a15.1.2.mini')
        stream = _read_pse(testfile)
        self.assertEqual(10, len(stream.traces))

    def test_read_alsep_pse_file_with_ignore_error(self):
        """
        Read ALSEP PSE file test via obspy.core.alsep._read.
        """
        testfile = os.path.join(self.path, 'data', 'pse.a15.1.2.mini')
        stream = _read_pse(testfile, ignore_error=True)
        self.assertEqual(4654, len(stream.traces))

    def test_read_alsep_wtn_file(self):
        """
        Read ALSEP WTN file test via obspy.core.alsep._read.
        """
        testfile = os.path.join(self.path, 'data', 'wtn.1.2.mini')
        stream = _read_wtn(testfile)
        self.assertEqual(27, len(stream.traces))

    def test_read_alsep_wth_file(self):
        """
        Read ALSEP WTH file test via obspy.core.alsep._read.
        """
        testfile = os.path.join(self.path, 'data', 'wth.1.5.mini')
        stream = _read_wth(testfile)
        self.assertEqual(12, len(stream.traces))

    def test_single_header_wtn(self):
        """
        Read single header WTN file test
        """
        testfile = os.path.join(self.path, 'data', 'wtn.6.30.mini')
        stream = _read_wtn(testfile)
        self.assertEqual(18, len(stream.traces))

    def test_single_header_wth(self):
        """
        Read single header WTH file test
        """
        testfile = os.path.join(self.path, 'data', 'wth.5.6.mini')
        stream = _read_wth(testfile)
        st_geophone1 = stream.select(id='XA.S17..GP1')
        self.assertEqual(3, len(st_geophone1))

    def test_pse_new_format(self):
        """
        Read PSE new format which does not have Apollo 12 SPZ
        """
        testfile = os.path.join(self.path, 'data', 'pse.a12.6.117.mini')
        stream = _read_pse(testfile)
        st_spz = stream.select(id='XA.S12..SPZ')
        self.assertEqual(0, len(st_spz))

    def test_frame_loss(self):
        """
        Check frame with many time skipping
        """
        testfile = os.path.join(self.path, 'data', 'pse.a14.4.171.mini')
        stream = _read_pse(testfile)
        st_lpx = stream.select(id='XA.S14..LPX')
        self.assertEqual(1, len(st_lpx))

    def test_pse_read_year_option(self):
        """
        Read pse data with year option to overwrite year
        """
        testfile = os.path.join(self.path, 'data', 'pse.a12.10.91.mini')
        stream = _read_pse(testfile)
        st_lpx = stream.select(id='XA.S12..LPX')
        self.assertEqual(1976, st_lpx[0].times("utcdatetime")[0].year)

        stream = _read_pse(testfile, year=1975)
        st_lpx = stream.select(id='XA.S12..LPX')
        self.assertEqual(1975, st_lpx[0].times("utcdatetime")[0].year)
