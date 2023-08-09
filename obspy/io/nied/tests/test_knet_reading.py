# -*- coding: utf-8 -*-
import os
import io
import unittest

import numpy as np

from obspy import read
from obspy.io.nied.knet import _is_knet_ascii


class KnetReadingTestCase(unittest.TestCase):
    """
    Test reading of K-NET and KiK-net ASCII format files from a file.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_read_knet_ascii(self):
        testfile = os.path.join(self.path, 'data', 'test.knet')
        tr = read(testfile)[0]
        tr.data *= tr.stats.calib
        tr.data -= tr.data.mean()
        max = np.abs(tr.data).max() * 100  # Maximum acc converted to gal
        np.testing.assert_array_almost_equal(max, tr.stats.knet.accmax,
                                             decimal=3)
        duration = int(tr.stats.endtime - tr.stats.starttime + 0.5)
        self.assertEqual(duration, int(tr.stats.knet.duration))

    def test_read_knet_ascii_from_open_files(self):
        """
        Test reading of K-NET and KiK-net ASCII format files from an open file.
        """
        testfile = os.path.join(self.path, 'data', 'test.knet')
        with open(testfile, "rb") as fh:
            tr = read(fh)[0]
            tr.data *= tr.stats.calib
            tr.data -= tr.data.mean()
            max = np.abs(tr.data).max() * 100  # Maximum acc converted to gal
            np.testing.assert_array_almost_equal(max, tr.stats.knet.accmax,
                                                 decimal=3)
            duration = int(tr.stats.endtime - tr.stats.starttime + 0.5)
            self.assertEqual(duration, int(tr.stats.knet.duration))

    def test_read_knet_ascii_from_bytes_io(self):
        """
        Tests that reading of K-NET and KiK-net ASCII format files from a
        BytesIO object works.
        """
        testfile = os.path.join(self.path, 'data', 'test.knet')
        with open(testfile, "rb") as fh:
            buf = io.BytesIO(fh.read())

        with buf:
            tr = read(buf)[0]
            tr.data *= tr.stats.calib
            tr.data -= tr.data.mean()
            max = np.abs(tr.data).max() * 100  # Maximum acc converted to gal
            np.testing.assert_array_almost_equal(max, tr.stats.knet.accmax,
                                                 decimal=3)
            duration = int(tr.stats.endtime - tr.stats.starttime + 0.5)
            self.assertEqual(duration, int(tr.stats.knet.duration))

    def test_station_name_hack(self):
        """
        Station names in K-NET and KiK-net are 6 characters long which does not
        conform with the SEED standard. Test hack to write the last 2
        characters of the station name into the location field.
        """
        testfile = os.path.join(self.path, 'data', 'test.knet')
        tr = read(testfile, convert_stnm=True)[0]
        self.assertEqual(tr.stats.location, '13')

    def test_is_knet_ascii(self):
        """
        This tests the _is_knet_ascii method by just validating that each file
        in the data directory is a K-NET ascii file and each file in the
        working directory is not.

        The filenames are hard coded so the test will not fail with future
        changes in the structure of the package.
        """
        # K-NET file names.
        knet_filenames = ['test.knet']

        # Non K-NET file names.
        non_knet_filenames = ['test_knet_reading.py',
                              '__init__.py']
        # Loop over K-NET files
        for _i in knet_filenames:
            filename = os.path.join(self.path, 'data', _i)
            is_knet = _is_knet_ascii(filename)
            self.assertTrue(is_knet)
        # Loop over non K-NET files
        for _i in non_knet_filenames:
            filename = os.path.join(self.path, _i)
            is_knet = _is_knet_ascii(filename)
            self.assertFalse(is_knet)
