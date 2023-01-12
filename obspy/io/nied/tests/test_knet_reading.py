# -*- coding: utf-8 -*-
import io

import numpy as np

from obspy import read
from obspy.io.nied.knet import _is_knet_ascii


class TestKnetReading():
    """
    Test reading of K-NET and KiK-net ASCII format files from a file.
    """
    def test_read_knet_ascii(self, testdata):
        testfile = testdata['test.knet']
        tr = read(testfile)[0]
        tr.data *= tr.stats.calib
        tr.data -= tr.data.mean()
        max = np.abs(tr.data).max() * 100  # Maximum acc converted to gal
        np.testing.assert_array_almost_equal(max, tr.stats.knet.accmax,
                                             decimal=3)
        duration = int(tr.stats.endtime - tr.stats.starttime + 0.5)
        assert duration == int(tr.stats.knet.duration)

    def test_read_knet_ascii_from_open_files(self, testdata):
        """
        Test reading of K-NET and KiK-net ASCII format files from an open file.
        """
        testfile = testdata['test.knet']
        with open(testfile, "rb") as fh:
            tr = read(fh)[0]
            tr.data *= tr.stats.calib
            tr.data -= tr.data.mean()
            max = np.abs(tr.data).max() * 100  # Maximum acc converted to gal
            np.testing.assert_array_almost_equal(max, tr.stats.knet.accmax,
                                                 decimal=3)
            duration = int(tr.stats.endtime - tr.stats.starttime + 0.5)
            assert duration == int(tr.stats.knet.duration)

    def test_read_knet_ascii_from_bytes_io(self, testdata):
        """
        Tests that reading of K-NET and KiK-net ASCII format files from a
        BytesIO object works.
        """
        testfile = testdata['test.knet']
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
            assert duration == int(tr.stats.knet.duration)

    def test_station_name_hack(self, testdata):
        """
        Station names in K-NET and KiK-net are 6 characters long which does not
        conform with the SEED standard. Test hack to write the last 2
        characters of the station name into the location field.
        """
        testfile = testdata['test.knet']
        tr = read(testfile, convert_stnm=True)[0]
        assert tr.stats.location == '13'

    def test_is_knet_ascii(self, testdata, datapath):
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
            filename = testdata[_i]
            is_knet = _is_knet_ascii(filename)
            assert is_knet
        # Loop over non K-NET files
        for _i in non_knet_filenames:
            filename = datapath.parent / _i
            is_knet = _is_knet_ascii(filename)
            assert not is_knet
