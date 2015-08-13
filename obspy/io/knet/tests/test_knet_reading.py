# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import unittest
from obspy import read
from obspy.io.knet.core import _is_knet_ascii
import numpy as np

class KnetReadingTestCase(unittest.TestCase):
    """
    Test everything related to the reading of K-NET ASCII format files.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_read_knet_ascii(self):
        testfile = os.path.join(self.path, 'data', 'AKT0139608110312.EW')
        tr = read(testfile)[0]
        tr.data *= tr.stats.calib
        tr.data -= tr.data.mean()
        max = np.abs(tr.data).max() * 100  # Maximum acc converted to gal
        np.testing.assert_array_almost_equal(max, tr.stats.knet.accmax,
                                             decimal=3)
        duration = int(tr.stats.endtime - tr.stats.starttime + 0.5)
        self.assertEqual(duration, int(tr.stats.knet.duration))

    def test_is_knet_ascii(self):
        """
        This tests the _is_knet_ascii method by just validating that each file in
        the data directory is a K-NET ascii file and each file in the working
        directory is not.

        The filenames are hard coded so the test will not fail with future
        changes in the structure of the package.
        """
        # K-NET file names.
        knet_filenames = ['AKT0139608110312.EW']

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


def suite():
    return unittest.makeSuite(KnetReadingTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
