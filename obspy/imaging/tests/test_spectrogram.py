# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectrogram test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import UTCDateTime, Stream, Trace
from obspy.core.util.testing import ImageComparison
from obspy.imaging import spectrogram
import numpy as np
import os
import unittest
import warnings


class SpectrogramTestCase(unittest.TestCase):
    """
    Test cases for spectrogram plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')

    def test_spectrogram(self):
        """
        Create spectrogram plotting examples in tests/output directory.
        """
        # Create dynamic test_files to avoid dependencies of other modules.
        # set specific seed value such that random numbers are reproducible
        np.random.seed(815)
        head = {
            'network': 'BW', 'station': 'BGLD',
            'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
            'sampling_rate': 200.0, 'channel': 'EHE'}
        tr = Trace(data=np.random.randint(0, 1000, 824), header=head)
        st = Stream([tr])
        # 1 - using log=True
        with ImageComparison(self.path, 'spectrogram_log.png') as ic:
            with warnings.catch_warnings(record=True):
                warnings.resetwarnings()
                np_err = np.seterr(all="warn")
                spectrogram.spectrogram(st[0].data, log=True, outfile=ic.name,
                                        samp_rate=st[0].stats.sampling_rate,
                                        show=False)
                np.seterr(**np_err)
        # 2 - using log=False
        with ImageComparison(self.path, 'spectrogram.png') as ic:
            spectrogram.spectrogram(st[0].data, log=False, outfile=ic.name,
                                    samp_rate=st[0].stats.sampling_rate,
                                    show=False)


def suite():
    return unittest.makeSuite(SpectrogramTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
