# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectogram test suite.
"""

from obspy import UTCDateTime, Stream, Trace
from obspy.core.util.base import ImageComparison, HAS_COMPARE_IMAGE, \
    getMatplotlibVersion
from obspy.core.util.decorator import skipIf
from obspy.imaging import spectrogram
import numpy as np
import os
import unittest


MATPLOTLIB_VERSION = getMatplotlibVersion()


class SpectrogramTestCase(unittest.TestCase):
    """
    Test cases for spectrogram plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')

    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib to old')
    def test_spectogram(self):
        """
        Create spectogram plotting examples in tests/output directory.
        """
        # Create dynamic test_files to avoid dependencies of other modules.
        # set specific seed value such that random numbers are reproduceable
        np.random.seed(815)
        head = {
            'network': 'BW', 'station': 'BGLD',
            'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
            'sampling_rate': 200.0, 'channel': 'EHE'}
        tr = Trace(data=np.random.randint(0, 1000, 824), header=head)
        st = Stream([tr])
        # 1 - using log=True
        with ImageComparison(self.path, 'spectogram_log.png') as ic:
            spectrogram.spectrogram(st[0].data, log=True, outfile=ic.name,
                                    samp_rate=st[0].stats.sampling_rate,
                                    show=False)
        # 2 - using log=False
        reltol = 1
        if MATPLOTLIB_VERSION < [1, 3, 0]:
            reltol = 3
        with ImageComparison(self.path, 'spectogram.png', reltol=reltol) as ic:
            spectrogram.spectrogram(st[0].data, log=False, outfile=ic.name,
                                    samp_rate=st[0].stats.sampling_rate,
                                    show=False)


def suite():
    return unittest.makeSuite(SpectrogramTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
