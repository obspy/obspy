# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectogram test suite.
"""

from obspy.core import UTCDateTime, Stream, Trace
from obspy.imaging import spectrogram
import inspect
import numpy as np
import os
import unittest


class SpectrogramTestCase(unittest.TestCase):
    """
    Test cases for spectrogram plotting.
    """
    def setUp(self):
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'output')

    def tearDown(self):
        pass

    def test_Waveform(self):
        """
        Create waveform plotting examples in tests/output directory.
        """
        # Create dynamic test_files to avoid dependencies of other modules.
        # set specific seed value such that random numbers are reproduceable
        np.random.seed(815)
        header = {'network': 'BW', 'station': 'BGLD',
            'starttime': UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
            'npts': 412, 'sampling_rate': 200.0,
            'channel': 'EHE'}
        trace = Trace(data=np.random.randint(0, 1000, 412), header=header)
        stream = Stream([trace])
        outfile = os.path.join(self.path, 'spectogram.png')
        spectrogram.spectrogram(stream[0].data[0:1000], samp_rate=200.0,
                                log=True, outfile=outfile)


def suite():
    return unittest.makeSuite(SpectrogramTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
