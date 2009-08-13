# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectogram test suite.
"""

from obspy.gse2 import tests as gse2tests
from obspy.imaging import spectrogram
import inspect
import obspy
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
        # read data
        path = os.path.dirname(inspect.getsourcefile(gse2tests))
        file = os.path.join(path, 'data', 'loc_RJOB20050831023349.z')
        g = obspy.read(file, format='GSE2')
        outfile = os.path.join(self.path, 'spectogram.png')
        spectrogram.spectroGram(g[0].data[0:1000], samp_rate=200.0, log=True,
                                outfile=outfile)


def suite():
    return unittest.makeSuite(SpectrogramTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
