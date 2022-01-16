# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectrogram test suite.
"""
import os
import warnings

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.imaging import spectrogram


class TestSpectrogram:
    """
    Test cases for spectrogram plotting.
    """
    path = os.path.join(os.path.dirname(__file__), 'images')

    def test_spectrogram(self, image_path):
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
        image_path_1 = image_path.parent / 'spectrogram1.png'
        with warnings.catch_warnings(record=True):
            warnings.resetwarnings()
            with np.errstate(all='warn'):
                spectrogram.spectrogram(
                    st[0].data, log=True, outfile=image_path_1,
                    samp_rate=st[0].stats.sampling_rate, show=False)
        # 2 - using log=False
        image_path_2 = image_path.parent / 'spectrogram2.png'
        spectrogram.spectrogram(st[0].data, log=False, outfile=image_path_2,
                                samp_rate=st[0].stats.sampling_rate,
                                show=False)
