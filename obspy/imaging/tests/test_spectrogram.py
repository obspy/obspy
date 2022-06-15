# -*- coding: utf-8 -*-
"""
The obspy.imaging.spectrogram test suite.
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pytest

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

    def test_spectrogram_defaults(self):
        """
        Make sure input at varying sampling rates and data lengths does not
        suffer from defaults (wlen etc) being calculated to nonsensical values
        resulting in errors raised from the underlying mlab spectrogram

        wlen (in s) used to be calculated as (samp_rate / 100) which makes no
        sense, it results in very large window sizes for high sampling rates
        and in window lengths that dont even include a two samples for low
        sampling rates like 1 Hz.
        """
        # input of 30 minutes sampled at 1 Hz should definitely be able to have
        # a sensible default for window length, but it used to fail
        data = np.ones(1800)
        spectrogram.spectrogram(data, samp_rate=1, show=False)
        plt.close('all')

    def test_spectrogram_nice_error_messages(self):
        """
        Make sure to have some nice messages on weird input
        """
        # Only 20 samples at 1 Hz and our default tries to window 128 samples,
        # so we dont even have enough data for a single window. This used to
        # fail inside mlab specgram with ugly error message about the input
        data = np.ones(20)
        msg = ('Input signal too short (20 samples, window length 6.4 '
               'seconds, nfft 128 samples, sampling rate 20.0 Hz)')
        with pytest.raises(ValueError) as e:
            spectrogram.spectrogram(data, samp_rate=20.0, show=False)
        assert str(e.value) == msg
        # Only 130 samples at 1 Hz and our default tries to window 128 samples,
        # so we dont have enough data for two windows. In principle we could
        # still plot that but our code currently relies on having at least two
        # windows for plotting. It does not seem worthwhile to change the code
        # to do so, so just show a nice error
        data = np.ones(130)
        msg = ('Input signal too short (130 samples, window length 6.4 '
               'seconds, nfft 128 samples, 115 samples window overlap, '
               'sampling rate 20.0 Hz)')
        with pytest.raises(ValueError) as e:
            spectrogram.spectrogram(data, samp_rate=20.0, show=False)
        assert str(e.value) == msg
        plt.close('all')
