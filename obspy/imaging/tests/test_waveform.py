# -*- coding: utf-8 -*-
"""
The obspy.imaging.waveform test suite.
"""
from obspy import Stream, Trace, UTCDateTime
from obspy.core.stream import read
from obspy.core.util.decorator import skipIf
import numpy as np
import os
import unittest
from copy import deepcopy


class WaveformTestCase(unittest.TestCase):
    """
    Test cases for waveform plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'output')

    def _createStream(self, starttime, endtime, sampling_rate):
        """
        Helper method to create a Stream object that can be used for testing
        waveform plotting.

        Takes the time frame of the Stream to be created and a sampling rate.
        Any other header information will have to be adjusted on a case by case
        basis. Please remember to use the same sampling rate for one Trace as
        merging and plotting will not work otherwise.

        This method will create a single sine curve to a first approximation
        with superimposed 10 smaller sine curves on it.

        :return: Stream object
        """
        time_delta = endtime - starttime
        number_of_samples = time_delta * sampling_rate + 1
        # Calculate first sine wave.
        curve = np.linspace(0, 2 * np.pi, int(number_of_samples // 2))
        # Superimpose it with a smaller but shorter wavelength sine wave.
        curve = np.sin(curve) + 0.2 * np.sin(10 * curve)
        # To get a thick curve alternate between two curves.
        data = np.empty(number_of_samples)
        # Check if even number and adjust if necessary.
        if number_of_samples % 2 == 0:
            data[0::2] = curve
            data[1::2] = curve + 0.2
        else:
            data[-1] = 0.0
            data[0:-1][0::2] = curve
            data[0:-1][1::2] = curve + 0.2
        tr = Trace()
        tr.stats.starttime = starttime
        tr.stats.sampling_rate = float(sampling_rate)
        # Fill dummy header.
        tr.stats.network = 'BW'
        tr.stats.station = 'OBSPY'
        tr.stats.channel = 'TEST'
        tr.data = data
        return Stream(traces=[tr])

    def test_dataRemainsUnchanged(self):
        """
        Data should not be changed when plotting.
        """
        # Use once with straight plotting with random calibration factor
        st = self._createStream(UTCDateTime(0), UTCDateTime(1000), 1)
        st[0].stats.calib = 0.2343
        org_data = deepcopy(st[0].data)
        st.plot(format='png')
        # compare with original data
        np.testing.assert_array_equal(org_data, st[0].data)
        # Now with min-max list creation (more than 400000 samples).
        st = self._createStream(UTCDateTime(0), UTCDateTime(600000), 1)
        st[0].stats.calib = 0.2343
        org_data = deepcopy(st[0].data)
        st.plot(format='png')
        # compare with original data
        np.testing.assert_array_equal(org_data, st[0].data)

    def test_plotEmptyStream(self):
        """
        Plotting of an empty stream should raise a warning.
        """
        st = Stream()
        self.assertRaises(IndexError, st.plot)

    def test_plotSameTraceDifferentSampleRates(self):
        """
        Plotting of a Stream object, that contains two traces with the same id
        and different sampling rates should raise an exception.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 10, 1.0)
        st += self._createStream(start + 10, start + 20, 10.0)
        self.assertRaises(Exception, st.plot)

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotOneHourManySamples(self):
        """
        Plots one hour, starting Jan 1970.

        Uses a frequency of 1000 Hz to get a sample count of over 3 Million and
        get in the range, that plotting will choose to use a minimum maximum
        approach to plot the data.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600, 1000.0)
        filename = 'waveform_one_hour_many_samples.png'
        st.plot(outfile=os.path.join(self.path, filename))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotOneHourFewSamples(self):
        """
        Plots one hour, starting Jan 1970.

        Uses a frequency of 10 Hz.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600, 10.0)
        filename = 'waveform_one_hour_few_samples.png'
        st.plot(outfile=os.path.join(self.path, filename))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotSimpleGapManySamples(self):
        """
        Plots three hours with a gap.

        There are 45 minutes of data at the beginning and 45 minutes of data at
        the end.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600 * 3 / 4, 500.0)
        st += self._createStream(start + 2.25 * 3600, start + 3 * 3600, 500.0)
        filename = 'waveform_simple_gap_many_samples.png'
        st.plot(outfile=os.path.join(self.path, filename))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotSimpleGapFewSamples(self):
        """
        Plots three hours with a gap.

        There are 45 minutes of data at the beginning and 45 minutes of data at
        the end.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600 * 3 / 4, 5.0)
        st += self._createStream(start + 2.25 * 3600, start + 3 * 3600, 5.0)
        filename = 'waveform_simple_gap_few_samples.png'
        st.plot(outfile=os.path.join(self.path, filename))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotComplexGapManySamples(self):
        """
        Plots three hours with a gap.

        There are 45 minutes of data at the beginning and 45 minutes of data at
        the end.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600 * 3 / 4, 500.0)
        st += self._createStream(start + 2.25 * 3600, start + 3 * 3600, 500.0)
        st[0].stats.location = '01'
        st[1].stats.location = '01'
        temp_st = self._createStream(start + 3600 * 3 / 4, start + 2.25 * 3600,
                                     500.0)
        temp_st[0].stats.location = '02'
        st += temp_st
        filename = 'waveform_complex_gap_many_samples.png'
        st.plot(outfile=os.path.join(self.path, filename))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotComplexGapFewSamples(self):
        """
        Plots three hours with a gap.

        There are 45 minutes of data at the beginning and 45 minutes of data at
        the end.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600 * 3 / 4, 5.0)
        st += self._createStream(start + 2.25 * 3600, start + 3 * 3600, 5.0)
        st[0].stats.location = '01'
        st[1].stats.location = '01'
        temp_st = self._createStream(start + 3600 * 3 / 4, start + 2.25 * 3600,
                                     5.0)
        temp_st[0].stats.location = '02'
        st += temp_st
        filename = 'waveform_complex_gap_few_samples.png'
        st.plot(outfile=os.path.join(self.path, filename))

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotMultipleTraces(self):
        """
        Plots multiple traces underneath.
        """
        st = read()
        # 1 trace
        outfile = os.path.join(self.path, 'waveform_1_trace.png')
        st[0].plot(outfile=outfile, automerge=False)
        # 3 traces
        outfile = os.path.join(self.path, 'waveform_3_traces.png')
        st.plot(outfile=outfile, automerge=False)
        # 5 traces
        st = st[0] * 5
        outfile = os.path.join(self.path, 'waveform_5_traces.png')
        st.plot(outfile=outfile, automerge=False)
        # 10 traces
        st = st[0] * 10
        outfile = os.path.join(self.path, 'waveform_10_traces.png')
        st.plot(outfile=outfile, automerge=False)
        # 10 traces - huge numbers
        st = st[0] * 10
        for i, tr in enumerate(st):
            # scale data to have huge numbers
            st[i].data = tr.data * 10 ** i
        outfile = os.path.join(self.path, 'waveform_10_traces_huge.png')
        st.plot(outfile=outfile, automerge=False, equal_scale=False)
        # 10 traces - tiny numbers
        st = st[0] * 10
        for i, tr in enumerate(st):
            # scale data to have huge numbers
            st[i].data = tr.data / (10 ** i)
        outfile = os.path.join(self.path, 'waveform_10_traces_tiny.png')
        st.plot(outfile=outfile, automerge=False, equal_scale=False)
        # 20 traces
        st = st[0] * 20
        outfile = os.path.join(self.path, 'waveform_20_traces.png')
        st.plot(outfile=outfile, automerge=False)

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotWithLabels(self):
        """
        Plots with labels.
        """
        st = read()
        st.label = u"Title #1 üöä?"
        st[0].label = 'Hello World!'
        st[1].label = u'Hällö Wörld & Marß'
        st[2].label = '*' * 80
        outfile = os.path.join(self.path, 'waveform_labels.png')
        st.plot(outfile=outfile)

    @skipIf(__name__ != '__main__', 'test must be started manually')
    def test_plotBinningError(self):
        """
        Tests the plotting of a trace with a certain amount of sampling that
        had a binning problem.
        """
        tr = Trace(data=np.sin(np.linspace(0, 200, 432000)))
        outfile = os.path.join(self.path, 'binning_error.png')
        tr.plot(outfile=outfile)

        tr = Trace(data=np.sin(np.linspace(0, 200, 431979)))
        outfile = os.path.join(self.path, 'binning_error_2.png')
        tr.plot(outfile=outfile)


def suite():
    return unittest.makeSuite(WaveformTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
