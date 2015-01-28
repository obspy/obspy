# -*- coding: utf-8 -*-
"""
The obspy.imaging.waveform test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import readEvents
from obspy.core.stream import read
from obspy.core.util import AttribDict
from obspy.core.util.testing import ImageComparison
import numpy as np
import os
import unittest


class WaveformTestCase(unittest.TestCase):
    """
    Test cases for waveform plotting.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'images')

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
        number_of_samples = int(time_delta * sampling_rate) + 1
        # Calculate first sine wave.
        curve = np.linspace(0, 2 * np.pi, number_of_samples // 2)
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
        org_st = st.copy()
        st.plot(format='png')
        self.assertEqual(st, org_st)
        # Now with min-max list creation (more than 400000 samples).
        st = self._createStream(UTCDateTime(0), UTCDateTime(600000), 1)
        st[0].stats.calib = 0.2343
        org_st = st.copy()
        st.plot(format='png')
        self.assertEqual(st, org_st)
        # Now only plot a certain time frame.
        st.plot(
            format='png', starrtime=UTCDateTime(10000),
            endtime=UTCDateTime(20000))
        self.assertEqual(st, org_st)

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

    def test_plotOneHourManySamples(self):
        """
        Plots one hour, starting Jan 1970.

        Uses a frequency of 1000 Hz to get a sample count of over 3 Million and
        get in the range, that plotting will choose to use a minimum maximum
        approach to plot the data.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600, 1000.0)
        # create and compare image
        image_name = 'waveform_one_hour_many_samples.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name)

    def test_plotOneHourFewSamples(self):
        """
        Plots one hour, starting Jan 1970.

        Uses a frequency of 10 Hz.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600, 10.0)
        # create and compare image
        image_name = 'waveform_one_hour_few_samples.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name)

    def test_plotSimpleGapManySamples(self):
        """
        Plots three hours with a gap.

        There are 45 minutes of data at the beginning and 45 minutes of data at
        the end.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600 * 3 / 4, 500.0)
        st += self._createStream(start + 2.25 * 3600, start + 3 * 3600, 500.0)
        # create and compare image
        image_name = 'waveform_simple_gap_many_samples.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name)

    def test_plotSimpleGapFewSamples(self):
        """
        Plots three hours with a gap.

        There are 45 minutes of data at the beginning and 45 minutes of data at
        the end.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600 * 3 / 4, 5.0)
        st += self._createStream(start + 2.25 * 3600, start + 3 * 3600, 5.0)
        # create and compare image
        image_name = 'waveform_simple_gap_few_samples.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name)

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
        # create and compare image
        image_name = 'waveform_complex_gap_many_samples.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name)

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
        # create and compare image
        image_name = 'waveform_complex_gap_few_samples.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name)

    def test_plotMultipleTraces(self):
        """
        Plots multiple traces underneath.
        """
        # 1 trace
        st = read()[1]
        with ImageComparison(self.path, 'waveform_1_trace.png') as ic:
            st.plot(outfile=ic.name, automerge=False)
        # 3 traces
        st = read()
        with ImageComparison(self.path, 'waveform_3_traces.png') as ic:
            st.plot(outfile=ic.name, automerge=False)
        # 5 traces
        st = st[1] * 5
        with ImageComparison(self.path, 'waveform_5_traces.png') as ic:
            st.plot(outfile=ic.name, automerge=False)
        # 10 traces
        st = st[1] * 10
        with ImageComparison(self.path, 'waveform_10_traces.png') as ic:
            st.plot(outfile=ic.name, automerge=False)
        # 10 traces - huge numbers
        st = st[1] * 10
        for i, tr in enumerate(st):
            # scale data to have huge numbers
            st[i].data = tr.data * 10 ** i
        with ImageComparison(self.path, 'waveform_10_traces_huge.png') as ic:
            st.plot(outfile=ic.name, automerge=False, equal_scale=False)
        # 10 traces - tiny numbers
        st = st[1] * 10
        for i, tr in enumerate(st):
            # scale data to have huge numbers
            st[i].data = tr.data / (10 ** i)
        with ImageComparison(self.path, 'waveform_10_traces_tiny.png') as ic:
            st.plot(outfile=ic.name, automerge=False, equal_scale=False)

    def test_plotWithLabels(self):
        """
        Plots with labels.
        """
        st = read()
        st.label = u"Title #1 üöä?"
        st[0].label = 'Hello World!'
        st[1].label = u'Hällö Wörld & Marß'
        st[2].label = '*' * 80
        # create and compare image
        with ImageComparison(self.path, 'waveform_labels.png') as ic:
            st.plot(outfile=ic.name)

    def test_plotBinningError(self):
        """
        Tests the plotting of a trace with a certain amount of sampling that
        had a binning problem.
        """
        tr = Trace(data=np.sin(np.linspace(0, 200, 432000)))
        # create and compare image
        with ImageComparison(self.path, 'waveform_binning_error.png') as ic:
            tr.plot(outfile=ic.name)

        tr = Trace(data=np.sin(np.linspace(0, 200, 431979)))
        # create and compare image
        with ImageComparison(self.path, 'waveform_binning_error_2.png') as ic:
            tr.plot(outfile=ic.name)

    def test_plotDefaultSection(self):
        """
        Tests plotting 10 in a section
        """
        start = UTCDateTime(0)
        st = Stream()
        for _i in range(10):
            this_start = start + 300 * np.sin(np.pi * _i / 9)
            st += self._createStream(this_start, this_start + 3600, 100)
            st[-1].stats.distance = _i * 10e3
        # create and compare image
        with ImageComparison(self.path, 'waveform_default_section.png') as ic:
            st.plot(outfile=ic.name, type='section')

    def test_plotAzimSection(self):
        """
        Tests plotting 10 in a azimuthal distant section
        """
        start = UTCDateTime(0)
        st = Stream()
        for _i in range(10):
            st += self._createStream(start, start + 3600, 100)
            st[-1].stats.coordinates = AttribDict({
                'latitude': _i,
                'longitude': _i})
        # create and compare image
        with ImageComparison(self.path, 'waveform_azim_section.png') as ic:
            st.plot(outfile=ic.name, type='section', dist_degree=True,
                    ev_coord=(0.0, 0.0))

    def test_plotRefTimeSection(self):
        """
        Tests plotting 10 in a section with alternate reference time
        """
        start = UTCDateTime(0)
        reftime = start + 600
        st = Stream()
        for _i in range(10):
            this_start = start + 300 * np.sin(np.pi * _i / 9)
            st += self._createStream(this_start, this_start + 3600, 100)
            st[-1].stats.distance = _i * 10e3
        # create and compare image
        with ImageComparison(self.path, 'waveform_reftime_section.png') as ic:
            st.plot(outfile=ic.name, type='section', reftime=reftime)

    def test_plotDefaultRelative(self):
        """
        Plots one hour, starting Jan 1970, with a relative scale.
        """
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3600, 100)
        # create and compare image
        image_name = 'waveform_default_relative.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name, type='relative')

    def test_plotRefTimeRelative(self):
        """
        Plots one hour, starting Jan 1970, with a relative scale.

        The reference time is at 300 seconds after the start.
        """
        start = UTCDateTime(0)
        ref = UTCDateTime(300)
        st = self._createStream(start, start + 3600, 100)
        # create and compare image
        image_name = 'waveform_reftime_relative.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name, type='relative', reftime=ref)

    def test_plotDayPlot(self):
        '''
        Plots day plot, starting Jan 1970.
        '''
        start = UTCDateTime(0)
        st = self._createStream(start, start + 3 * 3600, 100)
        # create and compare image
        image_name = 'waveform_dayplot.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name, type='dayplot',
                    timezone='EST', time_offset=-5)

    def test_plotDayPlotExplicitEvent(self):
        '''
        Plots day plot, starting Jan 1970, with several events.
        '''
        start = UTCDateTime(0)
        event1 = UTCDateTime(30)       # Event: Top left; Note: below right
        event2 = UTCDateTime(14 * 60)  # Event: Top right; Note: below left
        event3 = UTCDateTime(46 * 60)  # Event: Bottom left; Note: above right
        event4 = UTCDateTime(59 * 60)  # Event: Bottom right; Note: above left
        event5 = UTCDateTime(61 * 60)  # Should be ignored
        st = self._createStream(start, start + 3600, 100)
        # create and compare image
        image_name = 'waveform_dayplot_event.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name, type='dayplot',
                    timezone='EST', time_offset=-5,
                    events=[{'time': event1, 'text': 'Event 1'},
                            {'time': event2, 'text': 'Event 2'},
                            {'time': event3, 'text': 'Event 3'},
                            {'time': event4, 'text': 'Event 4'},
                            {'time': event5, 'text': 'Event 5'}])

    def test_plotDayPlotCatalog(self):
        '''
        Plots day plot, with a catalog of events.
        '''
        start = UTCDateTime(2012, 4, 4, 14, 0, 0)
        cat = readEvents()
        st = self._createStream(start, start + 3600, 100)
        # create and compare image
        image_name = 'waveform_dayplot_catalog.png'
        with ImageComparison(self.path, image_name) as ic:
            st.plot(outfile=ic.name, type='dayplot',
                    timezone='EST', time_offset=-5,
                    events=cat)


def suite():
    return unittest.makeSuite(WaveformTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
