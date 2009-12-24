# -*- coding: utf-8 -*-
"""
The obspy.imaging.waveform test suite.
"""

from obspy.core import Stream, Trace, UTCDateTime
import inspect
import numpy as N
import os
import unittest


class WaveformTestCase(unittest.TestCase):
    """
    Test cases for waveform plotting.
    """
    def setUp(self):
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'output')
        # Create a test file which can be used for all test cases. This creates
        # a wiggly sinus period.
        data = N.linspace(0, 2 * N.pi, 100000)
        data = 10 * (N.sin(data) + 0.2 * N.sin(20 * data) * N.sin(7777 * data))
        header = {'starttime' : UTCDateTime(0), 'endtime' : UTCDateTime(0) + \
                  100000, 'sampling_rate' : 1, 'npts' : data.size,
                  'network' : 'AA', 'station' : 'BB', 'channel' : 'CC'}
        self.stream = Stream(traces=[Trace(data=data, header=header)])
        # Create a the same file again but use more data values..
        data = N.linspace(0, 2 * N.pi, 1000000)
        data = 10 * (N.sin(data) + 0.2 * N.sin(20 * data) * N.sin(7777 * data))
        header = {'starttime' : UTCDateTime(0), 'endtime' : UTCDateTime(0) + \
                  100000, 'sampling_rate' : 10, 'npts' : data.size,
                  'network' : 'AA', 'station' : 'BB', 'channel' : 'CC'}
        self.large_stream = Stream(traces=[Trace(data=data, header=header)])

    def tearDown(self):
        pass

    def test_plotEmptyStream(self):
        """
        Plotting of an empty stream should raise a warning.
        """
        st = Stream()
        self.assertRaises(IndexError, st.plot)

    def test_WaveformStraightPlotting(self):
        """
        Create waveform plotting examples in tests/output directory.
        
        This approach uses small data sets and therefore the waveform plotting
        method just plots all data values.
        """
        # First create a reference plot.
        self.stream.plot(outfile=\
            os.path.join(self.path, 'waveform_straightPlotting-reference.png'))
        # Create a second identical Trace but shift the times a little bit.
        # Also make the plots green and the background in purple.
        self.stream += self.stream
        self.stream[0].stats.location = '01'
        self.stream[1].stats.location = '00'
        self.stream[0].stats.starttime = \
            self.stream[0].stats.starttime + 2 * 60 * 60
        self.stream[0].stats.endtime = \
            self.stream[0].stats.endtime + 2 * 60 * 60
        self.stream.plot(outfile=\
            os.path.join(self.path, 'waveform_straightPlotting-2traces.png'),
            color='green', bgcolor='#F5FEA5', face_color='purple')
        # Make a simple plot with a gap and adjust the ticks to show the
        # weekday and microsecond and rotate the ticks two degrees. All
        # background should also be transparent.
        self.stream.sort()
        self.stream[0].stats.location = ''
        self.stream[1].stats.location = ''
        self.stream[1].stats.starttime = \
            self.stream[0].stats.starttime + 2 * 24 * 60 * 60
        self.stream[1].stats.endtime = \
            self.stream[0].stats.endtime + 2 * 24 * 60 * 60
        self.stream.plot(outfile=\
            os.path.join(self.path,
            'waveform_straightPlotting-simple-gap-transparent.png'),
            tick_format='%a, %H:%M:%S:%f', tick_rotation=2,
            transparent=True)
        # Create a two Traces plot with a gap. This should
        # result in two Traces and the second trace should fit right in the
        # middle of the first trace. The second trace will also only have half
        # height but should still be centered despite being moved up a notch.
        self.stream.append(self.stream[0])
        self.stream[2].stats.starttime = self.stream[0].stats.endtime
        self.stream[2].stats.endtime = self.stream[0].stats.endtime + 100000
        self.stream[2].stats.network = 'ZZ'
        self.stream[1].stats.starttime = self.stream[2].stats.endtime
        self.stream[1].stats.endtime = self.stream[2].stats.endtime + 100000
        # Make half the amplitude but move it up 100 units.
        self.stream[2].data = 0.5 * self.stream[2].data + 100
        self.stream.plot(outfile=\
            os.path.join(self.path,
                'waveform_straightPlotting-2traces-gap.png'), color='lime')

    def test_WaveformMinMaxApproachPlotting(self):
        """
        Create waveform plotting examples in tests/output directory.
        
        This approach uses large data sets and therefore the waveform plotting
        method uses a minima and maxima approach to plot the data.
        """
        # First create a reference plot.
        self.large_stream.plot(outfile=\
            os.path.join(self.path, 'waveform_MinMaxPlotting-reference.png'))
        # Create a second identical Trace but shift the times a little bit.
        # Also make the plots green and the background in purple.
        self.large_stream += self.large_stream
        self.large_stream[0].stats.location = '01'
        self.large_stream[1].stats.location = '00'
        self.large_stream[0].stats.starttime = \
            self.large_stream[0].stats.starttime + 2 * 60 * 60
        self.large_stream[0].stats.endtime = \
            self.large_stream[0].stats.endtime + 2 * 60 * 60
        self.large_stream.plot(outfile=\
            os.path.join(self.path, 'waveform_MinMaxPlotting-2traces.png'),
            color='green', bgcolor='#F5FEA5', face_color='purple')
        # Make a simple plot with a gap and adjust the ticks to show the
        # weekday and microsecond and rotate the ticks two degrees. All
        # background should also be transparent.
        self.large_stream.sort()
        self.large_stream[0].stats.location = ''
        self.large_stream[1].stats.location = ''
        self.large_stream[1].stats.starttime = \
            self.large_stream[0].stats.starttime + 2 * 24 * 60 * 60
        self.large_stream[1].stats.endtime = \
            self.large_stream[0].stats.endtime + 2 * 24 * 60 * 60
        self.large_stream.plot(outfile=\
            os.path.join(self.path,
            'waveform_MinMaxPlotting-simple-gap-transparent.png'),
            tick_format='%a, %H:%M:%S:%f', tick_rotation=2,
            transparent=True)
        # Create a two Traces plot with a gap. This should
        # result in two Traces and the second trace should fit right in the
        # middle of the first trace. The second trace will also only have half
        # height but should still be centered despite being moved up a notch.
        self.large_stream.append(self.large_stream[0])
        self.large_stream[2].stats.starttime = \
                                            self.large_stream[0].stats.endtime
        self.large_stream[2].stats.endtime = \
                                    self.large_stream[0].stats.endtime + 100000
        self.large_stream[2].stats.network = 'ZZ'
        self.large_stream[1].stats.starttime = \
                                            self.large_stream[2].stats.endtime
        self.large_stream[1].stats.endtime = \
                                    self.large_stream[2].stats.endtime + 100000
        # Make half the amplitude but move it up 100 units.
        self.large_stream[2].data = 0.5 * self.large_stream[2].data + 100
        self.large_stream.plot(outfile=\
            os.path.join(self.path,
                'waveform_MinMaxPlotting-2traces-gap.png'), color='lime')


def suite():
    return unittest.makeSuite(WaveformTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
