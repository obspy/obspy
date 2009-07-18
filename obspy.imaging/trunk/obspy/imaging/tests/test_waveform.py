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
                  100000, 'sampling_rate' : 1, 'npts' : 100000,
                  'network' : 'AA', 'station' : 'BB', 'channel' : 'CC'}
        self.stream = Stream(traces=[Trace(data=data, header=header)])

    def tearDown(self):
        pass

    def test_Waveform(self):
        """
        Create waveform plotting examples in tests/output directory.
        """
        # First create a reference plot.
        self.stream.plot(outfile=\
            os.path.join(self.path, 'Waveform_ReferencePlot.png'))

        # Create a second identical Trace but shift the times a little bit.
        # Also make the plots green and the background light gray.
        self.stream += self.stream
        self.stream[0].stats.location = '01'
        self.stream[1].stats.location = '00'
        self.stream[0].stats.starttime = \
            self.stream[0].stats.starttime + 2 * 60 * 60
        self.stream[0].stats.endtime = \
            self.stream[0].stats.endtime + 2 * 60 * 60
        self.stream.plot(outfile=\
            os.path.join(self.path, 'Waveform_TwoTraces.png'),
            color='green', bgcolor='#F5FEA5', face_color='0.7')

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
            os.path.join(self.path, 'Waveform_SimpleGap_TransparentBG.png'),
            tick_format='%a, %H:%M:%S:%f', tick_rotation=2,
            transparent=True)

        # Create a two Traces plot with a gap and a color gradient. This should
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
            os.path.join(self.path, 'Waveform_TwoTrace_Gap_and_Gradient.png'),
            color=('#FF0000', '#00FFFF'))


def suite():
    return unittest.makeSuite(WaveformTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
