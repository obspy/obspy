# -*- coding: utf-8 -*-
"""
The sac.core test suite.
"""

from obspy.core import Stream, Trace, read
import copy
import inspect
import numpy as N
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    Test cases for sac core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data', 'test.sac')
        self.testdata = N.array([ -8.74227766e-08, -3.09016973e-01,
            - 5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
            - 1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
            - 5.87784529e-01, -3.09016049e-01], dtype='float32')

    def tearDown(self):
        pass

    def test_readViaObspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        self.assertEqual(tr.stats['station'], 'STA     ')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q       ')
        self.assertEqual(tr.stats.starttime.timestamp, 269596800.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        N.testing.assert_array_almost_equal(self.testdata[0:10], tr.data[0:10])

    def test_readHeadViaObspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC', headonly=True)[0]
        self.assertEqual(tr.stats['station'], 'STA     ')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q       ')
        self.assertEqual(tr.stats.starttime.timestamp, 269596800.0)
        self.assertEqual(tr.stats.sac.get('nvhdr'), 6)
        self.assertEqual(str(tr.data), '[]')

    def test_writeViaObspy(self):
        """
        Writing artificial files via L{obspy.Stream}
        """
        st = Stream(traces=[Trace(header={'sac':{}}, data=self.testdata)])
        tempfile = os.path.join(self.path, 'data', 'tmp.jjj')
        st.write(tempfile, format='SAC')
        tr = read(tempfile)[0]
        os.remove(tempfile)
        N.testing.assert_array_almost_equal(self.testdata, tr.data)

    def test_readAndWriteViaObspy(self):
        """
        Read and Write files via L{obspy.Stream}
        """
        # read trace
        tr = read(self.file)[0]
        # write comparison trace
        st2 = Stream()
        st2.traces.append(Trace())
        tr2 = st2[0]
        tr2.data = copy.deepcopy(tr.data)
        tr2.stats = copy.deepcopy(tr.stats)
        tempfile = os.path.join(self.path, 'data', 'tmp.jjj')
        st2.write(tempfile, format='SAC')
        # read comparison trace
        tr3 = read(tempfile)[0]
        os.remove(tempfile)
        # check if equal
        self.assertEqual(tr3.stats['station'], tr.stats['station'])
        self.assertEqual(tr3.stats.npts, tr.stats.npts)
        self.assertEqual(tr.stats['sampling_rate'], tr.stats['sampling_rate'])
        self.assertEqual(tr.stats.get('channel'), tr.stats.get('channel'))
        self.assertEqual(tr.stats.get('starttime'), tr.stats.get('starttime'))
        self.assertEqual(tr.stats.sac.get('nvhdr'), tr.stats.sac.get('nvhdr'))
        N.testing.assert_equal(tr.data, tr3.data)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
