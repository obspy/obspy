#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The gse2.core test suite.
"""

import obspy
import inspect, os, unittest, copy
import numpy as N


class CoreTestCase(unittest.TestCase):
    """
    Test cases for sacio core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data', 'test.sac')

    def tearDown(self):
        pass

    def test_readViaObspy(self):
        """
        Read files via L{obspy.Trace}
        """
        testdata = N.array([ -8.74227766e-08,  -3.09016973e-01,
            -5.87785363e-01, -8.09017122e-01,  -9.51056600e-01,
            -1.00000000e+00, -9.51056302e-01,  -8.09016585e-01,
            -5.87784529e-01, -3.09016049e-01], dtype='float32')
        #
        tr = obspy.read(self.file, format='SAC')[0]
        self.assertEqual(tr.stats['station'], 'STA     ')
        self.assertEqual(tr.stats.npts, 100)
        self.assertEqual(tr.stats['sampling_rate'], 1.0)
        self.assertEqual(tr.stats.get('channel'), 'Q       ')
        self.assertEqual(tr.stats.starttime.timestamp, 269596800.0)
        N.testing.assert_array_almost_equal(testdata[0:10], tr.data[0:10])

    def test_readAndWriteViaObspy(self):
        raise NotImplementedError("Write Method not implemented jet")

    #def test_readAndWriteViaObspy(self):
    #    """
    #    Read and Write files via L{obspy.Trace}
    #    """
    #    self.file = os.path.join(self.path, 'data', 'loc_RNON20040609200559.z')
    #    tmpfile = os.path.join(self.path, 'data', 'tmp.gse2')
    #    # read trace
    #    tr = obspy.read(self.file, format='GSE2')[0]
    #    # write comparison trace
    #    st2 = obspy.Stream()
    #    st2.traces.append(obspy.Trace())
    #    tr2 = st2[0]
    #    tr2.data = copy.deepcopy(tr.data)
    #    tr2.stats = copy.deepcopy(tr.stats)
    #    st2.write(tmpfile, format='GSE2')
    #    # read comparison trace
    #    tr3 = obspy.read(tmpfile)[0]
    #    os.remove(tmpfile)
    #    # check if equal
    #    self.assertEqual(tr3.stats['station'], tr.stats['station'])
    #    self.assertEqual(tr3.stats.npts, tr.stats.npts)
    #    self.assertEqual(tr.stats['sampling_rate'], tr.stats['sampling_rate'])
    #    self.assertEqual(tr.stats.get('channel'), tr.stats.get('channel'))
    #    self.assertEqual(tr.stats.get('starttime'), tr.stats.get('starttime'))
    #    self.assertEqual(tr.stats.get('vang'), tr.stats.get('vang'))
    #    self.assertEqual(tr.stats.get('calper'), tr.stats.get('calper'))
    #    self.assertEqual(tr.stats.get('calib'), tr.stats.get('calib'))
    #    N.testing.assert_equal(tr.data, tr3.data)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
