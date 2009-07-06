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
    Test cases for libgse2 core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data', 'loc_RJOB20050831023349.z')

    def tearDown(self):
        pass

    def test_readViaObspy(self):
        """
        Read files via L{obspy.Trace}
        """
        testdata = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        #
        tr = obspy.read(self.file, format='GSE2')[0]
        self.assertEqual(tr.stats['station'], 'RJOB ')
        self.assertEqual(tr.stats.npts, 12000)
        self.assertEqual(tr.stats['sampling_rate'], 200)
        self.assertEqual(tr.stats.get('channel'), '  Z')
        self.assertEqual(tr.stats.get('vang'), -1.0)
        self.assertEqual(tr.stats.get('calper'), 1.0)
        self.assertAlmostEqual(tr.stats.get('calib'), 9.49e-02)
        self.assertEqual(tr.stats.starttime.timestamp, 1125455629.849998)
        for _i in xrange(13):
            self.assertEqual(tr.data[_i], testdata[_i])

    def test_readAndWriteViaObspy(self):
        """
        Read and Write files via L{obspy.Trace}
        """
        self.file = os.path.join(self.path, 'data', 'loc_RNON20040609200559.z')
        tmpfile = os.path.join(self.path, 'data', 'tmp.gse2')
        # read trace
        tr = obspy.read(self.file, format='GSE2')[0]
        # write comparison trace
        st2 = obspy.Stream()
        st2.traces.append(obspy.Trace())
        tr2 = st2[0]
        tr2.data = copy.deepcopy(tr.data)
        tr2.stats = copy.deepcopy(tr.stats)
        st2.write(tmpfile, format='GSE2')
        # read comparison trace
        tr3 = obspy.read(tmpfile)[0]
        # check if equal
        self.assertEqual(tr3.stats['station'], tr.stats['station'])
        self.assertEqual(tr3.stats.npts, tr.stats.npts)
        self.assertEqual(tr3.stats['sampling_rate'], tr.stats['sampling_rate'])
        self.assertEqual(tr3.stats.get('channel'), tr.stats.get('channel'))
        self.assertEqual(tr3.stats.get('starttime'), tr.stats.get('starttime'))
        self.assertEqual(tr3.stats.get('vang'), tr.stats.get('vang'))
        self.assertEqual(tr3.stats.get('calper'), tr.stats.get('calper'))
        self.assertEqual(tr3.stats.get('calib'), tr.stats.get('calib'))
        N.testing.assert_equal(tr3.data, tr.data)
        #os.remove(tmpfile)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
