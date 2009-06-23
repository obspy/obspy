#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The gse2.core test suite.
"""

from obspy.core import Trace
import inspect, os, unittest, copy


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
        tr = Trace()
        tr.read(self.file, format='GSE2')
        self.assertEqual(tr.stats['station'], 'RJOB ')
        self.assertEqual(tr.stats.npts, 12000)
        self.assertEqual(tr.stats['sampling_rate'], 200)
        self.assertEqual(tr.stats.get('channel'), '  Z')
        self.assertEqual(tr.stats.starttime.timestamp(), 1125455629.849998)
        for _i in xrange(13):
            self.assertEqual(tr.data[_i], testdata[_i])
        tr2 = Trace()
        tr2.readGSE2(self.file, format='GSE2')
        self.assertEqual(tr2.stats['station'], 'RJOB ')
        self.assertEqual(tr2.stats.npts, 12000)
        self.assertEqual(tr2.stats['sampling_rate'], 200)
        self.assertEqual(tr2.stats.get('channel'), '  Z')
        self.assertEqual(tr2.stats.starttime.timestamp(), 1125455629.849998)
        for _i in xrange(13):
            self.assertEqual(tr2.data[_i], testdata[_i])

    def test_readAndWriteViaObspy(self):
        """
        Read and Write files via L{obspy.Trace}
        """
        self.file = os.path.join(self.path, 'data', 'loc_RNON20040609200559.z')
        tmpfile = os.path.join(self.path, 'data', 'tmp.gse2')
        #
        tr = Trace()
        tr.read(self.file, format='GSE2')
        tr2 = Trace()
        tr2.data = copy.deepcopy(tr.data)
        #tr2.data = copy.deepcopy(tr.data)
        for _i in ['d_year', 'd_mon', 'd_day', 't_hour', 't_min', 't_sec',
                   'station', 'channel', 'auxid', 'datatype', 'n_samps',
                   'samp_rate', 'calib', 'calper', 'instype', 'hang', 'vang']:
            setattr(tr2.stats, _i, getattr(tr.stats, _i))
        tr2.write(tmpfile, format='GSE2')
        tr3 = Trace()
        tr3.read(tmpfile)
        self.assertEqual(tr3.stats['station'], tr.stats['station'])
        self.assertEqual(tr3.stats.npts, tr.stats.npts)
        self.assertEqual(tr.stats['sampling_rate'], tr.stats['sampling_rate'])
        self.assertEqual(tr.stats.get('channel'), tr.stats.get('channel'))
        self.assertEqual(tr.stats.get('starttime'), tr.stats.get('starttime'))
        for _i in xrange(100):
            self.assertEqual(tr.data[_i], tr3.data[_i])
        os.remove(tmpfile)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
