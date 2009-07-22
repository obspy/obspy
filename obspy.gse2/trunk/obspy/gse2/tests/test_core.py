#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The gse2.core test suite.
"""

from obspy.core import Stream, Trace, UTCDateTime, read
import inspect
import os
import unittest
import copy
import numpy as N


class CoreTestCase(unittest.TestCase):
    """
    Test cases for libgse2 core interface
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))

    def tearDown(self):
        pass

    def test_readViaObspy(self):
        """
        Read files via L{obspy.Trace}
        """
        gse2file = os.path.join(self.path, 'data', 'loc_RJOB20050831023349.z')
        testdata = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        # read
        st = read(gse2file)
        st.verify()
        tr = st[0]
        self.assertEqual(tr.stats['station'], 'RJOB ')
        self.assertEqual(tr.stats.npts, 12000)
        self.assertEqual(tr.stats['sampling_rate'], 200)
        self.assertEqual(tr.stats.get('channel'), '  Z')
        self.assertEqual(tr.stats.gse2.get('vang'), -1.0)
        self.assertEqual(tr.stats.gse2.get('calper'), 1.0)
        self.assertAlmostEqual(tr.stats.gse2.get('calib'), 9.49e-02)
        self.assertEqual(tr.stats.starttime.timestamp, 1125455629.849998)
        for _i in xrange(13):
            self.assertEqual(tr.data[_i], testdata[_i])

    def test_readAndWriteViaObspy(self):
        """
        Read and Write files via L{obspy.Trace}
        """
        tempfile = 'temp1.gse2'
        gse2file = os.path.join(self.path, 'data', 'loc_RNON20040609200559.z')
        # read trace
        st1 = read(gse2file)
        st1.verify()
        tr1 = st1[0]
        # write comparison trace
        st2 = Stream()
        st2.traces.append(Trace())
        tr2 = st2[0]
        tr2.data = copy.deepcopy(tr1.data)
        tr2.stats = copy.deepcopy(tr1.stats)
        st2.write(tempfile, format='GSE2')
        # read comparison trace
        st3 = read(tempfile)
        os.remove(tempfile)
        st3.verify()
        tr3 = st3[0]
        # check if equal
        self.assertEqual(tr3.stats['station'], tr1.stats['station'])
        self.assertEqual(tr3.stats.npts, tr1.stats.npts)
        self.assertEqual(tr3.stats['sampling_rate'],
                         tr1.stats['sampling_rate'])
        self.assertEqual(tr3.stats.get('channel'),
                         tr1.stats.get('channel'))
        self.assertEqual(tr3.stats.get('starttime'),
                         tr1.stats.get('starttime'))
        self.assertEqual(tr3.stats.gse2.get('vang'),
                         tr1.stats.gse2.get('vang'))
        self.assertEqual(tr3.stats.gse2.get('calper'),
                         tr1.stats.gse2.get('calper'))
        self.assertEqual(tr3.stats.gse2.get('calib'),
                         tr1.stats.gse2.get('calib'))
        N.testing.assert_equal(tr3.data, tr1.data)

    def test_writeIntegersViaObsPy(self):
        """
        Write file test via L{obspy.Trace}.
        """
        tempfile = 'temp2.gse2'
        npts = 1000
        # data cloud of integers - float won't work!
        data = N.random.randint(-1000, 1000, npts)
        stats = {'network': 'BW', 'station': 'TEST', 'location':'',
                 'channel': 'EHE', 'npts': npts, 'sampling_rate': 200.0}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
        tr = Trace(data=data, header=stats)
        tr.verify()
        st = Stream([tr])
        st.verify()
        # write
        st.write(tempfile, format="GSE2")
        # read again
        stream = read(tempfile)
        os.remove(tempfile)
        stream.verify()
        self.assertEquals(stream[0].data.tolist(), data.tolist())


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
