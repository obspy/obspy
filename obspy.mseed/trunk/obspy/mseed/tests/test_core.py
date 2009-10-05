# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.mseed.core import readMSEED
import inspect
import numpy as N
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))

    def tearDown(self):
        pass

    def test_readFileViaLibMSEED(self):
        """
        Read file test via L{obspy.mseed.core.readMSEED}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        testdata = [2787, 2776, 2774, 2780, 2783]
        # read
        stream = readMSEED(testfile)
        stream._verify()
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(stream[0].stats.get('location'), '00')
        self.assertEqual(stream[0].stats.npts, 11947)
        self.assertEqual(stream[0].stats['sampling_rate'], 40.0)
        self.assertEqual(stream[0].stats.get('channel'), 'BHZ')
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])

    def test_readFileViaObsPy(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        testdata = [2787, 2776, 2774, 2780, 2783]
        # without given format -> auto detect file format
        stream = read(testfile)
        stream._verify()
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(stream[0].stats.npts, 11947)
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])
        # with given format
        stream = read(testfile, format='MSEED')
        stream._verify()
        self.assertEqual(stream[0].stats.get('location'), '00')
        self.assertEqual(stream[0].stats.get('channel'), 'BHZ')
        self.assertEqual(stream[0].stats['sampling_rate'], 40.0)
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])
        # file with gaps
        gapfile = os.path.join(self.path, 'data', 'gaps.mseed')
        # without given format -> autodetect using extension
        stream = read(gapfile)
        stream._verify()
        self.assertEqual(4, len(stream.traces))
        for _i in stream.traces:
            self.assertEqual(True, isinstance(_i, Trace))

    def test_readHeadFileViaObsPy(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        stream = read(testfile, headonly=True, format='MSEED')
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(str(stream[0].data), '[]')
        self.assertEqual(stream[0].stats.npts, 11947)
        #
        gapfile = os.path.join(self.path, 'data', 'gaps.mseed')
        # without given format -> autodetect using extension
        stream = read(gapfile, headonly=True)
        self.assertEqual(4, len(stream.traces))
        for _i in stream.traces:
            self.assertEqual(True, isinstance(_i, Trace))
            self.assertEqual(str(_i.data), '[]')

    def test_writeIntegersViaObsPy(self):
        """
        Write integer array via L{obspy.core.Stream}.
        """
        tempfile = 'temp1.mseed'
        npts = 1000
        # data array of integers - float won't work!
        data = N.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'',
                 'channel': 'EHE', 'npts': npts, 'sampling_rate': 200.0}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
        # write
        st.write(tempfile, format="MSEED")
        # read again
        stream = read(tempfile)
        os.remove(tempfile)
        stream._verify()
        self.assertEquals(stream[0].data.tolist(), data.tolist())

    def test_readWithWildCard(self):
        """
        Reads wildcard filenames.
        """
        files = os.path.join(self.path, 'data', 'BW.BGLD.__.EHE.D.2008.001.*')
        st = read(files)
        self.assertEquals(len(st), 4)
        st.merge()
        self.assertEquals(len(st), 1)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
