# -*- coding: utf-8 -*-

from obspy.core import Trace, read
from obspy.mseed.core import readMSEED
import inspect
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    """

    def setUp(self):
        #Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(path, 'data', 'test.mseed')
        self.gapfile = os.path.join(path, 'data', 'gaps.mseed')

    def tearDown(self):
        pass

    def test_readFileViaLibMSEED(self):
        """
        Read file test via L{obspy.mseed.core.readMSEED}.
        """
        testdata = [2787, 2776, 2774, 2780, 2783]
        tr = readMSEED(self.file)
        self.assertEqual(tr[0].stats.network, 'NL')
        self.assertEqual(tr[0].stats['station'], 'HGN')
        self.assertEqual(tr[0].stats.get('location'), '00')
        self.assertEqual(tr[0].stats.npts, 11947)
        self.assertEqual(tr[0].stats['sampling_rate'], 40.0)
        self.assertEqual(tr[0].stats.get('channel'), 'BHZ')
        for _i in xrange(5):
            self.assertEqual(tr[0].data[_i], testdata[_i])

    def test_writeFileViaLibMSEED(self):
        """
        Write file test via L{obspy.Trace}.
        """
        pass

    def test_readFileViaObsPy(self):
        """
        Read file test via L{obspy.core.Trace}
        """
        testdata = [2787, 2776, 2774, 2780, 2783]
        # without given format -> auto detect file format
        tr = read(self.file)
        self.assertEqual(tr[0].stats.network, 'NL')
        self.assertEqual(tr[0].stats['station'], 'HGN')
        self.assertEqual(tr[0].stats.npts, 11947)
        for _i in xrange(5):
            self.assertEqual(tr[0].data[_i], testdata[_i])
        # with given format
        tr = read(self.file, format='MSEED')
        self.assertEqual(tr[0].stats.get('location'), '00')
        self.assertEqual(tr[0].stats.get('channel'), 'BHZ')
        self.assertEqual(tr[0].stats['sampling_rate'], 40.0)
        for _i in xrange(5):
            self.assertEqual(tr[0].data[_i], testdata[_i])

    def test_readFileViaObsPyStream(self):
        """
        Read file test via L{obspy.Stream}
        
        Only a very short test. Still needs to be extended.
        """
        # without given format -> autodetect using extension
        st = read(self.gapfile)
        self.assertEqual(4, len(st.traces))
        for _i in st.traces:
            self.assertEqual(True, isinstance(_i, Trace))

    def test_writeFileViaObsPy(self):
        """
        Write file test via L{obspy.Trace}.
        """
        pass

    def test_setStats(self):
        """
        Tests related to issue #4.
        """
        st = read(self.file)
        st += st
        # change stats attributes
        st[0].stats.station = 'AAA'
        st[1].stats['station'] = 'BBB'
        self.assertEquals(st[0].stats.station, 'AAA')
        self.assertEquals(st[0].stats['station'], 'AAA')
        self.assertEquals(st[1].stats['station'], 'BBB')
        self.assertEquals(st[1].stats.station, 'BBB')


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
