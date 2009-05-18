# -*- coding: utf-8 -*-

from obspy.mseed import MSEEDTrace
from obspy import Trace, Stream
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
        self.gapfile =  os.path.join(path, 'data', 'gaps.mseed')
    
    def tearDown(self):
        pass
    
    def test_readFileViaLibMSEED(self):
        """
        Read file test via L{obspy.mseed.MSEEDTrace}.
        """
        testdata = [2787, 2776, 2774, 2780, 2783]
        tr = MSEEDTrace()
        tr.read(self.file)
        self.assertEqual(tr.stats.network, 'NL')
        self.assertEqual(tr.stats['station'], 'HGN')
        self.assertEqual(tr.stats.get('location'), '00')
        self.assertEqual(tr.stats.npts, 11947)
        self.assertEqual(tr.stats['sampling_rate'], 40.0)
        self.assertEqual(tr.stats.get('channel'), 'BHZ')
        for _i in xrange(5):
            self.assertEqual(tr.data[_i], testdata[_i])
    
    def test_writeFileViaLibMSEED(self):
        """
        Write file test via L{obspy.Trace}.
        """
        pass
    
    def test_readFileViaObsPy(self):
        """
        Read file test via L{obspy.Trace}
        """
        testdata = [2787, 2776, 2774, 2780, 2783]
        tr = Trace()
        # without given format -> autodetect using extension
        tr.read(self.file)
        self.assertEqual(tr.stats.network, 'NL')
        self.assertEqual(tr.stats['station'], 'HGN')
        for _i in xrange(5):
            self.assertEqual(tr.data[_i], testdata[_i])
        # with given format
        tr.read(self.file, format='MSEED')
        self.assertEqual(tr.stats.get('location'), '00')
        self.assertEqual(tr.stats.get('channel'), 'BHZ')
        for _i in xrange(5):
            self.assertEqual(tr.data[_i], testdata[_i])
        # with direct read call
        tr.readMSEED(self.file)
        self.assertEqual(tr.stats.npts, 11947)
        self.assertEqual(tr.stats['sampling_rate'], 40.0)
        for _i in xrange(5):
            self.assertEqual(tr.data[_i], testdata[_i])
            
    def test_readFileViaObsPyStream(self):
        """
        Read file test via L{obspy.Stream}
        
        Only a very short test. Still needs to be extended.
        """
        st = Stream()
        # without given format -> autodetect using extension
        st.read(self.gapfile)
        self.assertEqual(4, len(st.traces))
        for _i in st.traces:
            self.assertEqual(True, isinstance(_i, MSEEDTrace))
    
    def test_writeFileViaObsPy(self):
        """
        Write file test via L{obspy.Trace}.
        """
        pass


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
