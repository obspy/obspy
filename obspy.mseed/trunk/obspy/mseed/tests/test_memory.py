# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.core.util import libc
from obspy.mseed.core import readMSEED
from obspy.mseed import libmseed
import numpy as N, ctypes as C
import inspect, os, unittest


class MemoryTestCase(unittest.TestCase):
    """
    Test case of freeing memory in libmseed. All tests use 100 iterations.

    The memory allocated by libmseed is on the on hand the
    obspy.mseed.header.MSTraceGroup which defines the structure but does
    not allocate the memory for the MSTraceGroup.datasamples. This is done
    in libmseed by ms_readtraces using malloc. Thus in order to free this
    memory the counterpart of malloc called free needs to be used. You
    cannot use mst_free as part of the memory was allocated by python.

    test_readMemory:    Uses the stream class and iterates over 100
                        readings. The allocated memory in trace.data
                        is freed like in test_readMSTraces1 by the __del__
                        method of trace.
    test_readMSTraces1: Shows the current way how to free memory, this is
                        now done when a obspy.trace is deleted
    test_readMSTraces2: Delete the memory and access the data later. Of
                        course this results in a 'Segmentation Fault'
    test_readMSTraces3: This happens when you do NOT free the memory.
                        libmseed will exit with the error: 
                        'Error: mst_addmsr(): Cannot allocate memory'

    Run the tests by one of the following commands:
    $ python mseed/tests/test_memory.py MemoryTestCase.test_readMemory
    $ python mseed/tests/test_memory.py MemoryTestCase.test_readMSTraces1
    $ python mseed/tests/test_memory.py MemoryTestCase.test_readMSTraces2
    EXITS WITH Segmentation fault
    $ python mseed/tests/test_memory.py MemoryTestCase.test_readMSTraces3
    EXITS WITH Error: mst_addmsr(): Cannot allocate memory
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data', 'BW.BGLD..EHE.D.2008.001')

    def tearDown(self):
        pass

    def test_readMemory(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        for i in xrange(100):
            stream = read(self.file)
            stream.verify()
            for tr in stream:
                self.assertAlmostEqual(tr.data.mean(),-393.66969706930229)
                self.assertEqual(tr.stats.network, 'BW')
                self.assertEqual(tr.stats['station'], 'BGLD')
                self.assertEqual(tr.stats.get('npts'), 17280322)
            print '.',

    def test_readMSTraces1(self):
        """
        Memory is freed by libc.so.6, therefor no allocation problems
        """
        mseed = libmseed()
        for i in xrange(100):
            trace_list = mseed.readMSTraces(self.file)
            for trace in trace_list:
                print trace[1].max(),
                libc.free(trace[2])

    def test_readMSTraces2(self):
        """
        Memory is freed by libc.so.6, but accessed afterwards:
        'Segmentation fault'
        """
        mseed = libmseed()
        for i in xrange(100):
            trace_list = mseed.readMSTraces(self.file)
            for trace in trace_list:
                libc.free(trace[2])
                print self.__doc__
                print trace[1].max(),

    def test_readMSTraces3(self):
        """
        Memory is NOT freed, test exits on iteration ~21 with::
        'Error: mst_addmsr(): Cannot allocate memory'
        """
        mseed = libmseed()
        for i in xrange(100):
            trace_list = mseed.readMSTraces(self.file)
            for trace in trace_list:
                _max = trace[1].max()
                if i == 10:
                    print self.__doc__
            print '.',

def suite():
    return unittest.makeSuite(MemoryTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
