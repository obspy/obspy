# -*- coding: utf-8 -*-

from obspy.core import read
import inspect
import os
import time
import unittest


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
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.file = os.path.join(self.path, 'data',
                                 'BW.RJOB.__.EHZ.D.2009.056')

    def tearDown(self):
        pass

    def test_readMemory(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        for _ in xrange(200):
            start = time.time()
            stream = read(self.file)
            stream.verify()
            tr = stream[0]
            self.assertAlmostEqual(tr.data.mean(), 201.55502749647832)
            self.assertEqual(tr.stats.network, 'BW')
            self.assertEqual(tr.stats['station'], 'RJOB')
            self.assertEqual(tr.stats.get('npts'), 9380474)
            print "%.2fs" % (time.time() - start),


def suite():
    return unittest.makeSuite(MemoryTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
