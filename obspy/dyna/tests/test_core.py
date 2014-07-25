# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.sh.core import readDYNA, writeDYNA, isDYNA
import numpy as np
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_read13000pts(self):
        """
        Testing reading DYNA file with (about) 9000 pts.
        """
        testfile = os.path.join(self.path, 'data', 'IT.ARL..HGE.D.20140120.071240.X.ACC.ASC')
        # read
        stream = readDYNA(testfile)
        stream.verify()
        self.assertEqual(len(stream), 8997)

    def test_isDYNAFile(self):
        """
        Testing DYNA file format.
        """
        testfile = os.path.join(self.path, 'data', 'IT.ARL..HGE.D.20140120.071240.X.ACC.ASC')
        self.assertEqual(isDYNA(testfile), True)

    def _compareStream(self, stream):
        """
        Helper function to verify stream from file 'data/IT.ARL..HGE.D.20140120.071240.X.ACC.ASC'.
        """
        self.assertEqual(stream[0].stats.delta, 5.000000e-03)
        self.assertEqual(stream[0].stats.npts, 8997)
#        self.assertEqual(stream[0].stats.sh.COMMENT,
#                         'TEST TRACE IN QFILE #1')215717
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime(2014, 1, 20, 7, 12, 30))
        self.assertEqual(stream[0].stats.channel, 'E')
        self.assertEqual(stream[0].stats.station, 'ARL')
        # check last 4 samples
        data = [-0.032737, -0.037417, -0.030865, -0.021271]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data, 5)


    def test_readAndWriteDYNAFile(self):
        """
        Read and write DYNA file via obspy.sh.core.readDYNA.
        """
        origfile = os.path.join(self.path, 'data', 'IT.ARL..HGE.D.20140120.071240.X.ACC.ASC')
        # read original
        stream1 = readDYNA(origfile)
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile().name
        writeDYNA(stream1, tempfile)
        # read both files and compare the content
        text1 = open(origfile, 'rb').read()
        text2 = open(tempfile, 'rb').read()
        self.assertEquals(text1, text2)
        # read again
        stream2 = readDYNA(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        os.remove(tempfile)

    def test_readAndWriteDYNAFileViaObsPy(self):
        """
        Read and write ASC file test via obspy.core.
        """
        origfile = os.path.join(self.path, 'data', 'IT.ARL..HGE.D.20140120.071240.X.ACC.ASC')
        # read original
        stream1 = read(origfile, format="DYNA")
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile().name
        stream1.write(tempfile, format="DYNA")
        # read again w/ auto detection
        stream2 = read(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        os.remove(tempfile)

def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
