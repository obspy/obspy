# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.sh.core import readASC, writeASC, isASC, isQ
import inspect
import numpy as np
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

    def test_isASCFile(self):
        """
        """
        testfile = os.path.join(self.path, 'data', 'TEST_090101_0101.ASC')
        self.assertEqual(isASC(testfile), True)
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        self.assertEqual(isASC(testfile), False)

    def test_isQFile(self):
        """
        """
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        self.assertEqual(isQ(testfile), True)
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QBN')
        self.assertEqual(isQ(testfile), False)
        testfile = os.path.join(self.path, 'data', 'TEST_090101_0101.ASC')
        self.assertEqual(isQ(testfile), False)

    def test_readSingleChannelASCFile(self):
        """
        Read file test via obspy.sh.asc.readASC.
        """
        testfile = os.path.join(self.path, 'data', 'TEST_090101_0101.ASC')
        # read
        stream = readASC(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.delta, 5.000000e-02)
        self.assertEqual(stream[0].stats.npts, 1000)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime(2009, 1, 1, 1, 1, 1))
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        # check last 4 samples
        data = [2.176000e+01, 2.195485e+01, 2.213356e+01, 2.229618e+01]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def _compareStream(self, stream):
        """
        Helper function to verify stream from file 'data/QFILE-TEST-ASC.ASC'.
        """
        # channel 1
        self.assertEqual(stream[0].stats.delta, 5.000000e-02)
        self.assertEqual(stream[0].stats.npts, 801)
        self.assertEqual(stream[0].stats.sh.COMMENT,
                         'TEST TRACE IN QFILE #1')
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime(2009, 10, 1, 12, 46, 1))
        self.assertEqual(stream[0].stats.channel, 'BHN')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.calib, 1.500000e+00)
        # check last 4 samples
        data = [-4.070354e+01, -4.033876e+01, -3.995153e+01, -3.954230e+01]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # channel 2
        self.assertEqual(stream[1].stats.delta, 5.000000e-02)
        self.assertEqual(stream[1].stats.npts, 801)
        self.assertEqual(stream[1].stats.sh.COMMENT,
                         'TEST TRACE IN QFILE #2')
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime(2009, 10, 1, 12, 46, 1))
        self.assertEqual(stream[1].stats.channel, 'BHE')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.calib, 1.500000e+00)
        # check first 4 samples
        data = [-3.995153e+01, -4.033876e+01, -4.070354e+01, -4.104543e+01]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # channel 3
        self.assertEqual(stream[2].stats.delta, 1.000000e-02)
        self.assertEqual(stream[2].stats.npts, 4001)
        self.assertEqual(stream[2].stats.sh.COMMENT, '******')
        self.assertEqual(stream[2].stats.starttime,
                         UTCDateTime(2010, 1, 1, 1, 1, 5, 999000))
        self.assertEqual(stream[2].stats.channel, 'HHZ')
        self.assertEqual(stream[2].stats.station, 'WET')
        self.assertEqual(stream[2].stats.calib, 1.059300e+00)
        # check first 4 samples
        data = [4.449060e+02, 4.279572e+02, 4.120677e+02, 4.237200e+02]
        np.testing.assert_array_almost_equal(stream[2].data[0:4], data)

    def test_readAndWriteMultiChannelASCFile(self):
        """
        Read and write file via obspy.sh.asc.readASC.
        """
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-ASC.ASC')
        # read original
        stream1 = readASC(origfile)
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile().name
        writeASC(stream1, tempfile)
        # read both files and compare the content
        text1 = open(origfile, 'rb').read()
        text2 = open(tempfile, 'rb').read()
        self.assertEquals(text1, text2)
        # read again
        stream2 = readASC(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        os.remove(tempfile)

    def test_readAndWriteMultiChannelASCFileViaObsPy(self):
        """
        Read and write file test via obspy.core.
        """
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-ASC.ASC')
        # read original
        stream1 = read(origfile, "SH_ASC")
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile().name
        stream1.write(tempfile, format="SH_ASC")
        # read again
        stream2 = readASC(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        os.remove(tempfile)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
