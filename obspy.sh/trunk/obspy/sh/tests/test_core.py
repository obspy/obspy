# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.sh.core import readASC, writeASC, isASC, isQ, readQ, writeQ
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
        Testing ASC file format. 
        """
        testfile = os.path.join(self.path, 'data', 'TEST_090101_0101.ASC')
        self.assertEqual(isASC(testfile), True)
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        self.assertEqual(isASC(testfile), False)

    def test_isQFile(self):
        """
        Testing Q header file format.
        """
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        self.assertEqual(isQ(testfile), True)
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QBN')
        self.assertEqual(isQ(testfile), False)
        testfile = os.path.join(self.path, 'data', 'TEST_090101_0101.ASC')
        self.assertEqual(isQ(testfile), False)

    def test_readSingleChannelASCFile(self):
        """
        Read ASC file test via obspy.sh.core.readASC.
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
        Helper function to verify stream from file 'data/QFILE-TEST*'.
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
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data, 5)
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
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data, 5)
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
        np.testing.assert_array_almost_equal(stream[2].data[0:4], data, 4)

    def test_readAndWriteMultiChannelASCFile(self):
        """
        Read and write ASC file via obspy.sh.core.readASC.
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
        Read and write ASC file test via obspy.core.
        """
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-ASC.ASC')
        # read original
        stream1 = read(origfile, format="SH_ASC")
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile().name
        stream1.write(tempfile, format="SH_ASC")
        # read again w/ auto detection
        stream2 = read(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        os.remove(tempfile)

    def test_readAndWriteMultiChannelQFile(self):
        """
        Read and write Q file via obspy.sh.core.readQ.
        """
        #1 - little endian (PC)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST.QHD')
        # read original
        stream1 = readQ(origfile)
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile(suffix='.QHD').name
        writeQ(stream1, tempfile)
        # read again
        stream2 = readQ(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        # remove binary file too (dynamically created)
        os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        os.remove(tempfile)
        #2 - big endian (SUN)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        # read original
        stream1 = readQ(origfile, byteorder=">")
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile(suffix='.QHD').name
        writeQ(stream1, tempfile, byteorder=">")
        # read again
        stream2 = readQ(tempfile, byteorder=">")
        stream2.verify()
        self._compareStream(stream2)
        # remove binary file too (dynamically created)
        os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        os.remove(tempfile)

    def test_readAndWriteMultiChannelQFileViaObsPy(self):
        """
        Read and write Q file test via obspy.core.
        """
        #1 - little endian (PC)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST.QHD')
        # read original
        stream1 = read(origfile, format="Q")
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile(suffix='.QHD').name
        stream1.write(tempfile, format="Q")
        # read again w/ auto detection
        stream2 = read(tempfile)
        stream2.verify()
        self._compareStream(stream2)
        # remove binary file too (dynamically created)
        os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        os.remove(tempfile)
        #2 - big endian (SUN)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        # read original
        stream1 = read(origfile, format="Q", byteorder=">")
        stream1.verify()
        self._compareStream(stream1)
        # write
        tempfile = NamedTemporaryFile(suffix='.QHD').name
        stream1.write(tempfile, format="Q", byteorder=">")
        # read again w/ auto detection
        stream2 = read(tempfile, byteorder=">")
        stream2.verify()
        self._compareStream(stream2)
        # remove binary file too (dynamically created)
        os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        os.remove(tempfile)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
