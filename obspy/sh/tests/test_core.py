# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

import numpy as np

from obspy import Trace, UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.sh.core import (STANDARD_ASC_HEADERS, isASC, isQ, readASC, readQ,
                           writeASC, writeQ)


class CoreTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_read101Traces(self):
        """
        Testing reading Q file with more than 100 traces.
        """
        testfile = os.path.join(self.path, 'data', '101.QHD')
        # read
        stream = readQ(testfile)
        stream.verify()
        self.assertEqual(len(stream), 101)

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
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            writeASC(stream1, tempfile, STANDARD_ASC_HEADERS + ['COMMENT'])
            # read both files and compare the content
            with open(origfile, 'rt') as f:
                text1 = f.readlines()
            with open(tempfile, 'rt') as f:
                text2 = f.readlines()
            self.assertEqual(text1, text2)
            # read again
            stream2 = readASC(tempfile)
            stream2.verify()
            self._compareStream(stream2)

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
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            hd = STANDARD_ASC_HEADERS + ['COMMENT']
            stream1.write(tempfile, format="SH_ASC", included_headers=hd)
            # read again w/ auto detection
            stream2 = read(tempfile)
            stream2.verify()
            self._compareStream(stream2)

    def test_readAndWriteMultiChannelQFile(self):
        """
        Read and write Q file via obspy.sh.core.readQ.
        """
        # 1 - little endian (PC)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST.QHD')
        # read original
        stream1 = readQ(origfile)
        stream1.verify()
        self._compareStream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            writeQ(stream1, tempfile, append=False)
            # read again
            stream2 = readQ(tempfile)
            stream2.verify()
            self._compareStream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        # 2 - big endian (SUN)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        # read original
        stream1 = readQ(origfile, byteorder=">")
        stream1.verify()
        self._compareStream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            writeQ(stream1, tempfile, byteorder=">", append=False)
            # read again
            stream2 = readQ(tempfile, byteorder=">")
            stream2.verify()
            self._compareStream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')

    def test_readAndWriteMultiChannelQFileViaObsPy(self):
        """
        Read and write Q file test via obspy.core.
        """
        # 1 - little endian (PC)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST.QHD')
        # read original
        stream1 = read(origfile, format="Q")
        stream1.verify()
        self._compareStream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            stream1.write(tempfile, format="Q", append=False)
            # read again w/ auto detection
            stream2 = read(tempfile)
            stream2.verify()
            self._compareStream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')
        # 2 - big endian (SUN)
        origfile = os.path.join(self.path, 'data', 'QFILE-TEST-SUN.QHD')
        # read original
        stream1 = read(origfile, format="Q", byteorder=">")
        stream1.verify()
        self._compareStream(stream1)
        # write
        with NamedTemporaryFile(suffix='.QHD') as tf:
            tempfile = tf.name
            stream1.write(tempfile, format="Q", byteorder=">", append=False)
            # read again w/ auto detection
            stream2 = read(tempfile, byteorder=">")
            stream2.verify()
            self._compareStream(stream2)
            # remove binary file too (dynamically created)
            os.remove(os.path.splitext(tempfile)[0] + '.QBN')

    def test_skipASClines(self):
        testfile = os.path.join(self.path, 'data', 'QFILE-TEST-ASC.ASC')
        # read
        stream = readASC(testfile, skip=100, delta=0.1, length=2)
        stream.verify()
        # skip force one trace only
        self.assertEqual(len(stream), 1)
        # headers
        self.assertEqual(stream[0].stats.delta, 1.000000e-01)
        self.assertEqual(stream[0].stats.npts, 2)
        # check samples
        self.assertEqual(len(stream[0].data), 2)
        self.assertAlmostEqual(stream[0].data[0], 111.7009, 4)
        self.assertAlmostEqual(stream[0].data[1], 119.5831, 4)

    def test_writeSmallTrace(self):
        """
        Tests writing Traces containing 0, 1 or 2 samples only.
        """
        for format in ['SH_ASC', 'Q']:
            for num in range(0, 4):
                tr = Trace(data=np.arange(num))
                with NamedTemporaryFile() as tf:
                    tempfile = tf.name
                    if format == 'Q':
                        tempfile += '.QHD'
                    tr.write(tempfile, format=format)
                    # test results
                    with warnings.catch_warnings() as _:  # NOQA
                        warnings.simplefilter("ignore")
                        st = read(tempfile, format=format)
                    self.assertEqual(len(st), 1)
                    self.assertEqual(len(st[0]), num)
                    # Q files consist of two files - deleting additional file
                    if format == 'Q':
                        os.remove(tempfile[:-4] + '.QBN')
                        os.remove(tempfile[:-4] + '.QHD')


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
