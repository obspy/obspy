#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The libgse2 test suite.
"""
from ctypes import ArgumentError
from obspy import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.gse2 import libgse2
from obspy.gse2.libgse2 import ChksumError
import numpy as np
import os
import unittest


class LibGSE2TestCase(unittest.TestCase):
    """
    Test cases for libgse2.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_read(self):
        """
        Compares waveform data read by libgse2 with an ASCII dump.

        Checks the first 13 datasamples when reading loc_RJOB20050831023349.z.
        The values are assumed to be correct. The values were created using
        getevents. Only checks relative values.
        """
        gse2file = os.path.join(self.path, 'loc_RJOB20050831023349.z')
        # list of known data samples
        datalist = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        f = open(gse2file, 'rb')
        header, data = libgse2.read(f, verify_chksum=True)
        self.assertEqual('RJOB ', header['station'])
        self.assertEqual('  Z', header['channel'])
        self.assertEqual(200.0, header['samp_rate'])
        self.assertEqual('20050831023349.850', "%04d%02d%02d%02d%02d%06.3f" % (
            header['d_year'],
            header['d_mon'],
            header['d_day'],
            header['t_hour'],
            header['t_min'],
            header['t_sec']))
        self.assertAlmostEquals(9.49e-02, header['calib'])
        self.assertEqual(1.0, header['calper'])
        self.assertEqual(-1.0, header['vang'])
        self.assertEqual(-1.0, header['hang'])
        self.assertEqual(data[0:13].tolist(), datalist)
        f.close()

    def test_readWithWrongChecksum(self):
        """
        """
        # read original file
        gse2file = os.path.join(self.path,
                                'loc_RJOB20050831023349.z.wrong_chksum')
        # should fail
        fp = open(gse2file, 'rb')
        self.assertRaises(ChksumError, libgse2.read, fp, verify_chksum=True)
        # should not fail
        fp.seek(0)
        _trl = libgse2.read(fp, verify_chksum=False)
        fp.close()

    def test_readAndWrite(self):
        """
        Writes, reads and compares files created via libgse2.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        f = open(gse2file, 'rb')
        header, data = libgse2.read(f)
        f.close()
        tmp_file = NamedTemporaryFile().name
        f = open(tmp_file, 'wb')
        libgse2.write(header, data, f)
        f.close()
        newheader, newdata = libgse2.read(open(tmp_file, 'rb'))
        self.assertEqual(header, newheader)
        np.testing.assert_equal(data, newdata)
        os.remove(tmp_file)

    def test_readHeaderInfo(self):
        """
        Reads and compares header info from the first record.

        The values can be read from the filename.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        header = libgse2.readHead(open(gse2file, 'rb'))
        self.assertEqual('RNON ', header['station'])
        self.assertEqual('  Z', header['channel'])
        self.assertEqual(200, header['samp_rate'])
        self.assertEqual('20040609200559.850', "%04d%02d%02d%02d%02d%06.3f" % (
            header['d_year'],
            header['d_mon'],
            header['d_day'],
            header['t_hour'],
            header['t_min'],
            header['t_sec']))

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        # get the start- and end time
        times = libgse2.getStartAndEndTime(open(gse2file, 'rb'))
        self.assertEqual(UTCDateTime(2004, 6, 9, 20, 5, 59, 849998), times[0])
        self.assertEqual(UTCDateTime(2004, 6, 9, 20, 6, 59, 849998), times[1])
        self.assertEqual(1086811559.849998, times[2])
        self.assertEqual(1086811619.849998, times[3])

    def test_isWidi2(self):
        """
        See if first 4 characters are WID2, if not raise type error.
        """
        f = open(os.path.join(self.path, 'loc_RNON20040609200559.z'), 'rb')
        pos = f.tell()
        self.assertEqual(None, libgse2.isGse2(f))
        self.assertEqual(pos, f.tell())
        f.seek(10)
        self.assertRaises(TypeError, libgse2.isGse2, f)
        self.assertEqual(10, f.tell())

    def test_maxValueExceeded(self):
        """
        Test that exception is raised when data values exceed the maximum
        of 2^26
        """
        testfile = NamedTemporaryFile().name
        data = np.array([2 ** 26 + 1], dtype='int32')
        header = {}
        header['samp_rate'] = 200
        header['n_samps'] = 1
        header['datatype'] = 'CM6'
        f = open(testfile, 'wb')
        self.assertRaises(OverflowError, libgse2.write, header, data, f)
        f.close()
        os.remove(testfile)

    def test_arrayNotNumpy(self):
        """
        Test if exception is raised when data are not of type int32 NumPy array
        """
        testfile = NamedTemporaryFile().name
        data = [2, 26, 1]
        header = {}
        header['samp_rate'] = 200
        header['n_samps'] = 1
        header['datatype'] = 'CM6'
        f = open(testfile, 'wb')
        self.assertRaises(ArgumentError, libgse2.write, header, data,
                          testfile)
        f.close()
        f = open(testfile, 'wb')
        data = np.array([2, 26, 1], dtype='f')
        self.assertRaises(ArgumentError, libgse2.write, header, data,
                          testfile)
        f.close()
        os.remove(testfile)

    def test_CHK2InCM6(self):
        """
        Tests a file which contains the "CHK2" string in the CM6 encoded
        string (line 13 of twiceCHK2.gse2).
        """
        f = open(os.path.join(self.path, 'twiceCHK2.gse2'), 'rb')
        header, data = libgse2.read(f, verify_chksum=True)
        self.assertEqual(header['n_samps'], 750)
        np.testing.assert_array_equal(data[-4:],
                                      np.array([-139, -153, -169, -156]))


def suite():
    return unittest.makeSuite(LibGSE2TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
