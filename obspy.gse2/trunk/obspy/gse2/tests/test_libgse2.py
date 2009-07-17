#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The libgse2 test suite.
"""

from obspy.gse2 import libgse2
from obspy.core import UTCDateTime
import inspect, os, unittest
import numpy as N

class LibGSE2TestCase(unittest.TestCase):
    """
    Test cases for libgse2.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')

    def tearDown(self):
        pass

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
        header, data = libgse2.read(gse2file)
        self.assertEqual('RJOB ', header['station'])
        self.assertEqual('  Z', header['channel'])
        self.assertEqual(200, header['samp_rate'])
        self.assertEqual('20050831023349.850', "%04d%02d%02d%02d%02d%06.3f" % (
            header['d_year'],
            header['d_mon'],
            header['d_day'],
            header['t_hour'],
            header['t_min'],
            header['t_sec'])
        )
        for i in range(len(datalist) - 1):
            self.assertEqual(datalist[i] - datalist[i + 1], data[i] - data[i + 1])
        #from pylab import plot,array,show;plot(array(data));show()

    def test_readAnWrite(self):
        """
        Writes, reads and compares files created via libgse2.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        header, data = libgse2.read(gse2file)
        temp_file = os.path.join(self.path, 'tmp.gse2')
        libgse2.write(header, data.copy(), temp_file)
        newheader, newdata = libgse2.read(temp_file)
        self.assertEqual(header, newheader)
        N.testing.assert_equal(data, newdata)
        os.remove(temp_file)

    def test_readHeaderInfo(self):
        """
        Reads and compares header info from the first record.
        
        The values can be read from the filename.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        header = libgse2.readHead(gse2file)
        self.assertEqual('RNON ', header['station'])
        self.assertEqual('  Z', header['channel'])
        self.assertEqual(200, header['samp_rate'])
        self.assertEqual('20040609200559.850', "%04d%02d%02d%02d%02d%06.3f" % (
            header['d_year'],
            header['d_mon'],
            header['d_day'],
            header['t_hour'],
            header['t_min'],
            header['t_sec'])
        )

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        # get the start- and end time
        times = libgse2.getStartAndEndTime(gse2file)
        self.assertEqual(UTCDateTime(2004, 6, 9, 20, 5, 59, 849998), times[0])
        self.assertEqual(UTCDateTime(2004, 6, 9, 20, 6, 59, 849998), times[1])
        self.assertEqual(1086811559.849998, times[2])
        self.assertEqual(1086811619.849998, times[3])

    def test_isWidi2(self):
        """
        See if first 4 characters are WID2, if not raise type error.
        """
        self.assertRaises(TypeError, libgse2.read, __file__)
        self.assertRaises(TypeError, libgse2.getStartAndEndTime, __file__)
        self.assertRaises(TypeError, libgse2.readHead, __file__)


    def test_maxvalueExceeded(self):
        """
        Test that exception is raised when data values exceed the maximum
        of 2^26
        """
        testfile = os.path.join(self.path, 'tmp.gse2')
        data = N.array([2 ** 26 + 1])
        header = {}
        header['samp_rate'] = 200
        header['n_samps'] = 1
        header['datatype'] = 'CM6'
        self.assertRaises(OverflowError, libgse2.write, header, data, testfile)

    def test_arrayNotNumpy(self):
        """
        Test that exception is raised when data are not of type int32 numpy array
        """
        testfile = os.path.join(self.path, 'tmp.gse2')
        data = [2, 26, 1]
        header = {}
        header['samp_rate'] = 200
        header['n_samps'] = 1
        header['datatype'] = 'CM6'
        self.assertRaises(AssertionError, libgse2.write, header, data, testfile)
        data = N.array([2, 26, 1], dtype='f')
        self.assertRaises(AssertionError, libgse2.write, header, data, testfile)

def suite():
    return unittest.makeSuite(LibGSE2TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
