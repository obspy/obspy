# -*- coding: utf-8 -*-
"""
The obspy.segy Seismic Unix test suite.
"""

from __future__ import with_statement
from obspy.core.util import NamedTemporaryFile
from obspy.segy.segy import readSU, SEGYTraceReadingError
from StringIO import StringIO
import numpy as np
import os
import unittest


class SUTestCase(unittest.TestCase):
    """
    Test cases for SU reading and writing.

    Since the Seismic Unix format is a subset of the SEG Y file format a lot of
    the SEG Y tests cover certain aspects of the SU format and ensure that the
    SU implementation is working correctly.
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(__file__)
        self.path = os.path.join(self.dir, 'data')

    def test_readAndWriteSU(self):
        """
        Reading and writing a SU file should not change it.
        """
        file = os.path.join(self.path, '1.su_first_trace')
        # Read the original file once.
        with open(file, 'rb') as f:
            org_data = f.read()
        outfile = NamedTemporaryFile().name
        # Read the SU file.
        su = readSU(file)
        # Write it.
        su.write(outfile)
        with open(outfile, 'rb') as f:
            new_data = f.read()
        os.remove(outfile)
        # Should be identical!
        self.assertEqual(org_data, new_data)

    def test_enforcingByteordersWhileReading(self):
        """
        Tests whether or not enforcing the byteorder while reading and writing
        does something and works at all. Using the wrong byteorder will most
        likely raise an Exception.
        """
        # This file is little endian.
        file = os.path.join(self.path, '1.su_first_trace')
        # The following should both work.
        su = readSU(file)
        self.assertEqual(su.endian, '<')
        su = readSU(file, endian='<')
        self.assertEqual(su.endian, '<')
        # The following not because it will unpack the header and try to unpack
        # the number of data samples specified there which will of course not
        # correct.
        self.assertRaises(SEGYTraceReadingError, readSU, file, endian='>')

    def test_readingAndWritingDifferentByteorders(self):
        """
        Writing different byteorders should not change
        """
        # This file is little endian.
        file = os.path.join(self.path, '1.su_first_trace')
        outfile = NamedTemporaryFile().name
        # The following should both work.
        su = readSU(file)
        data = su.traces[0].data
        # Also read the original file.
        with open(file, 'rb') as f:
            org_data = f.read()
        self.assertEqual(su.endian, '<')
        # Write it little endian.
        su.write(outfile, endian='<')
        with open(outfile, 'rb') as f:
            new_data = f.read()
        self.assertEqual(org_data, new_data)
        su2 = readSU(outfile)
        self.assertEqual(su2.endian, '<')
        np.testing.assert_array_equal(data, su2.traces[0].data)
        os.remove(outfile)
        # Write it big endian.
        su.write(outfile, endian='>')
        with open(outfile, 'rb') as f:
            new_data = f.read()
        self.assertFalse(org_data == new_data)
        su3 = readSU(outfile)
        os.remove(outfile)
        self.assertEqual(su3.endian, '>')
        np.testing.assert_array_equal(data, su3.traces[0].data)

    def test_unpackingSUData(self):
        """
        Unpacks data and compares them to data unpacked by Madagascar.
        """
        # This file has the same data as 1.sgy_first_trace.
        file = os.path.join(self.path, '1.su_first_trace')
        data_file = os.path.join(self.path, '1.sgy_first_trace.npy')
        su = readSU(file)
        data = su.traces[0].data
        # The data is written as integer so it is also converted to float32.
        correct_data = np.require(np.load(data_file).ravel(), 'float32')
        # Compare both.
        np.testing.assert_array_equal(correct_data, data)

    def test_readStringIO(self):
        """
        Tests reading from StringIO instances.
        """
        # 1
        file = os.path.join(self.path, '1.su_first_trace')
        data = open(file, 'rb').read()
        st = readSU(StringIO(data))
        self.assertEqual(len(st.traces[0].data), 8000)


def suite():
    return unittest.makeSuite(SUTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
