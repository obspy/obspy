# -*- coding: utf-8 -*-
"""
The obspy.io.segy Seismic Unix test suite.
"""
import io
import os
import unittest

import numpy as np

import obspy
from obspy.core.util import NamedTemporaryFile
from obspy.io.segy.segy import SEGYTraceReadingError, _read_su, iread_su


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

    def test_read_and_write_su(self):
        """
        Reading and writing a SU file should not change it.
        """
        file = os.path.join(self.path, '1.su_first_trace')
        # Read the original file once.
        with open(file, 'rb') as f:
            org_data = f.read()
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            # Read the SU file.
            su = _read_su(file)
            # Write it.
            su.write(outfile)
            with open(outfile, 'rb') as f:
                new_data = f.read()
        # Should be identical!
        self.assertEqual(org_data, new_data)

    def test_enforcing_byteorders_while_reading(self):
        """
        Tests whether or not enforcing the byte order while reading and writing
        does something and works at all. Using the wrong byte order will most
        likely raise an Exception.
        """
        # This file is little endian.
        file = os.path.join(self.path, '1.su_first_trace')
        # The following should both work.
        su = _read_su(file)
        self.assertEqual(su.endian, '<')
        su = _read_su(file, endian='<')
        self.assertEqual(su.endian, '<')
        # The following not because it will unpack the header and try to unpack
        # the number of data samples specified there which will of course not
        # correct.
        self.assertRaises(SEGYTraceReadingError, _read_su, file, endian='>')

    def test_reading_and_writing_different_byteorders(self):
        """
        Writing different byte orders should not change
        """
        # This file is little endian.
        file = os.path.join(self.path, '1.su_first_trace')
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            # The following should both work.
            su = _read_su(file)
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
            su2 = _read_su(outfile)
            self.assertEqual(su2.endian, '<')
            np.testing.assert_array_equal(data, su2.traces[0].data)
            # Write it big endian.
            su.write(outfile, endian='>')
            with open(outfile, 'rb') as f:
                new_data = f.read()
            self.assertFalse(org_data == new_data)
            su3 = _read_su(outfile)
        self.assertEqual(su3.endian, '>')
        np.testing.assert_array_equal(data, su3.traces[0].data)

    def test_unpacking_su_data(self):
        """
        Unpacks data and compares them to data unpacked by Madagascar.
        """
        # This file has the same data as 1.sgy_first_trace.
        file = os.path.join(self.path, '1.su_first_trace')
        data_file = os.path.join(self.path, '1.sgy_first_trace.npy')
        su = _read_su(file)
        data = su.traces[0].data
        # The data is written as integer so it is also converted to float32.
        correct_data = np.require(np.load(data_file).ravel(), np.float32)
        # Compare both.
        np.testing.assert_array_equal(correct_data, data)

    def test_read_bytes_io(self):
        """
        Tests reading from BytesIO instances.
        """
        # 1
        filename = os.path.join(self.path, '1.su_first_trace')
        with open(filename, 'rb') as fp:
            data = fp.read()
        st = _read_su(io.BytesIO(data))
        self.assertEqual(len(st.traces[0].data), 8000)

    def test_iterative_reading(self):
        """
        Tests iterative reading.
        """
        # Read normally.
        filename = os.path.join(self.path, '1.su_first_trace')
        st = obspy.read(filename, unpack_trace_headers=True)

        # Read iterative.
        ist = [_i for _i in iread_su(filename, unpack_headers=True)]

        del ist[0].stats.su.data_encoding

        self.assertEqual(st.traces, ist)
