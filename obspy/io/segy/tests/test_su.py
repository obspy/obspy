# -*- coding: utf-8 -*-
"""
The obspy.io.segy Seismic Unix test suite.
"""
import io

import numpy as np

import obspy
from obspy.core.util import NamedTemporaryFile
from obspy.io.segy.segy import SEGYTraceReadingError, _read_su, iread_su
import pytest


class TestSU():
    """
    Test cases for SU reading and writing.

    Since the Seismic Unix format is a subset of the SEG Y file format a lot of
    the SEG Y tests cover certain aspects of the SU format and ensure that the
    SU implementation is working correctly.
    """
    def test_read_and_write_su(self, testdata):
        """
        Reading and writing a SU file should not change it.
        """
        file = testdata['1.su_first_trace']
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
        assert org_data == new_data

    def test_enforcing_byteorders_while_reading(self, testdata):
        """
        Tests whether or not enforcing the byte order while reading and writing
        does something and works at all. Using the wrong byte order will most
        likely raise an Exception.
        """
        # This file is little endian.
        file = testdata['1.su_first_trace']
        # The following should both work.
        su = _read_su(file)
        assert su.endian == '<'
        su = _read_su(file, endian='<')
        assert su.endian == '<'
        # The following not because it will unpack the header and try to unpack
        # the number of data samples specified there which will of course not
        # correct.
        with pytest.raises(SEGYTraceReadingError):
            _read_su(file, endian='>')

    def test_reading_and_writing_different_byteorders(self, testdata):
        """
        Writing different byte orders should not change
        """
        # This file is little endian.
        file = testdata['1.su_first_trace']
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            # The following should both work.
            su = _read_su(file)
            data = su.traces[0].data
            # Also read the original file.
            with open(file, 'rb') as f:
                org_data = f.read()
            assert su.endian == '<'
            # Write it little endian.
            su.write(outfile, endian='<')
            with open(outfile, 'rb') as f:
                new_data = f.read()
            assert org_data == new_data
            su2 = _read_su(outfile)
            assert su2.endian == '<'
            np.testing.assert_array_equal(data, su2.traces[0].data)
            # Write it big endian.
            su.write(outfile, endian='>')
            with open(outfile, 'rb') as f:
                new_data = f.read()
            assert not (org_data == new_data)
            su3 = _read_su(outfile)
        assert su3.endian == '>'
        np.testing.assert_array_equal(data, su3.traces[0].data)

    def test_unpacking_su_data(self, testdata):
        """
        Unpacks data and compares them to data unpacked by Madagascar.
        """
        # This file has the same data as 1.sgy_first_trace.
        file = testdata['1.su_first_trace']
        data_file = testdata['1.sgy_first_trace.npy']
        su = _read_su(file)
        data = su.traces[0].data
        # The data is written as integer so it is also converted to float32.
        correct_data = np.require(np.load(data_file).ravel(), np.float32)
        # Compare both.
        np.testing.assert_array_equal(correct_data, data)

    def test_read_bytes_io(self, testdata):
        """
        Tests reading from BytesIO instances.
        """
        # 1
        filename = testdata['1.su_first_trace']
        with open(filename, 'rb') as fp:
            data = fp.read()
        st = _read_su(io.BytesIO(data))
        assert len(st.traces[0].data) == 8000

    def test_iterative_reading(self, testdata):
        """
        Tests iterative reading.
        """
        # Read normally.
        filename = testdata['1.su_first_trace']
        st = obspy.read(filename, unpack_trace_headers=True)

        # Read iterative.
        ist = [_i for _i in iread_su(filename, unpack_headers=True)]

        del ist[0].stats.su.data_encoding

        assert st.traces == ist
