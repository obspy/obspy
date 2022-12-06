#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The libgse2 test suite.
"""
import io
import os
import unittest
from ctypes import ArgumentError
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core.util import SuppressOutput, NamedTemporaryFile
from obspy.io.gse2 import libgse2
from obspy.io.gse2.libgse2 import (ChksumError, GSEUtiError, compile_sta2,
                                   parse_sta2)


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
        self.assertEqual('RJOB', header['station'])
        self.assertEqual('Z', header['channel'])
        self.assertEqual(200.0, header['sampling_rate'])
        self.assertEqual(UTCDateTime(2005, 8, 31, 2, 33, 49, 850000),
                         header['starttime'])
        self.assertAlmostEqual(9.49e-02, header['calib'])
        self.assertEqual(1.0, header['gse2']['calper'])
        self.assertEqual(-1.0, header['gse2']['vang'])
        self.assertEqual(-1.0, header['gse2']['hang'])
        self.assertEqual(data[0:13].tolist(), datalist)
        f.close()

    def test_read_with_wrong_checksum(self):
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
        libgse2.read(fp, verify_chksum=False)
        fp.close()

    def test_read_and_write(self):
        """
        Writes, reads and compares files created via libgse2.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        with open(gse2file, 'rb') as f:
            header, data = libgse2.read(f)
        with NamedTemporaryFile() as f:
            # raises "UserWarning: Bad value in GSE2 header field"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                libgse2.write(header, data, f)
            f.flush()
            with open(f.name, 'rb') as f2:
                newheader, newdata = libgse2.read(f2)
        self.assertEqual(header, newheader)
        np.testing.assert_equal(data, newdata)

    def test_bytes_io(self):
        """
        Checks that reading and writing works via BytesIO.
        """
        gse2file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        with open(gse2file, 'rb') as f:
            fin = io.BytesIO(f.read())
        header, data = libgse2.read(fin)
        # be sure something es actually read
        self.assertEqual(12000, header['npts'])
        self.assertEqual(1, data[-1])
        fout = io.BytesIO()
        # raises "UserWarning: Bad value in GSE2 header field"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            libgse2.write(header, data, fout)
        fout.seek(0)
        newheader, newdata = libgse2.read(fout)
        self.assertEqual(header, newheader)
        np.testing.assert_equal(data, newdata)

    def test_read_header(self):
        """
        Reads and compares header info from the first record.

        The values can be read from the filename.
        """
        gse2file = os.path.join(self.path, 'twiceCHK2.gse2')
        with open(gse2file, 'rb') as f:
            header = libgse2.read_header(f)
        self.assertEqual('RNHA', header['station'])
        self.assertEqual('EHN', header['channel'])
        self.assertEqual(200, header['sampling_rate'])
        self.assertEqual(750, header['npts'])
        self.assertEqual('M24', header['gse2']['instype'])
        self.assertEqual(UTCDateTime(2009, 5, 18, 6, 47, 20, 255000),
                         header['starttime'])

    def test_is_widi_2(self):
        """
        See if first 4 characters are WID2, if not raise type error.
        """
        filename = os.path.join(self.path, 'loc_RNON20040609200559.z')
        with open(filename, 'rb') as f:
            pos = f.tell()
            self.assertEqual(None, libgse2.is_gse2(f))
            self.assertEqual(pos, f.tell())
            f.seek(10)
            self.assertRaises(TypeError, libgse2.is_gse2, f)
            self.assertEqual(10, f.tell())

    def test_max_value_exceeded(self):
        """
        Test that exception is raised when data values exceed the maximum
        of 2^26
        """
        data = np.array([2 ** 26 + 1], dtype=np.int32)
        header = {}
        header['samp_rate'] = 200
        header['n_samps'] = 1
        header['datatype'] = 'CM6'
        with NamedTemporaryFile() as tf:
            testfile = tf.name
            with open(testfile, 'wb') as f:
                self.assertRaises(OverflowError, libgse2.write, header, data,
                                  f)

    def test_array_not_numpy(self):
        """
        Test if exception is raised when data are not of type int32 NumPy array
        """
        header = {}
        header['samp_rate'] = 200
        header['n_samps'] = 1
        header['datatype'] = 'CM6'
        with NamedTemporaryFile() as tf:
            testfile = tf.name
            data = [2, 26, 1]
            with open(testfile, 'wb') as f:
                self.assertRaises(ArgumentError, libgse2.write, header, data,
                                  f)
            data = np.array([2, 26, 1], dtype=np.float32)
            with open(testfile, 'wb') as f:
                self.assertRaises(ArgumentError, libgse2.write, header, data,
                                  f)

    def test_chk2_in_cm6(self):
        """
        Tests a file which contains the "CHK2" string in the CM6 encoded
        string (line 13 of twiceCHK2.gse2).
        """
        with open(os.path.join(self.path, 'twiceCHK2.gse2'), 'rb') as f:
            header, data = libgse2.read(f, verify_chksum=True)
        self.assertEqual(header['npts'], 750)
        np.testing.assert_array_equal(data[-4:],
                                      np.array([-139, -153, -169, -156]))

    def test_broken_head(self):
        """
        Tests that gse2 files with n_samps=0 will not end up with a
        segmentation fault
        """
        with open(os.path.join(self.path, 'broken_head.gse2'), 'rb') as f:
            self.assertRaises(ChksumError, libgse2.read, f)

    def test_no_dat2_null_pointer(self):
        """
        Checks that null pointers are returned correctly by read83 function
        of read. Error "decomp_6b: Neither DAT2 or DAT1 found!" is on
        purpose.
        """
        filename = os.path.join(self.path,
                                'loc_RJOB20050831023349_first100_dos.z')
        fout = io.BytesIO()
        with open(filename, 'rb') as fin:
            lines = (line for line in fin if not line.startswith(b'DAT2'))
            fout.write(b"".join(lines))
        fout.seek(0)
        with SuppressOutput():
            # omit C level error "decomp_6b: Neither DAT2 or DAT1 found!"
            self.assertRaises(GSEUtiError, libgse2.read, fout)

    def test_parse_sta2(self):
        """
        Tests parsing of STA2 lines on a collection of (modified) real world
        examples.
        """
        filename = os.path.join(self.path,
                                'STA2.testlines')
        filename2 = os.path.join(self.path,
                                 'STA2.testlines_out')
        results = [
            {'network': 'ABCD', 'lon': 12.12345, 'edepth': 0.0, 'elev': -290.0,
             'lat': 37.12345, 'coordsys': 'WGS-84'},
            {'network': 'ABCD', 'lon': 12.12345, 'edepth': 0.0, 'elev': -50.0,
             'lat': -37.12345, 'coordsys': 'WGS-84'},
            {'network': 'ABCD', 'lon': 2.12345, 'edepth': 0.0, 'elev': -2480.0,
             'lat': 7.1234, 'coordsys': 'WGS-84'},
            {'network': 'ABCD', 'lon': 2.1234, 'edepth': 0.0, 'elev': -2480.0,
             'lat': 37.12345, 'coordsys': 'WGS-84'},
            {'network': 'ABCD', 'lon': 2.123, 'edepth': 0.0, 'elev': -2480.0,
             'lat': -7.1234, 'coordsys': 'WGS-84'},
            {'network': 'ABCD', 'lon': -12.12345, 'edepth': 0.0, 'elev': 1.816,
             'lat': 36.12345, 'coordsys': 'WGS-84'},
            {'network': 'abcdef', 'lon': -112.12345, 'edepth': 0.002,
             'elev': 0.254, 'lat': 37.12345, 'coordsys': 'WGS84'},
            {'network': 'ABCD', 'lon': 12.12345, 'edepth': 0.0,
             'elev': -240000.0, 'lat': 37.12345, 'coordsys': 'WGS-84'},
            {'network': 'ABCD', 'lon': 1.12345, 'edepth': 1.234,
             'elev': -123.456, 'lat': 12.12345, 'coordsys': 'WGS-84'},
            {'network': '', 'lon': -999.0, 'edepth': -0.999, 'elev': -0.999,
             'lat': -99.0, 'coordsys': ''},
            {'network': '', 'lon': -999.0, 'edepth': -0.999, 'elev': -0.999,
             'lat': -99.0, 'coordsys': ''}]
        with open(filename) as fh:
            lines = fh.readlines()
        with open(filename2) as fh:
            lines2 = fh.readlines()
        for line, line2, expected in zip(lines, lines2, results):
            # test parsing a given STA2 line
            got = parse_sta2(line)
            self.assertEqual(got, expected)
            # test that compiling it again gives expected result
            header = {}
            header['network'] = got.pop("network")
            header['gse2'] = got
            # raises "UserWarning: Bad value in GSE2 header field"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                got = compile_sta2(header)
            self.assertEqual(got.decode(), line2)
