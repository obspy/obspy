# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import datetime

import numpy as np

from obspy import UTCDateTime
from obspy.core.util import NamedTemporaryFile
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

from ..sactrace import SACTrace
from ..util import SacHeaderError


class SACTraceTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.sac.sactrace
    """
    def setUp(self):
        self.path = os.path.dirname(__file__)
        self.file = os.path.join(self.path, 'data', 'test.sac')
        self.filexy = os.path.join(self.path, 'data', 'testxy.sac')
        self.filebe = os.path.join(self.path, 'data', 'test.sac.swap')
        self.fileseis = os.path.join(self.path, 'data', 'seism.sac')
        self.testdata = np.array(
            [-8.74227766e-08, -3.09016973e-01,
             -5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
             -1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
             -5.87784529e-01, -3.09016049e-01], dtype=np.float32)

    def test_read_binary(self):
        """
        Tests for SACTrace binary file read
        """
        sac = SACTrace.read(self.file, byteorder='little')
        self.assertEqual(sac.npts, 100)
        self.assertEqual(sac.kstnm, 'STA')
        self.assertEqual(sac.delta, 1.0)
        self.assertEqual(sac.kcmpnm, 'Q')
        self.assertEqual(sac.reftime.datetime,
                         datetime.datetime(1978, 7, 18, 8, 0))
        self.assertEqual(sac.nvhdr, 6)
        self.assertEqual(sac.b, 10.0)
        self.assertAlmostEqual(sac.depmen, 9.0599059e-8)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             sac.data[0:10])

    def test_read_binary_headonly(self):
        """
        A headonly read should return readable headers and data is None
        """
        sac = SACTrace.read(self.file, byteorder='little', headonly=True)
        self.assertEqual(sac.data, None)
        self.assertEqual(sac.npts, 100)
        self.assertEqual(sac.depmin, -1.0)
        self.assertAlmostEqual(sac.depmen, 8.344650e-8)
        self.assertEqual(sac.depmax, 1.0)

    def test_read_sac_byteorder(self):
        """
        A read should fail if the byteorder is wrong
        """
        with self.assertRaises(IOError):
            sac = SACTrace.read(self.filebe, byteorder='little')
        with self.assertRaises(IOError):
            sac = SACTrace.read(self.file, byteorder='big')
        # a SACTrace should show the correct byteorder
        sac = SACTrace.read(self.filebe, byteorder='big')
        self.assertEqual(sac.byteorder, 'big')
        sac = SACTrace.read(self.file, byteorder='little')
        self.assertEqual(sac.byteorder, 'little')
        # a SACTrace should autodetect the correct byteorder
        sac = SACTrace.read(self.file)
        self.assertEqual(sac.byteorder, 'little')
        sac = SACTrace.read(self.filebe)
        self.assertEqual(sac.byteorder, 'big')

    def test_write_sac(self):
        """
        A trace you've written and read in again should look the same as the
        one you started with.
        """
        sac1 = SACTrace.read(self.file, byteorder='little')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            sac1.write(tempfile, byteorder='little')
            sac2 = SACTrace.read(tempfile, byteorder='little')
        np.testing.assert_array_equal(sac1.data, sac2.data)
        self.assertEqual(sac1._header, sac2._header)

    def test_write_binary_headonly(self):
        """
        A trace you've written headonly should only modify the header of an
        existing file, and fail if the file doesn't exist.
        """
        # make a sac trace
        sac = SACTrace.read(self.file, byteorder='little')
        # write it all to temp file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            sac.write(tempfile, byteorder='little')
            # read it headonly and modify the header
            # modify the data, too, and verify it didn't get written
            sac2 = SACTrace.read(tempfile, headonly=True, byteorder='little')
            sac2.kcmpnm = 'xyz'
            sac2.b = 7.5
            sac2.data = np.array([1.5, 2e-3, 17], dtype=np.float32)
            # write it again (write over)
            sac2.write(tempfile, headonly=True, byteorder='little')
            # read it all and compare
            sac3 = SACTrace.read(tempfile, byteorder='little')
        self.assertEqual(sac3.kcmpnm, 'xyz')
        self.assertEqual(sac3.b, 7.5)
        np.testing.assert_array_equal(sac3.data, sac.data)

        # ...and fail if the file doesn't exist
        sac = SACTrace.read(self.file, headonly=True, byteorder='little')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
        with self.assertRaises(IOError):
            sac.write(tempfile, headonly=True, byteorder='little')

    def test_read_sac_ascii(self):
        """
        Read an ASCII SAC file.
        """
        sac = SACTrace.read(self.filexy, ascii=True)
        self.assertEqual(sac.npts, 100)
        self.assertEqual(sac.kstnm, 'sta')
        self.assertEqual(sac.delta, 1.0)
        self.assertEqual(sac.kcmpnm, 'Q')
        self.assertEqual(sac.nvhdr, 6)
        self.assertEqual(sac.b, 10.0)
        self.assertAlmostEqual(sac.depmen, 9.4771387e-08)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             sac.data[0:10])

    def test_reftime(self):
        """
        A SACTrace.reftime should be created correctly from a file's nz-times
        """
        sac = SACTrace.read(self.fileseis)
        self.assertEqual(sac.reftime,
                         UTCDateTime('1981-03-29T10:38:14.000000Z'))
        # changes to a reftime should be reflected in the nz times and reftime
        nzsec, nzmsec = sac.nzsec, sac.nzmsec
        sac.reftime = sac.reftime + 2.5
        self.assertEqual(sac.nzsec, nzsec + 2)
        self.assertEqual(sac.nzmsec, nzmsec + 500)
        self.assertEqual(sac.reftime,
                         UTCDateTime('1981-03-29T10:38:16.500000Z'))
        # changes in the nztimes should be reflected reftime
        sac.nzyear = 2001
        self.assertEqual(sac.reftime.year, 2001)

    def test_reftime_relative_times(self):
        """
        Changes in the reftime shift all relative time headers
        """
        sac = SACTrace.read(self.fileseis)
        a, b, t1 = sac.a, sac.b, sac.t1
        sac.reftime -= 10.0
        self.assertAlmostEqual(sac.a, a + 10.0, 5)
        self.assertAlmostEqual(sac.b, b + 10.0)
        self.assertAlmostEqual(sac.t1, t1 + 10.0)
        # changes in the reftime should push remainder microseconds to the
        # relative time headers, and milliseconds to the nzmsec
        sac = SACTrace(b=5.0, t1=20.0)
        b, t1, nzmsec = sac.b, sac.t1, sac.nzmsec
        sac.reftime += 1.2e-3
        self.assertEqual(sac.nzmsec, nzmsec + 1)
        self.assertAlmostEqual(sac.b, b - 1.0e-3, 6)
        self.assertAlmostEqual(sac.t1, t1 - 1.0e-3, 5)

    def test_lcalda(self):
        """
        Test that distances are set when geographic information is present and
        lcalda is True, and that they're not set when geographic information
        is missing or lcalca is false.
        """
        stla, stlo, evla, evlo = -35.0, 100, 42.5, -37.5
        meters, az, baz = gps2dist_azimuth(evla, evlo, stla, stlo)
        km = meters / 1000.0
        gcarc = kilometer2degrees(km)

        # distances are set when lcalda True and all distance values set
        sac = SACTrace(lcalda=True, stla=stla, stlo=stlo, evla=evla, evlo=evlo)
        self.assertAlmostEqual(sac.az, az, places=4)
        self.assertAlmostEqual(sac.baz, baz, places=4)
        self.assertAlmostEqual(sac.dist, km, places=4)
        self.assertAlmostEqual(sac.gcarc, gcarc, places=4)
        # distances are not set when lcalda False and all distance values set
        sac = SACTrace(lcalda=False, stla=stla, stlo=stlo, evla=evla,
                       evlo=evlo)
        self.assertIs(sac.az, None)
        self.assertIs(sac.baz, None)
        self.assertIs(sac.dist, None)
        self.assertIs(sac.gcarc, None)
        # distances are not set when lcalda True, not all distance values set
        sac = SACTrace(lcalda=True, stla=stla)
        self.assertIs(sac.az, None)
        self.assertIs(sac.baz, None)
        self.assertIs(sac.dist, None)
        self.assertIs(sac.gcarc, None)
        # exception raised when set_distances is forced but not all distances
        # values are set. NOTE: still have a problem when others are "None".
        sac = SACTrace(lcalda=True, stla=stla)
        self.assertRaises(SacHeaderError, sac._set_distances, force=True)


def suite():
    return unittest.makeSuite(SACTraceTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
