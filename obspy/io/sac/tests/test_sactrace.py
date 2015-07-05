import os
import unittest
import datetime

import numpy as np

from future.utils import native_str

from obspy import UTCDateTime
from obspy.core.util import NamedTemporaryFile

from ..sactrace import SACTrace

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
        #Tests for SACTrace binary file read
        sac = SACTrace.read(self.file, byteorder='little')
        self.assertEqual(sac.npts, 100)
        self.assertEqual(sac.kstnm, 'STA')
        self.assertEqual(sac.delta, 1.0)
        self.assertEqual(sac.kcmpnm, 'Q')
        self.assertEqual(sac.reftime.datetime, datetime.datetime(1978, 7, 18, 8, 0))
        self.assertEqual(sac.nvhdr, 6)
        self.assertEqual(sac.b, 10.0)
        self.assertAlmostEqual(sac.depmen, 9.0599059e-8)
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             sac.data[0:10])

    def test_read_binary_headonly(self):
        # a headonly read should return readable headers and data == None
        sac = SACTrace.read(self.file, byteorder='little', headonly=True)
        self.assertEqual(sac.data, None)
        self.assertEqual(sac.npts, 100)
        self.assertEqual(sac.depmin, -1.0)
        self.assertAlmostEqual(sac.depmen, 8.344650e-8)
        self.assertEqual(sac.depmax, 1.0)

    def test_read_sac_byteorder(self):
        # a read should fail if the byteorder is wrong
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
        # A trace you've written and read in again should look the same as the
        # one you started with.
        sac1 = SACTrace.read(self.file, byteorder='little')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            sac1.write(tempfile, byteorder='little')
            sac2 = SACTrace.read(tempfile, byteorder='little')
        np.testing.assert_array_equal(sac1.data, sac2.data)
        self.assertEqual(sac1._header, sac2._header)

    def test_write_binary_headonly(self):
        # A trace you've written headonly should only modify the header of an
        # existing file, and fail if the file doesn't exist.
        #
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

    @unittest.skip("Not implemented yet.")
    def test_write_sac_byteorder(self):
        pass

    def test_read_sac_ascii(self):
        sac = SACTrace.read(self.filexy, ascii=True)
        self.assertEqual(sac.npts, 100)
        self.assertEqual(sac.kstnm, 'sta')
        self.assertEqual(sac.delta, 1.0)
        self.assertEqual(sac.kcmpnm, 'Q')
        self.assertEqual(sac.nvhdr, 6)
        self.assertEqual(sac.b, 10.0)
        self.assertAlmostEqual(sac.depmen, 9.4771387e-08)
        np.testing.assert_array_almost_equal(self.testdata[0:10], sac.data[0:10])

    @unittest.skip("Not implemented yet.")
    def test_write_sac_ascii(self):
        pass

    def test_reftime(self):
        # a SACTrace.reftime should be created correctly from a file's nz-times
        sac = SACTrace.read(self.fileseis)
        self.assertEqual(sac.reftime, UTCDateTime('1981-03-29T10:38:14.000000Z'))
        # changes to a reftime should be reflected in the nz times and reftime
        nzsec, nzmsec= sac.nzsec, sac.nzmsec
        sac.reftime = sac.reftime + 2.5
        self.assertEqual(sac.nzsec, nzsec + 2)
        self.assertEqual(sac.nzmsec, nzmsec + 500)
        self.assertEqual(sac.reftime, UTCDateTime('1981-03-29T10:38:16.500000Z'))
        # changes in the nztimes should be reflected reftime
        sac.nzyear = 2001
        self.assertEqual(sac.reftime.year, 2001)

    def test_reftime_relative_times(self):
        # changes in the reftime shift all relative time headers
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


    @unittest.skip("Not implemented yet.")
    def test_dict_to_header_arrays(self):
        pass
