#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The sacio test suite.
"""

from obspy.core.util import NamedTemporaryFile
from obspy.sac import sacio, SacError
import inspect, os, unittest
import numpy as np


class SacioTestCase(unittest.TestCase):
    """
    Test cases for sacio.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')

    def tearDown(self):
        pass

    def test_Write(self):
        """
        Tests for sacio, writing artificial seismograms
        """
        #data = array.array('f', [1.1, -1.2, 1.3, -1.4, 1.5, -1.6, 1.7, -1.8,
        #                         1.9, -2.0])
        data = np.array([1.1, -1.2, 1.3, -1.4, 1.5, -1.6, 1.7, -1.8,
                           1.9, -2.0], dtype='<f4')
        t = sacio.ReadSac()
        t.fromarray(data)
        tempfile = NamedTemporaryFile().name
        t.WriteSacBinary(tempfile)
        u = sacio.ReadSac(tempfile)
        for _k in ["kstnm", "npts", "nvhdr", "delta"]:
            self.assertEqual(t.GetHvalue(_k), u.GetHvalue(_k))
        self.assertEqual(t.GetHvalue("kstnm"), "-12345  ")
        np.testing.assert_array_equal(t.seis, u.seis)
        os.remove(tempfile)

    def test_Date(self):
        """
        Test for sacio '_get_date_'-function to calculate timestamp
        """
        fn = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        t = sacio.ReadSac(fn)
        self.assertEqual(t.starttime.timestamp, 269596800.0)
        diff = t.GetHvalue('npts')
        self.assertEqual(int(t.endtime - t.starttime), diff)

    def test_read(self):
        """
        Tests for sacio read and write
        """
        data = np.array([-8.7422776573475858e-08, -0.30901697278022766,
                         - 0.58778536319732666, -0.8090171217918396,
                         - 0.95105659961700439, -1.0, -0.95105630159378052,
                         - 0.80901658535003662, -0.5877845287322998,
                         - 0.30901604890823364, 1.1285198979749111e-06],
                         dtype='<f4')
        sacfile = os.path.join(self.path, 'test.sac')
        t = sacio.ReadSac()
        t.ReadSacFile(sacfile)
        np.testing.assert_array_equal(t.seis[0:11], data)
        self.assertEqual(t.GetHvalue('npts'), 100)
        self.assertEqual(t.GetHvalue("kstnm"), "STA     ")

    def test_readWrite(self):
        """
        Tests for sacio read and write
        """
        sacfile = os.path.join(self.path, 'test.sac')
        tempfile = NamedTemporaryFile().name
        t = sacio.ReadSac()
        t.ReadSacFile(sacfile)
        self.assertEqual(t.GetHvalue('npts'), 100)
        self.assertEqual(t.GetHvalue("kcmpnm"), "Q       ")
        self.assertEqual(t.GetHvalue("kstnm"), "STA     ")
        t.SetHvalue("kstnm", "spiff")
        self.assertEqual(t.GetHvalue('kstnm'), 'spiff   ')
        t.WriteSacBinary(tempfile)
        self.assertEqual(os.path.exists(tempfile), True)
        t.ReadSacHeader(tempfile)
        self.assertEqual((t.hf != None), True)
        t.SetHvalue("kstnm", "spoff")
        self.assertEqual(t.GetHvalue('kstnm'), 'spoff   ')
        t.WriteSacHeader(tempfile)
        t.SetHvalueInFile(tempfile, "kcmpnm", 'Z       ')
        self.assertEqual(t.GetHvalueFromFile(tempfile, "kcmpnm"), 'Z       ')
        t.IsValidSacFile(tempfile)
        os.remove(tempfile)

    def test_readWriteXY(self):
        """
        Tests for ascii sac io
        """
        tempfile = NamedTemporaryFile().name
        tempfile2 = NamedTemporaryFile().name
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        t = sacio.ReadSac(tfile)
        t.WriteSacXY(tempfile)
        d = sacio.ReadSac(tempfile, alpha=True)
        d.WriteSacBinary(tempfile2)
        size1 = os.stat(tempfile2)[6]
        size2 = os.stat(tfile)[6]
        self.assertEqual(size1, size2)
        np.testing.assert_array_almost_equal(t.seis, d.seis, decimal=5)
        os.remove(tempfile)
        os.remove(tempfile2)


    def test_readBigEnd(self):
        """
        Test reading big endian binary files
        """
        tfilel = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        tfileb = os.path.join(os.path.dirname(__file__), 'data', 'test.sac.swap')
        tl = sacio.ReadSac(tfilel)
        tb = sacio.ReadSac(tfileb)
        self.assertEqual(tl.GetHvalue('kevnm'), tb.GetHvalue('kevnm'))
        self.assertEqual(tl.GetHvalue('npts'), tb.GetHvalue('npts'))
        self.assertEqual(tl.GetHvalueFromFile(tfilel, 'kcmpnm'), tb.GetHvalueFromFile(tfileb, 'kcmpnm'))





    def test_isSAC(self):
        """
        See if assertation is Raised if file ist not a sac file
        """
        t = sacio.ReadSac()
        self.assertRaises(SacError, t.ReadSacFile, __file__)


def suite():
    return unittest.makeSuite(SacioTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
