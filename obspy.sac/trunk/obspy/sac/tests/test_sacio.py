#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The sacio test suite.
"""

from obspy.core.util import NamedTemporaryFile
from obspy.sac import SacIO, SacError, ReadSac
import numpy as np
import inspect
import os
import unittest
import sys


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
        data = np.array([1.1, -1.2, 1.3, -1.4, 1.5, -1.6, 1.7, -1.8,
                           1.9, -2.0], dtype='<f4')
        t = ReadSac()
        t.fromarray(data)
        tempfile = NamedTemporaryFile().name
        t.WriteSacBinary(tempfile)
        u = ReadSac(tempfile)
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
        t = ReadSac(fn)
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
        t = ReadSac()
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
        t = ReadSac()
        t.ReadSacFile(sacfile)
        self.assertEqual(t.GetHvalue('npts'), 100)
        self.assertEqual(t.GetHvalue("kcmpnm"), "Q       ")
        self.assertEqual(t.GetHvalue("kstnm"), "STA     ")
        t.SetHvalue("kstnm", "spiff")
        self.assertEqual(t.GetHvalue('kstnm'), 'spiff   ')
        t.WriteSacBinary(tempfile)
        self.assertEqual(os.stat(sacfile)[6],os.stat(tempfile)[6])
        self.assertEqual(os.path.exists(tempfile), True)
        t.ReadSacHeader(tempfile)
        self.assertEqual((t.hf != None), True)
        t.SetHvalue("kstnm", "spoff")
        self.assertEqual(t.GetHvalue('kstnm'), 'spoff   ')
        t.WriteSacHeader(tempfile)
        t.SetHvalueInFile(tempfile, "kcmpnm", 'Z       ')
        self.assertEqual(t.GetHvalueFromFile(tempfile, "kcmpnm"), 'Z       ')
        self.assertEqual(ReadSac(tempfile,headonly=True).GetHvalue('kcmpnm'),'Z       ')
        self.assertEqual(t.IsValidSacFile(tempfile),True)
        self.assertEqual(t.IsValidXYSacFile(tempfile),False)
        self.assertEqual(ReadSac().GetHvalueFromFile(sacfile,'npts'),100)
        self.assertEqual(ReadSac(sacfile).GetHvalue('npts'),100)
        os.remove(tempfile)

    def test_readWriteXY(self):
        """
        Tests for ascii sac io
        """
        tempfile = NamedTemporaryFile().name
        tempfile2 = NamedTemporaryFile().name
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        t = ReadSac(tfile)
        t.WriteSacXY(tempfile)
        d = ReadSac(tempfile, alpha=True)
        e = ReadSac()
        e.ReadSacXY(tempfile)
        self.assertEqual(e.GetHvalue('npts'),d.GetHvalue('npts'))
        self.assertEqual(e.IsValidXYSacFile(tempfile),True)
        self.assertEqual(e.IsValidSacFile(tempfile),False)
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
        tfileb = os.path.join(os.path.dirname(__file__), 'data',
                              'test.sac.swap')
        tl = ReadSac(tfilel)
        tb = ReadSac(tfileb)
        self.assertEqual(tl.GetHvalue('kevnm'), tb.GetHvalue('kevnm'))
        self.assertEqual(tl.GetHvalue('npts'), tb.GetHvalue('npts'))
        self.assertEqual(tl.GetHvalueFromFile(tfilel, 'kcmpnm'),
                         tb.GetHvalueFromFile(tfileb, 'kcmpnm'))
        np.testing.assert_array_equal(tl.seis, tb.seis)

    def test_swapbytes(self):
        tfilel = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        tfileb = os.path.join(os.path.dirname(__file__), 'data',
                              'test.sac.swap')
        tempfile = NamedTemporaryFile().name
        tb = ReadSac(tfileb)
        tb.swap_byte_order()
        tb.WriteSacBinary(tempfile)
        tr1 = ReadSac(tempfile)
        tl = ReadSac(tfilel)
        np.testing.assert_array_equal(tl.seis, tr1.seis)
        self.assertEqual(tl.GetHvalue('kevnm'), tr1.GetHvalue('kevnm'))
        self.assertEqual(tl.GetHvalue('npts'), tr1.GetHvalue('npts'))
        self.assertEqual(tl.GetHvalueFromFile(tfilel, 'kcmpnm'),
                         tr1.GetHvalueFromFile(tempfile, 'kcmpnm'))
        os.remove(tempfile)

    def test_getdist(self):
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        tempfile = NamedTemporaryFile().name
        t = ReadSac(tfile)
        t.SetHvalue('evla', 48.15)
        t.SetHvalue('evlo', 11.58333)
        t.SetHvalue('stla', -41.2869)
        t.SetHvalue('stlo', 174.7746)
        t.SetHvalue('lcalda', 1)
        t.WriteSacBinary(tempfile)
        t2 = ReadSac(tempfile)
        b = np.array([18486532.5788 / 1000., 65.654154562, 305.975459869],
                     dtype='>f4')
        self.assertEqual(t2.GetHvalue('dist'), b[0])
        self.assertEqual(t2.GetHvalue('az'), b[1])
        self.assertEqual(t2.GetHvalue('baz'), b[2])
        os.remove(tempfile)

    def test_isSAC(self):
        """
        Assertion is raised if file is not a SAC file
        """
        t = ReadSac()
        self.assertRaises(SacError, t.ReadSacFile, __file__)

    def test_getattr(self):
        tfile = os.path.join(os.path.dirname(__file__), 'data', 'test.sac')
        tr = ReadSac(tfile)
        self.assertEqual(tr.npts, tr.GetHvalue('npts'))
        self.assertEqual(tr.kstnm, tr.GetHvalue('kstnm'))

    ### def test_raiseOnGetDist(self):
    ###     """
    ###     Test case to check that SACError is raised if obspy.signal is not
    ###     installed. SACError must be raised as it is catched by various
    ###     methods. The import of setuptools introduces a function
    ###     findall, which recursively searches directories for pth files.
    ###     Could not get obspy.signal out of the path so far...
    ###     """
    ###     t = ReadSac()
    ###     t.SetHvalue('evla',48.15)
    ###     t.SetHvalue('evlo',11.58333)
    ###     t.SetHvalue('stla',-41.2869)
    ###     t.SetHvalue('stlo',174.7746)
    ###     delete obspy.signal from system path list
    ###     signal_path = [sys.path.pop(sys.path.index(j)) for j in \
    ###             [i for i in sys.path if 'obspy.signal' in i]]
    ###     # delete obspy.signal from all imported modules dict
    ###     #[sys.modules.pop(i) for i in \
    ###     #        sys.modules.keys() if 'obspy.signal' in i]
    ###     #import ipdb; ipdb.set_trace()
    ###     self.assertRaises(SacError, t._get_dist_)
    ###     sys.path.extend(signal_path)

def suite():
    return unittest.makeSuite(SacioTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
