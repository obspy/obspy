#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The sacio test suite.
"""

from obspy.sac import sacio, SacError
import inspect, os, unittest
import array


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
        data = array.array('f', [ 1.1, -1.2, 1.3, -1.4, 1.5,
                                 - 1.6, 1.7, -1.8, 1.9, -2.0])
        t = sacio.ReadSac()
        t.InitArrays()
        t.seis = data
        t.SetHvalue("kstnm", "RJOB")
        t.SetHvalue("npts", len(data)) # set the number of data points
        t.SetHvalue('nvhdr', 1)  # SAC version needed 0<version<20
        t.SetHvalue('delta', 200) # sampling rate
        t.WriteSacBinary('test.sac')
        u = sacio.ReadSac()
        u.ReadSacFile('test.sac')
        for _k in ["kstnm", "npts", "nvhdr", "delta"]:
            self.assertEqual(t.GetHvalue(_k), u.GetHvalue(_k))
        self.assertEqual(t.GetHvalue("kstnm"), "RJOB    ")
        self.assertEqual(t.seis, u.seis)
        os.remove('test.sac')

    def test_readWrite(self):
        """
        Tests for sacio read and write
        """
        sacfile = os.path.join(self.path, 'test.sac')
        t = sacio.ReadSac()
        t.ReadSacFile(sacfile)
        # commented get_attr method in sacio
        #self.assertEqual(t.get_attr(), 1)
        self.assertEqual(t.GetHvalue('npts'), 100)
        self.assertEqual(t.GetHvalue("kstnm"), "STA     ")
        t.SetHvalue("kstnm", "spiff")
        self.assertEqual(t.GetHvalue('kstnm'), 'spiff   ')
        t.WriteSacBinary('test2.sac')
        self.assertEqual(os.path.exists('test2.sac'), True)
        t.ReadSacHeader('test2.sac')
        self.assertEqual((t.hf != None), True)
        t.SetHvalue("kstnm", "spoff")
        self.assertEqual(t.GetHvalue('kstnm'), 'spoff   ')
        t.WriteSacHeader('test2.sac')
        t.SetHvalueInFile('test2.sac', "kcmpnm", 'Z')
        self.assertEqual(t.GetHvalueFromFile('test2.sac', "kcmpnm"),
                         'Z       ')
        t.IsValidSacFile('test2.sac')
        os.remove('test2.sac')

    def test_readWriteXY(self):
        t = sacio.ReadSac()
        t.ReadXYSacFile(os.path.join(self.path, 'testxy.sac'))
        self.assertEqual(t.GetHvalue('npts'), 100)
        t.WriteSacBinary('testbin.sac')
        self.assertEqual(os.path.exists('testbin.sac'), True)
        os.remove('testbin.sac')

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

