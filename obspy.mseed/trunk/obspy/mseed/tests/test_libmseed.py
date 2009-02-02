# -*- coding: utf-8 -*-

from obspy.mseed.libmseed import libmseed
import hashlib
import inspect
import os
import unittest


class LibMSEEDTestCase(unittest.TestCase):
    """
    Tests whether all test files are available and correct.
    """
    
    def setUp(self):
        #Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass
    
    def mkmd5sum(self, filename):
        """
        returns the md5 sum of a file
        """
        return hashlib.md5(file(filename).read()).hexdigest()

    
    def test_checkChecksum(self):
        """
        All files needed for the tests should be available with the correct 
        checksum.
        """
        # List of all necessary files and their corresponding md5 hash
        files_and_checksums = [
            ('test.mseed' , '435ad74033321e8634926bddf85338f2'),
            ('BW.BGLD..EHE.D.2008.001' , 'a01892544a9035f851d8c5f5d9224725')
        ]
        for (filename, md5sum) in files_and_checksums:
            file = os.path.join(self.path, filename)
            self.assertEqual(self.mkmd5sum(file), md5sum)
    
    def test_printMSR(self):
        """
        Tests the msr_print function provided by libmseed
        """
        #HOW TO TEST THE OUTPUT??
        #expoutput = "NL_HGN_00_BHZ, 000001, R, 4096, 5980 samples, 40 Hz, 2003,149,02:13:22.043400" +"\n"+"NL_HGN_00_BHZ, 000002, R, 4096, 5967 samples, 40 Hz, 2003,149,02:15:51.54340"
        #mseed=libmseed()
        #self.assertEqual(mseed.msr_print('test.mseed'), print self.expoutput)
        pass

    def test_variousAttributes(self):
        """
        Tests various attributes extracted from the miniseed files.
        """
        #Fails sometimes to due memory/pointer issues
        file = os.path.join(self.path, "test.mseed")
        mseed=libmseed(file)
        self.assertEqual(mseed.samprate, 40.0)


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
