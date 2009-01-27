import unittest

#for md5 checksums
import hashlib

#Append path to find libmseedclass - Other possibilities to achieve this?
#Another problem is to locate the libmseed library elsewhere than in this
#directory
import sys
sys.path.append("..") 

from libmseedclass import *

class FileTest(unittest.TestCase):
    "Tests whether all test files are available and correct."
    
    #List of all necessary files and their correspondig md5 hash
    FilesandChecksums = ( ('test.mseed' , '435ad74033321e8634926bddf85338f2'),
                          ('BW.BGLD..EHE.D.2008.001' , 'a01892544a9035f851d8c5f5d9224725')
                        )
    
    #Directory where the test files are located
    TestFilesDir = "TestFiles/"
    
    def mkmd5sum(self, filename):
        """
        returns the md5 sum of a file
        """
        return hashlib.md5(file(filename).read()).hexdigest()

    
    def testfiles(self):
        """
        All files needed for the tests should be available with the correct checksum
        """
        for (filename, md5sum) in self.FilesandChecksums:
            self.assertEqual(self.mkmd5sum(self.TestFilesDir+filename), md5sum, "The file "+filename+
                             " is either damaged, not available or not correct")

class MSRPrint(unittest.TestCase):
    "Tests the msr_print function provided by libmseed"
    #HOW TO TEST THE OUTPUT??
    #expoutput = "NL_HGN_00_BHZ, 000001, R, 4096, 5980 samples, 40 Hz, 2003,149,02:13:22.043400" +"\n"+"NL_HGN_00_BHZ, 000002, R, 4096, 5967 samples, 40 Hz, 2003,149,02:15:51.54340"
    #mseed=libmseed()
    #self.assertEqual(mseed.msr_print('test.mseed'), print self.expoutput)
    pass

class MSReasMSR(unittest.TestCase):
    "Tests various attributes extracted from the miniseed files by ms_read_msr()"
    #Fails sometimes to due memory/pointer issues
    def testattrs(self):
        mseed=libmseed("TestFiles/test.mseed")
        self.assertEqual(mseed.samprate, 40.0)
        self.assertEqual(mseed.reclen, 4096)
    
           
if __name__ == "__main__":
    unittest.main()   