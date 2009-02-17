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

    def test_writeMS(self):
        """
        A reencoded file should still have the same values regardless of the
        used record length, encoding and byteorder
        """
        #Define tested values
        record_length_values=[]
        for _i in range(8,21):
            record_length_values.append(2**_i)
        encoding_values=[1,3,10,11] #Offered integer encodings
        byteorder_values=[0,1]
        mseed=libmseed()
        header, data, numtraces=mseed.read_ms(os.path.join(self.path,'test.mseed'))
        #Deletes the dataquality indicators
        testheader=header.copy()
        #Loops over the tested values
        del testheader['dataquality']
        for recvals in record_length_values:
            for bytevals in byteorder_values:
                for encvals in encoding_values:
                    mseed.write_ms(header, data, os.path.join(self.path,'temp.mseed'),
                                   numtraces, encoding=encvals, 
                                   byteorder = bytevals, reclen=recvals)
                    newheader, newdata, newnumtraces=mseed.read_ms(os.path.join(self.path,'temp.mseed'))
                    del newheader['dataquality']
                    self.assertEqual(testheader, newheader)
                    self.assertEqual(data, newdata)
                    self.assertEqual(numtraces, newnumtraces)
                    os.remove(os.path.join(self.path,'temp.mseed'))

def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
