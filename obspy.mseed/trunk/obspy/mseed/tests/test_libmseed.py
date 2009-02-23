# -*- coding: utf-8 -*-

from obspy.mseed.libmseed import libmseed
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
    
    def test_writeMS(self):
        """
        A reencoded SEED file should still have the same values regardless of 
        the used record length, encoding and byteorder.
        """
        # define test ranges
        record_length_values = [2**i for i in range(8,21)]
        encoding_values = [1, 3, 10, 11]
        byteorder_values = [0, 1]
        
        mseed=libmseed() 
        header, data, numtraces=mseed.read_ms(os.path.join(self.path,
                                                           'test.mseed'))
        # Deletes the dataquality indicators
        testheader=header.copy()
        del testheader['dataquality']
        # loops over all combinations of test values
        for reclen in record_length_values:
            for byteorder in byteorder_values:
                for encoding in encoding_values:
                    filename = 'temp.%s.%s.%s.mseed' % (reclen, byteorder, 
                                                        encoding)
                    temp_file = os.path.join(self.path, filename)
                    mseed.write_ms(header, data, temp_file,
                                   numtraces, encoding=encoding, 
                                   byteorder=byteorder, reclen=reclen)
                    newheader, newdata, newnumtraces=mseed.read_ms(temp_file)
                    del newheader['dataquality']
                    self.assertEqual(testheader, newheader)
                    self.assertEqual(data, newdata)
                    self.assertEqual(numtraces, newnumtraces)
                    os.remove(temp_file)


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
