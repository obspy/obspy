# -*- coding: utf-8 -*-
"""
The libmseed test suite.
"""

from datetime import datetime
from obspy.mseed import libmseed
import inspect
import os
import unittest


class LibMSEEDTestCase(unittest.TestCase):
    """
    Test cases for libmseed.
    """
    def setUp(self):
        # Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass
    
    def test_readTraces(self):
        """
        Compares waveform data read by libmseed with an ASCII dump.
        
        Checks the first 13 datasamples when reading the first record of 
        BW.BGLD..EHE.D.2008.001 using traces. The values are assumed to
        be correct. The values were created using Pitsa.
        Only checks relative values.
        """
        mseed_file = os.path.join(self.path, 
                                  'BW.BGLD..EHE.D.2008.001.first_record')
        mseed=libmseed()
        
        datalist=[-363, -382, -388, -420, -417, -397, -418, -390, -388, -385,
                        -367, -414, -427]
        header, data, numtraces=mseed.read_ms_using_traces(mseed_file)
        self.assertEqual('BGLD', header['station'])
        self.assertEqual('EHE', header['channel'])
        self.assertEqual(200, header['samprate'])
        self.assertEqual(1199145599915000, header['starttime'])
        self.assertEqual(numtraces, 1)
        for i in range(len(datalist)-1):
            self.assertEqual(datalist[i]-datalist[i+1], data[i]-data[i+1])
    
#    def test_readAnWriteTraces(self):
#        """
#        Writes, reads and compares files created via libmseed.
#        
#        This uses all possible encodings, record lengths and the byte order 
#        options. A reencoded SEED file should still have the same values 
#        regardless of write options.
#        """
#        # define test ranges
#        record_length_values = [2**i for i in range(8, 21)]
#        encoding_values = [1, 3, 10, 11]
#        byteorder_values = [0, 1]
#        
#        mseed=libmseed() 
#        mseed_file = os.path.join(self.path, 'test.mseed')
#        header, data, numtraces=mseed.read_ms_using_traces(mseed_file)
#        # Deletes the dataquality indicators
#        testheader=header.copy()
#        del testheader['dataquality']
#        # loops over all combinations of test values
#        for reclen in record_length_values:
#            for byteorder in byteorder_values:
#                for encoding in encoding_values:
#                    filename = 'temp.%s.%s.%s.mseed' % (reclen, byteorder, 
#                                                        encoding)
#                    temp_file = os.path.join(self.path, filename)
#                    mseed.write_ms(header, data, temp_file,
#                                   numtraces, encoding=encoding, 
#                                   byteorder=byteorder, reclen=reclen)
#                    result = mseed.read_ms_using_traces(temp_file)
#                    newheader, newdata, newnumtraces = result
#                    del newheader['dataquality']
#                    self.assertEqual(testheader, newheader)
#                    self.assertEqual(data, newdata)
#                    self.assertEqual(numtraces, newnumtraces)
#                    os.remove(temp_file)
    
    def test_getGapList(self):
        """
        Searches gaps via libmseed and compares the result with known values.
        
        The values are compared with the original printgaplist method of the 
        libmseed library and manually with the SeisGram2K viewer.
        """
        mseed = libmseed()
        # test file with 3 gaps
        filename = os.path.join(self.path, 'gaps.mseed')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(len(gap_list), 3)
        self.assertEqual(gap_list[0][0], 'BW')
        self.assertEqual(gap_list[0][1], 'BGLD')
        self.assertEqual(gap_list[0][2], '')
        self.assertEqual(gap_list[0][3], 'EHE')
        self.assertEqual(gap_list[0][4], datetime(2008, 1, 1, 0, 0, 1, 970000))
        self.assertEqual(gap_list[0][5], datetime(2008, 1, 1, 0, 0, 4, 35000))
        self.assertEqual(gap_list[0][6], 2.065)
        self.assertEqual(gap_list[0][7], 412)
        self.assertEqual(gap_list[1][6], 2.065)
        self.assertEqual(gap_list[1][7], 412)
        self.assertEqual(gap_list[2][6], 4.125)
        self.assertEqual(gap_list[2][7], 824)
        # real example without gaps
        filename = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(gap_list, [])
        # real example with a gap
        filename = os.path.join(self.path, 'BW.RJOB..EHZ.D.2009.056')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(len(gap_list), 1)


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
