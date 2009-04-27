# -*- coding: utf-8 -*-
"""
The libmseed test suite.
"""

from datetime import datetime
from obspy.mseed import libmseed
import inspect
import os
import random
import unittest


class LibMSEEDTestCase(unittest.TestCase):
    """
    Test cases for libmseed.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass
    
    def test_convertDatetime(self):
        """
        Tests all time conversion methods.
        
        This is a sanity test. A conversion to a format and back should not
        change the value
        """
        # These values are created using the Linux "date -u -d @TIMESTRING"
        # command. These values are assumed to be correct.
        timesdict = {'1234567890' : datetime(2009, 2, 13, 23, 31, 30),
                     '1111111111' : datetime(2005, 3, 18, 1, 58, 31),
                     '1212121212' : datetime(2008, 5, 30, 4, 20, 12),
                     '1313131313' : datetime(2011, 8, 12, 6, 41, 53),
                     '100000' : datetime(1970, 1, 2, 3, 46, 40),
                     '200000000' : datetime(1976, 5, 3, 19, 33, 20)}
        mseed = libmseed()
        # Loop over timesdict.
        for _i in timesdict.keys():
            self.assertEqual(timesdict[_i],
                    mseed.convertMSTimeToDatetime(long(_i)*1000000L))
            self.assertEqual(long(_i) * 1000000L,
                    mseed.convertDatetimeToMSTime(timesdict[_i]))
        # Additional sanity tests.
        # Today.
        now = datetime.now()
        self.assertEqual(now, mseed.convertMSTimeToDatetime(
                              mseed.convertDatetimeToMSTime(now)))
        # Some random date.
        timestring = random.randint(0, 2000000) * 1e6
        self.assertEqual(timestring, mseed.convertDatetimeToMSTime(
                        mseed.convertMSTimeToDatetime(timestring)))
        
    def test_CFilePointer(self):
        """
        Tests whether the convertion to a C file pointer works.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, 
                                  'BW.BGLD..EHE.D.2008.001')
        open_file = open(filename, 'rb')
        pointer = mseed._convertToCFilePointer(open_file)
        self.assertNotEqual(str(pointer).find('LP_FILE'), -1)
        open_file.close()
        
    def test_readMSRec(self):
        """
        Compares waveform data read by libmseed with an ASCII dump.
        
        Checks the first 13 datasamples when reading the first record of 
        BW.BGLD..EHE.D.2008.001 using traces. The values are assumed to
        be correct. The values were created using Pitsa.
        Only checks relative values.
        """
        filename = os.path.join(self.path, 
                                  'BW.BGLD..EHE.D.2008.001.first_record')
        mseed = libmseed()
        # list of known data samples
        datalist = [-363, -382, -388, -420, -417, -397, -418, -390, -388, -385,
                    -367, -414, -427]
        msr = mseed.read_MSRec(filename)
        chain = msr.contents
        self.assertEqual('BGLD', chain.station)
        self.assertEqual('EHE', chain.channel)
        self.assertEqual(200, chain.samprate)
        data = chain.datasamples[0:13]
        for i in range(len(datalist)-1):
            self.assertEqual(datalist[i]-datalist[i+1], data[i]-data[i+1])
    
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
        mseed = libmseed()
        # list of known data samples
        datalist = [-363, -382, -388, -420, -417, -397, -418, -390, -388, -385,
                    -367, -414, -427]
        header, data, numtraces = mseed.read_ms_using_traces(mseed_file)
        self.assertEqual('BGLD', header['station'])
        self.assertEqual('EHE', header['channel'])
        self.assertEqual(200, header['samprate'])
        self.assertEqual(1199145599915000, header['starttime'])
        self.assertEqual(numtraces, 1)
        for i in range(len(datalist)-1):
            self.assertEqual(datalist[i]-datalist[i+1], data[i]-data[i+1])
    
    def test_readAnWriteTraces(self):
        """
        Writes, reads and compares files created via libmseed.
        
        This uses all possible encodings, record lengths and the byte order 
        options. A reencoded SEED file should still have the same values 
        regardless of write options.
        """
        mseed = libmseed() 
        mseed_file = os.path.join(self.path, 'test.mseed')
        header, data, numtraces = mseed.read_ms_using_traces(mseed_file)
        # define test ranges
        record_length_values = [2**i for i in range(8, 21)]
        encoding_values = [1, 3, 10, 11]
        byteorder_values = [0, 1]
        # deletes the data quality indicators
        testheader = header.copy()
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
                    result = mseed.read_ms_using_traces(temp_file)
                    newheader, newdata, newnumtraces = result
                    del newheader['dataquality']
                    self.assertEqual(testheader, newheader)
                    self.assertEqual(data, newdata)
                    self.assertEqual(numtraces, newnumtraces)
                    os.remove(temp_file)
    
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
    
    def test_readFirstHeaderInfo(self):
        """
        Reads and compares header info from the first record.
        
        The values can be read from the filename.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        self.assertEqual(header['channel'], 'EHE')
    
    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        
        The values are compared with the readTraces() method which parses the
        whole file. This will only work for files with only one trace and with-
        out any gaps or overlaps.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
        # get the start- and end time
        times = mseed.getStartAndEndTime(filename)
        # parse the whole file
        mstg = mseed.readTraces(filename, dataflag = 0)
        chain = mstg.contents.traces.contents
        self.assertEqual(times[0],
                         mseed.convertMSTimeToDatetime(chain.starttime))
        self.assertEqual(times[1],
                         mseed.convertMSTimeToDatetime(chain.endtime))
        

def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
