# -*- coding: utf-8 -*-
"""
The libmseed test suite.
"""

from obspy.core import UTCDateTime
from obspy.mseed import libmseed
import copy
import inspect
import numpy as N
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
        timesdict = {
            1234567890 : UTCDateTime(2009, 2, 13, 23, 31, 30),
            1111111111 : UTCDateTime(2005, 3, 18, 1, 58, 31),
            1212121212 : UTCDateTime(2008, 5, 30, 4, 20, 12),
            1313131313 : UTCDateTime(2011, 8, 12, 6, 41, 53),
            100000 : UTCDateTime(1970, 1, 2, 3, 46, 40),
            100000.111112 : UTCDateTime(1970, 1, 2, 3, 46, 40, 111112),
            200000000 : UTCDateTime(1976, 5, 3, 19, 33, 20)
        }
        mseed = libmseed()
        # Loop over timesdict.
        for ts, dt in timesdict.iteritems():
            self.assertEqual(dt, mseed._convertMSTimeToDatetime(ts * 1000000L))
            self.assertEqual(ts * 1000000L, mseed._convertDatetimeToMSTime(dt))
        # Additional sanity tests.
        # Today.
        now = UTCDateTime.now()
        self.assertEqual(now, mseed._convertMSTimeToDatetime(
                              mseed._convertDatetimeToMSTime(now)))
        # Some random date.
        timestring = random.randint(0, 2000000) * 1e6
        self.assertEqual(timestring, mseed._convertDatetimeToMSTime(
                        mseed._convertMSTimeToDatetime(timestring)))

    def test_CFilePointer(self):
        """
        Tests whether the convertion to a C file pointer works.
        """
        mseed = libmseed()
        filename = os.path.join(self.path,
                                  u'BW.BGLD..EHE.D.2008.001')
        fh = open(filename, 'rb')
        pointer = mseed._convertToCFilePointer(fh)
        self.assertNotEqual(str(pointer).find('LP_FILE'), -1)
        fh.close()

    def test_readMSRec(self):
        """
        Compares waveform data read by libmseed with an ASCII dump.
        
        Checks the first 13 datasamples when reading the first record of 
        BW.BGLD..EHE.D.2008.001 using traces. The values are assumed to
        be correct. The values were created using Pitsa.
        Only checks relative values.
        """
        filename = os.path.join(self.path,
                                  u'BW.BGLD..EHE.D.2008.001.first_record')
        mseed = libmseed()
        # list of known data samples
        datalist = [-363, -382, -388, -420, -417, -397, -418, -390, -388, -385,
                    - 367, -414, -427]
        msr = mseed.readSingleRecordToMSR(filename)
        chain = msr.contents
        self.assertEqual('BGLD', chain.station)
        self.assertEqual('EHE', chain.channel)
        self.assertEqual(200, chain.samprate)
        data = chain.datasamples[0:13]
        for i in range(len(datalist) - 1):
            self.assertEqual(datalist[i] - datalist[i + 1],
                             data[i] - data[i + 1])

    def test_readTraces(self):
        """
        Compares waveform data read by libmseed with an ASCII dump.
        
        Checks the first 13 datasamples when reading the first record of 
        BW.BGLD..EHE.D.2008.001 using traces. The values are assumed to
        be correct. The values were created using Pitsa.
        Only checks relative values.
        """
        mseed_file = os.path.join(self.path,
                                  u'BW.BGLD..EHE.D.2008.001.first_record')
        mseed = libmseed()
        # list of known data samples
        datalist = [-363, -382, -388, -420, -417, -397, -418, -390, -388, -385,
                    - 367, -414, -427]
        trace_list = mseed.readMSTraces(mseed_file)
        header = trace_list[0][0]
        self.assertEqual('BGLD', header['station'])
        self.assertEqual('EHE', header['channel'])
        self.assertEqual(200, header['samprate'])
        self.assertEqual(1199145599915000, header['starttime'])
        data = trace_list[0][1][0:13].tolist()
        for i in range(len(datalist) - 1):
            self.assertEqual(datalist[i] - datalist[i + 1],
                             data[i] - data[i + 1])

    def test_readAndWriteTraces(self):
        """
        Writes, reads and compares files created via libmseed.
        
        This uses all possible encodings, record lengths and the byte order 
        options. A reencoded SEED file should still have the same values 
        regardless of write options.
        """
        mseed = libmseed()
        mseed_file = os.path.join(self.path, u'test.mseed')
        trace_list = mseed.readMSTraces(mseed_file)
        # define test ranges
        record_length_values = [2 ** i for i in range(8, 21)]
        encoding_values = [1, 3, 10, 11]
        byteorder_values = [0, 1]
        # deletes the data quality indicators
        testheader = trace_list[0][0].copy()
        del testheader['dataquality']
        # loops over all combinations of test values
        for reclen in record_length_values:
            for byteorder in byteorder_values:
                for encoding in encoding_values:
                    filename = u'temp.%s.%s.%s.mseed' % (reclen, byteorder,
                                                        encoding)
                    temp_file = os.path.join(self.path, filename)
                    mseed.writeMSTraces(copy.deepcopy(trace_list), temp_file,
                                        encoding=encoding, byteorder=byteorder,
                                        reclen=reclen)
                    new_trace_list = mseed.readMSTraces(temp_file)
                    del new_trace_list[0][0]['dataquality']
                    self.assertEqual(testheader, new_trace_list[0][0])
                    N.testing.assert_array_equal(trace_list[0][1],
                                                 new_trace_list[0][1])
                    os.remove(temp_file)

    def test_readAndWriteFileWithGaps(self):
        """
        Tests reading and writing files with more than one trace.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, u'gaps.mseed')
        # Read file and test if all traces are being read.
        trace_list = mseed.readMSTraces(filename)
        self.assertEqual(len(trace_list), 4)
        # Four traces need to have three gaps.
        gap_list = mseed.getGapList(filename)
        self.assertEqual(len(gap_list), len(trace_list) - 1)
        # Write File to temporary file.
        outfile = u'tempfile.mseed'
        mseed.writeMSTraces(copy.deepcopy(trace_list), outfile)
        # Read the same file again and compare it to the original file.
        new_trace_list = mseed.readMSTraces(outfile)
        self.assertEqual(len(trace_list), len(new_trace_list))
        new_gap_list = mseed.getGapList(outfile)
        self.assertEqual(gap_list, new_gap_list)
        # Compare new_trace_list with trace_list
        for _i in xrange(len(trace_list)):
            self.assertEqual(trace_list[_i][0], new_trace_list[_i][0])
            N.testing.assert_array_equal(trace_list[_i][1],
                                         new_trace_list[_i][1])
        os.remove(outfile)

    def test_getGapList(self):
        """
        Searches gaps via libmseed and compares the result with known values.
        
        The values are compared with the original printgaplist method of the 
        libmseed library and manually with the SeisGram2K viewer.
        """
        mseed = libmseed()
        # test file with 3 gaps
        filename = os.path.join(self.path, u'gaps.mseed')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(len(gap_list), 3)
        self.assertEqual(gap_list[0][0], 'BW')
        self.assertEqual(gap_list[0][1], 'BGLD')
        self.assertEqual(gap_list[0][2], '')
        self.assertEqual(gap_list[0][3], 'EHE')
        self.assertEqual(gap_list[0][4],
                         UTCDateTime(2008, 1, 1, 0, 0, 1, 970000))
        self.assertEqual(gap_list[0][5],
                         UTCDateTime(2008, 1, 1, 0, 0, 4, 35000))
        self.assertEqual(gap_list[0][6], 2.065)
        self.assertEqual(gap_list[0][7], 412)
        self.assertEqual(gap_list[1][6], 2.065)
        self.assertEqual(gap_list[1][7], 412)
        self.assertEqual(gap_list[2][6], 4.125)
        self.assertEqual(gap_list[2][7], 824)
        # real example without gaps
        filename = os.path.join(self.path, u'BW.BGLD..EHE.D.2008.001')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(gap_list, [])
        # real example with a single gap
        filename = os.path.join(self.path, u'BW.RJOB..EHZ.D.2009.056')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(len(gap_list), 1)

    def test_readFirstHeaderInfo(self):
        """
        Reads and compares header info from the first record.
        
        The values can be read from the filename.
        """
        mseed = libmseed()
        # Example 1
        filename = os.path.join(self.path, u'BW.BGLD..EHE.D.2008.001')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        self.assertEqual(header['channel'], 'EHE')
        # Example 2
        filename = os.path.join(self.path, u'BW.RJOB..EHZ.D.2009.056')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'RJOB')
        self.assertEqual(header['channel'], 'EHZ')

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        
        The values are compared with the readFileToTraceGroup() method which 
        parses the whole file. This will only work for files with only one
        trace and without any gaps or overlaps.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, u'BW.BGLD..EHE.D.2008.001')
        # get the start- and end time
        (start, end) = mseed.getStartAndEndTime(filename)
        # parse the whole file
        mstg = mseed.readFileToTraceGroup(filename, dataflag=0)
        chain = mstg.contents.traces.contents
        self.assertEqual(start,
                         mseed._convertMSTimeToDatetime(chain.starttime))
        self.assertEqual(end,
                         mseed._convertMSTimeToDatetime(chain.endtime))

    def test_cutMSFileByRecord(self):
        """
        Tests file cutting on a record basis. 
        
        The cut file is compared to a manually cut file which start and end 
        times will be read
        """
        mseed = libmseed()
        original_filename = os.path.join(self.path, u'BW.BGLD..EHE.D.2008.001')
        # Compare a cut file with the a manually cut file. The start time is
        # not set - therefore it will start from the beginning.
        filename = original_filename + '.first_10_percent'
        (_, end) = mseed.getStartAndEndTime(filename)
        data = mseed.cutMSFileByRecords(original_filename, endtime=end)
        data2 = open(filename, 'rb').read()
        self.assertEqual(data, data2)
        # Compare a cut file with the a manually cut file.
        filename = original_filename + '.first_record'
        (_, end) = mseed.getStartAndEndTime(filename)
        data = mseed.cutMSFileByRecords(original_filename, endtime=end)
        data2 = open(filename, 'rb').read()
        self.assertEqual(data, data2)
        # Cut only first record
        (start, _) = mseed.getStartAndEndTime(original_filename)
        data = mseed.cutMSFileByRecords(original_filename, endtime=start + 1)
        data2 = open(filename, 'rb').read()
        self.assertEqual(data, data2)
        # cuts nothing if start time equals end time
        (start, _) = mseed.getStartAndEndTime(original_filename)
        data = mseed.cutMSFileByRecords(original_filename, endtime=start)
        self.assertEqual(data, '')
        # cuts nothing if start time equals end time
        (_, end) = mseed.getStartAndEndTime(original_filename)
        data = mseed.cutMSFileByRecords(original_filename, starttime=end)
        self.assertEqual(data, '')

    def test_mergeAndCutMSFiles(self):
        """
        Creates ten small files, randomizes their order, merges the middle
        eight files and compares it to the desired result.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, u'BW.BGLD..EHE.D.2008.001')
        #Create 10 small files.
        fh = open(filename, 'rb')
        first_ten_records = fh.read(10 * 512)
        fh.close()
        for _i in range(10):
            fh = open(str(_i) + u'_temp.mseed', 'wb')
            fh.write(first_ten_records[_i * 512: (_i + 1) * 512])
            fh.close()
        file_list = [str(_i) + u'_temp.mseed' for _i in range(10)]
        # Randomize list.
        file_list.sort(key=lambda x: random.random())
        # Get the needed start- and endtime.
        second_msr = mseed.readSingleRecordToMSR(filename, dataflag=0,
                                                 record_number=1)
        starttime = mseed._convertMSTimeToDatetime(
                                        second_msr.contents.starttime)
        ninth_msr = mseed.readSingleRecordToMSR(filename, dataflag=0,
                                                 record_number=9)
        endtime = mseed._convertMSTimeToDatetime(
                                        ninth_msr.contents.starttime)
        # Create the merged file<
        data = mseed.mergeAndCutMSFiles(file_list, starttime, endtime)
        # Compare the file to the desired output.
        self.assertEqual(data, first_ten_records[512:-512])
        for _i in file_list:
            os.remove(_i)

    def test_getDataQualityFlagsCount(self):
        """
        This test reads a self-made Mini-SEED file with set Data Quality Bits.
        A real test file would be better as this test tests a file that was
        created by the inverse method that reads the bits.
        """
        mseed = libmseed()
        filename = os.path.join(self.path, u'qualityflags.mseed')
        # Read quality flags.
        flags = mseed.getDataQualityFlagsCount(filename)
        # The test file contains 18 records. The first record has no set bit,
        # bit 0 of the second record is set, bit 1 of the third, ..., bit 7 of
        # the 9th record is set. The last nine records have 0 to 8 set bits,
        # starting with 0 bits, bit 0 is set, bits 0 and 1 are set...
        # Altogether the file contains 44 set bits.
        self.assertEqual(flags, [9, 8, 7, 6, 5, 4, 3, 2])

    def test_getTimingQuality(self):
        """
        This test reads a self-made Mini-SEED file with Timing Quality
        information in Blockette 1001. A real test file would be better.
        
        The test file contains 101 records with the timing quality ranging from
        0 to 100 in steps of 1.
        
        The result is compared to the result from the following R command:
        
        V <- 0:100; min(V); max(V); mean(V); median(V); quantile(V, 0.75,
        type = 3); quantile(V, 0.25, type = 3)
        """
        mseed = libmseed()
        filename = os.path.join(self.path, u'timingquality.mseed')
        tq = mseed.getTimingQuality(filename)
        self.assertEqual(tq, {'min': 0.0, 'max': 100.0, 'average': 50.0,
                              'median': 50.0, 'upper_quantile': 75.0,
                              'lower_quantile': 25.0})

    def test_isMSEED(self):
        """
        This tests the isMSEED method by just validating that each file in the
        data directory is a Mini-SEED file and each file in the working
        directory is not a Mini-SEED file.
        
        The filenames are hard coded so the test will not fail with future
        changes in the structure of the package.
        """
        mseed = libmseed()
        # Mini-SEED filenames.
        mseed_filenames = [u'BW.BGLD..EHE.D.2008.001', u'gaps.mseed',
                           u'qualityflags.mseed', u'test.mseed',
                           u'timingquality.mseed']
        # Non Mini-SEED filenames.
        non_mseed_filenames = [u'test_libmseed.py', u'__init__.py',
                               u'test_core.py']
        # Loop over Mini-SEED files
        for _i in mseed_filenames:
            filename = os.path.join(self.path, _i)
            isMSEED = mseed.isMSEED(filename)
            self.assertTrue(isMSEED)
        # Loop over non Mini-SEED files
        for _i in non_mseed_filenames:
            filename = os.path.join(self.path, _i)
            isMSEED = mseed.isMSEED(filename)
            self.assertFalse(isMSEED)


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
