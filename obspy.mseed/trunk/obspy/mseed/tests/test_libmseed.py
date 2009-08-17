# -*- coding: utf-8 -*-
"""
The libmseed test suite.
"""

from obspy.core import UTCDateTime
from obspy.mseed import libmseed
from obspy.mseed.libmseed import clibmseed
from StringIO import StringIO
import copy
import inspect
import numpy as N
import ctypes as C
import threading
import os
import random
import time
import unittest


class LibMSEEDTestCase(unittest.TestCase):
    """
    Test cases for libmseed.
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(self.dir, 'data')

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

    def test_readMSTracesViaRecords(self):
        """
        Compares waveform data read by libmseed with an ASCII dump.
        
        Checks the first 9 datasamples of each entry in trace_list of 
        gaps.mseed. The values are assumed to be correct. The first values
        were created using Pitsa.
        """
        mseed_file = os.path.join(self.path, u'gaps.mseed')
        mseed = libmseed()
        # list of known data samples
        starttime = [1199145599915000L, 1199145604035000L, 1199145610215000L,
                     1199145618455000L]
        datalist = [[-363, -382, -388, -420, -417, -397, -418, -390, -388],
                    [-427, -416, -393, -430, -426, -407, -401, -422, -439],
                    [-396, -399, -387, -384, -393, -380, -365, -394, -426],
                    [-389, -428, -409, -389, -388, -405, -390, -368, -368]]
        #trace_list = mseed.readMSTracesViaRecords(mseed_file)
        i = 0
        trace_list = mseed.readMSTracesViaRecords(mseed_file)
        #import pdb;pdb.set_trace()
        for header, data in trace_list:
            self.assertEqual('BGLD', header['station'])
            self.assertEqual('EHE', header['channel'])
            self.assertEqual(200, header['samprate'])
            self.assertEqual(starttime[i], header['starttime'])
            self.assertEqual(datalist[i], data[0:9].tolist())
            i += 1
        del trace_list, header, data
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_record',
                           u'qualityflags.mseed', u'test.mseed',
                           u'timingquality.mseed']
        samprate = [200.0, 200.0, 40.0, 200.0]
        station = ['BGLD', 'BGLD', 'HGN', 'BGLD']
        npts = [412, 412, 11947, 41604, 1]
        for i, _f in enumerate(mseed_filenames):
            file = os.path.join(self.path, _f)
            trace_list = mseed.readMSTraces(file)
            header = trace_list[0][0]
            self.assertEqual(samprate[i], header['samprate'])
            self.assertEqual(station[i], header['station'])
            self.assertEqual(npts[i], len(trace_list[0][1]))
        del trace_list, header

    def test_readMSTraces(self):
        """
        Compares waveform data read by libmseed with an ASCII dump.
        
        Checks the first 9 datasamples of each entry in trace_list of 
        gaps.mseed. The values are assumed to be correct. The first values
        were created using Pitsa.
        """
        mseed_file = os.path.join(self.path, u'gaps.mseed')
        mseed = libmseed()
        # list of known data samples
        starttime = [1199145599915000L, 1199145604035000L, 1199145610215000L,
                     1199145618455000L]
        datalist = [[-363, -382, -388, -420, -417, -397, -418, -390, -388],
                    [-427, -416, -393, -430, -426, -407, -401, -422, -439],
                    [-396, -399, -387, -384, -393, -380, -365, -394, -426],
                    [-389, -428, -409, -389, -388, -405, -390, -368, -368]]
        #trace_list = mseed.readMSTracesViaRecords(mseed_file)
        i = 0
        trace_list = mseed.readMSTraces(mseed_file)
        #import pdb;pdb.set_trace()
        for header, data in trace_list:
            self.assertEqual('BGLD', header['station'])
            self.assertEqual('EHE', header['channel'])
            self.assertEqual(200, header['samprate'])
            self.assertEqual(starttime[i], header['starttime'])
            self.assertEqual(datalist[i], data[0:9].tolist())
            i += 1
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_record',
                           u'qualityflags.mseed', u'test.mseed',
                           u'timingquality.mseed']
        samprate = [200.0, 200.0, 40.0, 200.0]
        station = ['BGLD', 'BGLD', 'HGN', 'BGLD']
        npts = [412, 412, 11947, 41604, 1]
        for i, _f in enumerate(mseed_filenames):
            file = os.path.join(self.path, _f)
            trace_list = mseed.readMSTraces(file)
            header = trace_list[0][0]
            self.assertEqual(samprate[i], header['samprate'])
            self.assertEqual(station[i], header['station'])
            self.assertEqual(npts[i], len(trace_list[0][1]))
        del trace_list, header, data

    def test_readHeader(self):
        """
        Compares header data read by libmseed
        """
        mseed = libmseed()
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_record',
                           u'gaps.mseed', u'qualityflags.mseed',
                           u'test.mseed', u'timingquality.mseed']
        samprate = [200.0, 200.0, 200.0, 40.0, 200.0]
        station = ['BGLD', 'BGLD', 'BGLD', 'HGN', 'BGLD']
        starttime = [1199145599915000, 1199145599915000, 1199145599915000,
                     1054174402043400L, 1199145599765000]
        for _i, _f in enumerate(mseed_filenames):
            file = os.path.join(self.path, _f)
            trace_list = mseed.readMSTraces(file)
            header = trace_list[0][0]
            self.assertEqual(samprate[_i], header['samprate'])
            self.assertEqual(station[_i], header['station'])
            self.assertEqual(starttime[_i], header['starttime'])
        del trace_list, header

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
        # index are now changed, 0 contains head of trace[0]
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
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(gap_list[1:], [])
        # real example with a single gap
        filename = os.path.join(self.path, u'BW.RJOB.__.EHZ.D.2009.056')
        gap_list = mseed.getGapList(filename)
        self.assertEqual(len(gap_list), 1)

    def test_readFirstHeaderInfo(self):
        """
        Reads and compares header info from the first record.
        
        The values can be read from the filename.
        """
        mseed = libmseed()
        # Example 1
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        self.assertEqual(header['channel'], 'EHE')
        # Example 2
        filename = os.path.join(self.path, u'BW.RJOB.__.EHZ.D.2009.056')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'RJOB')
        self.assertEqual(header['channel'], 'EHZ')

    def test_readFirstHeaderInfo2(self):
        """
        Reads and compares header info from the first record.
        Tests method using ms_readmsr_r. Multiple readings in order to see
        if memory leaks arrise.
        
        The values can be read from the filename.
        """
        mseed = libmseed()
        # Example 1
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        header = mseed.getFirstRecordHeaderInfo2(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        # Example 2
        filename = os.path.join(self.path, u'BW.RJOB.__.EHZ.D.2009.056')
        header = mseed.getFirstRecordHeaderInfo2(filename)
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'RJOB')
        self.assertEqual(header['channel'], 'EHZ')
        # Example 3 again for leak checking
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        header = mseed.getFirstRecordHeaderInfo2(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        del header

    def test_getStartAndEndTime2(self):
        """
        Tests getting the start- and end time of a file.
        
        The values are compared with the readFileToTraceGroup() method which 
        parses the whole file. This will only work for files with only one
        trace and without any gaps or overlaps.
        """
        mseed = libmseed()
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_10_percent',
                           u'test.mseed', u'timingquality.mseed']
        for _i in mseed_filenames:
            filename = os.path.join(self.path, _i)
            # get the start- and end time
            (start, end) = mseed.getStartAndEndTime2(filename)
            # parse the whole file
            mstg = mseed.readFileToTraceGroup(filename, dataflag=0)
            chain = mstg.contents.traces.contents
            self.assertEqual(start,
                             mseed._convertMSTimeToDatetime(chain.starttime))
            self.assertEqual(end,
                             mseed._convertMSTimeToDatetime(chain.endtime))
            clibmseed.mst_freegroup(C.pointer(mstg))
            del mstg, chain

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        
        The values are compared with the readFileToTraceGroup() method which 
        parses the whole file. This will only work for files with only one
        trace and without any gaps or overlaps.
        """
        mseed = libmseed()
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_10_percent',
                           u'test.mseed', u'timingquality.mseed']
        for _i in mseed_filenames:
            filename = os.path.join(self.path, _i)
            # get the start- and end time
            (start, end) = mseed.getStartAndEndTime(filename)
            # parse the whole file
            mstg = mseed.readFileToTraceGroup(filename, dataflag=0)
            chain = mstg.contents.traces.contents
            self.assertEqual(start,
                             mseed._convertMSTimeToDatetime(chain.starttime))
            self.assertEqual(end,
                             mseed._convertMSTimeToDatetime(chain.endtime))
            clibmseed.mst_freegroup(C.pointer(mstg))
            del mstg, chain

    def test_getMSStarttime(self):
        """
        Tests getting the starttime of a record.
        
        The values are compared with the readFileToTraceGroup() method which 
        parses the whole file.
        """
        mseed = libmseed()
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_10_percent',
                           u'gaps.mseed', u'qualityflags.mseed', u'test.mseed',
                           u'timingquality.mseed']
        for _i in mseed_filenames:
            filename = os.path.join(self.path, _i)
            # get the start- and end time
            f = open(filename, 'rb')
            start = mseed._getMSStarttime(f)
            f.close()
            # parse the whole file
            mstg = mseed.readFileToTraceGroup(filename, dataflag=0)
            chain = mstg.contents.traces.contents
            self.assertEqual(start,
                             mseed._convertMSTimeToDatetime(chain.starttime))
            clibmseed.mst_freegroup(C.pointer(mstg))
            del mstg, chain

    def test_cutMSFileByRecord(self):
        """
        Tests file cutting on a record basis. 
        
        The cut file is compared to a manually cut file which start and end 
        times will be read
        """
        mseed = libmseed()
        temp = os.path.join(self.path, u'BW.BGLD.__.EHE.D.2008.001')
        file = temp + '.first_10_percent'
        # initialize first record
        file1 = temp + '.first_record'
        start1, end1 = mseed.getStartAndEndTime(file1)
        self.assertEqual(start1, UTCDateTime(2007, 12, 31, 23, 59, 59, 915000))
        self.assertEqual(end1, UTCDateTime(2008, 1, 1, 0, 0, 1, 970000))
        record1 = open(file1, 'rb').read()
        # initialize second record
        file2 = temp + '.second_record'
        start2, end2 = mseed.getStartAndEndTime(file2)
        self.assertEqual(start2, UTCDateTime(2008, 1, 1, 0, 0, 1, 975000))
        self.assertEqual(end2, UTCDateTime(2008, 1, 1, 0, 0, 4, 30000))
        record2 = open(file2, 'rb').read()
        # initialize third record
        file3 = temp + '.third_record'
        start3, end3 = mseed.getStartAndEndTime(file3)
        self.assertEqual(start3, UTCDateTime(2008, 1, 1, 0, 0, 4, 35000))
        self.assertEqual(end3, UTCDateTime(2008, 1, 1, 0, 0, 6, 90000))
        record3 = open(file3, 'rb').read()
        # Cut first record using fixed start and end time
        data = mseed.cutMSFileByRecords(file, starttime=start1, endtime=end1)
        self.assertEqual(data, record1)
        # Cut first record using end time with rounding error and no given
        # start time
        end = UTCDateTime(2008, 1, 1, 0, 0, 1, 969999)
        data = mseed.cutMSFileByRecords(file, endtime=end)
        self.assertEqual(data, record1)
        # Cut first two records using start time with rounding error
        start = start2 - 0.000001
        end = start2 + 0.000001
        data = mseed.cutMSFileByRecords(file, starttime=start, endtime=end)
        self.assertEqual(data, record1 + record2)
        # Cut second record without rounding error
        start = UTCDateTime(2008, 1, 1, 0, 0, 1, 975000)
        end = UTCDateTime(2008, 1, 1, 0, 0, 1, 975001)
        data = mseed.cutMSFileByRecords(file, starttime=start, endtime=end)
        self.assertEqual(data, record2)
        # Cut first three records using times between records
        start = end1 + 0.0025
        end = start3 - 0.0025
        data = mseed.cutMSFileByRecords(file, starttime=start, endtime=end)
        self.assertEqual(data, record1 + record2 + record3)
        # Cut nothing if end time is equal or less than start time
        data = mseed.cutMSFileByRecords(file1, endtime=start1)
        self.assertEqual(data, '')
        data = mseed.cutMSFileByRecords(file1, endtime=start1 - 1)
        self.assertEqual(data, '')
        # Cut nothing if start time is equal or more than end time
        data = mseed.cutMSFileByRecords(file1, starttime=end1)
        self.assertEqual(data, '')
        data = mseed.cutMSFileByRecords(file1, starttime=end1 + 1)
        self.assertEqual(data, '')

    def test_mergeAndCutMSFiles(self):
        """
        Creates ten small files, randomizes their order, merges the middle
        eight files and compares it to the desired result.
        """
        mseed = libmseed()
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
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
        file_list.sort(key=lambda _x: random.random())
        # Get the needed start- and endtime.
        info = mseed._getMSFileInfo(filename)
        open_file = open(filename, 'rb')
        open_file.seek(info['record_length'])
        starttime = mseed._getMSStarttime(open_file)
        open_file.seek(9 * info['record_length'])
        endtime = mseed._getMSStarttime(open_file)
        open_file.close()
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
        # No timing quality set should result in an emtpy dictionary.
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        a = time.time()
        tq = mseed.getTimingQuality(filename)
        b = time.time()
        self.assertEqual(tq, {})
        # It should be much slower when reading every record is parsed even if
        # no timing quality is set. This test should work regardless of
        # current CPU load due to the enormous difference in times.
        c = time.time()
        tq = mseed.getTimingQuality(filename, first_record=False)
        d = time.time()
        self.assertEqual(tq, {})
        self.assertTrue(d - c > 10 * (b - a))

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
        mseed_filenames = [u'BW.BGLD.__.EHE.D.2008.001.first_10_percent',
                           u'gaps.mseed', u'qualityflags.mseed', u'test.mseed',
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
            filename = os.path.join(self.dir, _i)
            isMSEED = mseed.isMSEED(filename)
            self.assertFalse(isMSEED)

    def test_calculateSamplingRate(self):
        """
        Tests calulating the sample rate using the examples in the SEED manual
        page 100. The sample rate always should be a float.
        """
        mseed = libmseed()
        self.assertEqual(mseed._calculateSamplingRate(33, 10), 330)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(33, 10), \
                                   float))
        self.assertEqual(mseed._calculateSamplingRate(330, 1), 330)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(330, 1), \
                                   float))
        self.assertEqual(mseed._calculateSamplingRate(3306, -10), 330.6)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(3306, -10), \
                                   float))
        self.assertEqual(mseed._calculateSamplingRate(-60, 1), float(1) / 60)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(-60, 1), \
                                   float))
        self.assertEqual(mseed._calculateSamplingRate(1, -10), 0.1)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(1, -10), \
                                   float))
        self.assertEqual(mseed._calculateSamplingRate(-10, 1), 0.1)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(-10, 1), \
                                   float))
        self.assertEqual(mseed._calculateSamplingRate(-1, -10), 0.1)
        self.assertTrue(isinstance(mseed._calculateSamplingRate(-1, -10), \
                                   float))

    def test_getMSFileInfo(self):
        """
        Tests the getMSFileInfo method with known values.
        """
        mseed = libmseed()
        filename = os.path.join(self.path,
                                u'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        # Simply reading the file.
        info = mseed._getMSFileInfo(filename)
        self.assertEqual(info['filesize'], 2201600)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 4300)
        self.assertEqual(info['excess_bytes'], 0)
        # Now with an open file. This should work regardless of the current
        # value of the file pointer and it should also not change the file
        # pointer.
        open_file = open(filename, 'rb')
        open_file.seek(1234)
        info = mseed._getMSFileInfo(open_file, filename)
        self.assertEqual(info['filesize'], 2201600)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 4300)
        self.assertEqual(info['excess_bytes'], 0)
        self.assertEqual(open_file.tell(), 1234)
        open_file.close()
        # Now test with a StringIO with the first ten percent.
        open_file = open(filename, 'rb')
        open_file_string = StringIO(open_file.read())
        open_file.close()
        open_file_string.seek(111)
        info = mseed._getMSFileInfo(open_file_string, filename)
        self.assertEqual(info['filesize'], 2201600)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 4300)
        self.assertEqual(info['excess_bytes'], 0)
        self.assertEqual(open_file_string.tell(), 111)
        # One more file containing two records.
        filename = os.path.join(self.path, u'test.mseed')
        info = mseed._getMSFileInfo(filename)
        self.assertEqual(info['filesize'], 8192)
        self.assertEqual(info['record_length'], 4096)
        self.assertEqual(info['number_of_records'], 2)
        self.assertEqual(info['excess_bytes'], 0)

    def test_readMSTracesViaRecords_thread_safety(self):
        """
        Tests for race conditions. Reading n_threads (currently 30) times
        the same mseed file in parallel and compare the results which must
        be all the same.
        
        Fails with readMSTracesViaRecords and passes with readMSTraces!
        """
        n_threads = 30
        mseed = libmseed()
        # Use a medium sized file.
        mseed_file = os.path.join(self.path, u'test.mseed')
        # Read file into memory.
        f = open(mseed_file, 'rb')
        buffer = f.read()
        f.close()
        def test_function(_i, values):
            temp_file = os.path.join(self.path,
                                u'temp_file_' + str(_i))
            # CHANGE FUNCTION TO BE TESTED HERE!
            values[_i] = mseed.readMSTracesViaRecords(temp_file)
        # Create the same file twenty times in a row.
        for _i in xrange(n_threads):
            temp_file = os.path.join(self.path,
                                u'temp_file_' + str(_i))
            f = open(temp_file, 'wb')
            f.write(buffer)
            f.close()
        # Create empty dict for storing the values
        values = {}
        # Read the ten files at one and save the output in the just created
        # class.
        for _i in xrange(n_threads):
            thread = threading.Thread(target=test_function, args=(_i, values))
            thread.start()
        start = time.time()
        # Loop until all threads are finished.
        while True:
            if threading.activeCount() == 1:
                break
            # Avoid infinite loop and leave after 10 seconds which should be
            # enough for any more or less modern computer.
            elif time.time() - start >= 10:
                msg = 'Not all threads finished!'
                raise Warning(msg)
                break
            else:
                continue
        # Compare all values which should be identical and clean up files
        for _i in xrange(n_threads - 1):
            self.assertEqual(values[_i][0][0],
                             values[_i + 1][0][0])
            N.testing.assert_array_equal(values[_i][0][1],
                                         values[_i + 1][0][1])
            temp_file = os.path.join(self.path,
                                u'temp_file_' + str(_i))
            os.remove(temp_file)
        temp_file = os.path.join(self.path, u'temp_file_' + str(_i + 1))
        os.remove(temp_file)


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
