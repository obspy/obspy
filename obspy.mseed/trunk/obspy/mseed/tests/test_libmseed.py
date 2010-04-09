# -*- coding: utf-8 -*-
"""
The libmseed test suite.
"""

from StringIO import StringIO
from obspy.core import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.core.util import NamedTemporaryFile
from obspy.mseed import LibMSEED
from obspy.mseed.headers import PyFile_FromFile, HPTMODULUS
from obspy.mseed.libmseed import clibmseed, MSStruct
import copy
import ctypes as C
import inspect
import numpy as np
import os
import random
import sys
import threading
import time
import unittest


class LibMSEEDTestCase(unittest.TestCase):
    """
    Test cases for the libmseed.
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(self.dir, 'data')
        # mseed steim compression is big endian
        if sys.byteorder == 'little':
            self.swap = 1
        else:
            self.swap = 0

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
        mseed = LibMSEED()
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
        random.seed(815) # make test reproducable
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
        mseed_file = os.path.join(self.path, unicode('gaps.mseed'))
        mseed = LibMSEED()
        # list of known data samples
        starttime = [1199145599915000L, 1199145604035000L, 1199145610215000L,
                     1199145618455000L]
        datalist = [[-363, -382, -388, -420, -417, -397, -418, -390, -388],
                    [-427, -416, -393, -430, -426, -407, -401, -422, -439],
                    [-396, -399, -387, -384, -393, -380, -365, -394, -426],
                    [-389, -428, -409, -389, -388, -405, -390, -368, -368]]
        i = 0
        trace_list = mseed.readMSTracesViaRecords(mseed_file)
        for header, data in trace_list:
            self.assertEqual('BGLD', header['station'])
            self.assertEqual('EHE', header['channel'])
            self.assertEqual(200, header['samprate'])
            self.assertEqual(starttime[i], header['starttime'])
            self.assertEqual(datalist[i], data[0:9].tolist())
            i += 1
        del trace_list, header, data
        mseed_filenames = [unicode('BW.BGLD.__.EHE.D.2008.001.first_record'),
                           unicode('qualityflags.mseed'),
                           unicode('test.mseed'),
                           unicode('timingquality.mseed')]
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
        mseed_file = os.path.join(self.path, unicode('gaps.mseed'))
        mseed = LibMSEED()
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
        for header, data in trace_list:
            self.assertEqual('BGLD', header['station'])
            self.assertEqual('EHE', header['channel'])
            self.assertEqual(200, header['samprate'])
            self.assertEqual(starttime[i], header['starttime'])
            self.assertEqual(datalist[i], data[0:9].tolist())
            i += 1
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_record',
                           'qualityflags.mseed', 'test.mseed',
                           'timingquality.mseed']
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

    def test_readMSTraces_window(self):
        """
        Tests reading only the first couple of samples.
        """
        mseed_file = os.path.join(self.path,
                                  'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        mseed = LibMSEED()
        # Read the usual way.
        trace_list = mseed.readMSTraces(mseed_file)
        self.assertEqual(len(trace_list), 1)
        org_data = trace_list[0][1]
        # Get start list and convert it.
        start = trace_list[0][0]['starttime']
        start = UTCDateTime(start / float(HPTMODULUS))
        end = trace_list[0][0]['endtime']
        end = UTCDateTime(end / float(HPTMODULUS))
        # Read the first 10 seconds.
        trace_list2 = mseed.readMSTraces(mseed_file, starttime=start,
                                        endtime=start + 10)
        new_data = trace_list2[0][1]
        # Make sure the array is actually smaller.
        self.assertTrue(len(org_data) > len(new_data))
        # Assert the arrays.
        np.testing.assert_array_equal(org_data[:len(new_data)], new_data)
        # Read a last time and read the whole file.
        trace_list3 = mseed.readMSTraces(mseed_file, starttime=start,
                                        endtime=end)
        np.testing.assert_array_equal(org_data, trace_list3[0][1])
        del trace_list, trace_list2, trace_list3

    def test_readHeader(self):
        """
        Compares header data read by libmseed
        """
        mseed = LibMSEED()
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_record',
                           'gaps.mseed', 'qualityflags.mseed',
                           'test.mseed', 'timingquality.mseed']
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
        Note: Test currently only tests the first trace
        """
        mseed = LibMSEED()
        mseed_file = os.path.join(self.path, 'test.mseed')
        trace_list = mseed.readMSTraces(mseed_file)
        data = trace_list[0][1].copy()
        # define test ranges
        record_length_values = [2 ** i for i in range(8, 21)]
        encoding_values = {0: "a", 3: "i", 4: "f", 5: "d",
                           10: "i", 11: "i"}
        byteorder_values = [0, 1]
        # deletes the data quality indicators
        testheader = trace_list[0][0].copy()
        del testheader['dataquality']
        # loops over all combinations of test values
        for reclen in record_length_values:
            for byteorder in byteorder_values:
                for encoding in encoding_values.keys():
                    trace_list[0][1] = data.astype(encoding_values[encoding])
                    trace_list[0][0]['sampletype'] = encoding_values[encoding]
                    temp_file = NamedTemporaryFile().name
                    mseed.writeMSTraces(copy.deepcopy(trace_list), temp_file,
                                        encoding=encoding, byteorder=byteorder,
                                        reclen=reclen)
                    new_trace_list = mseed.readMSTraces(temp_file)
                    del new_trace_list[0][0]['dataquality']
                    testheader['sampletype'] = encoding_values[encoding]
                    self.assertEqual(testheader, new_trace_list[0][0])
                    np.testing.assert_array_equal(trace_list[0][1],
                                                  new_trace_list[0][1])
                    os.remove(temp_file)

    def test_readAndWriteFileWithGaps(self):
        """
        Tests reading and writing files with more than one trace.
        """
        mseed = LibMSEED()
        filename = os.path.join(self.path, 'gaps.mseed')
        # Read file and test if all traces are being read.
        trace_list = mseed.readMSTraces(filename)
        self.assertEqual(len(trace_list), 4)
        # Write File to temporary file.
        outfile = NamedTemporaryFile().name
        mseed.writeMSTraces(copy.deepcopy(trace_list), outfile)
        # Read the same file again and compare it to the original file.
        new_trace_list = mseed.readMSTraces(outfile)
        self.assertEqual(len(trace_list), len(new_trace_list))
        # Compare new_trace_list with trace_list
        for _i in xrange(len(trace_list)):
            self.assertEqual(trace_list[_i][0], new_trace_list[_i][0])
            np.testing.assert_array_equal(trace_list[_i][1],
                                         new_trace_list[_i][1])
        os.remove(outfile)

    def test_readFirstHeaderInfo(self):
        """
        Reads and compares header info from the first record.
        
        The values can be read from the filename.
        """
        mseed = LibMSEED()
        # Example 1
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        self.assertEqual(header['channel'], 'EHE')
        # Example 2 again for leak checking
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        header = mseed.getFirstRecordHeaderInfo(filename)
        self.assertEqual(header['location'], '')
        self.assertEqual(header['network'], 'BW')
        self.assertEqual(header['station'], 'BGLD')
        del header


    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        
        The values are compared with the readFileToTraceGroup() method which 
        parses the whole file. This will only work for files with only one
        trace and without any gaps or overlaps.
        """
        mseed = LibMSEED()
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_percent',
                           'test.mseed', 'timingquality.mseed']
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


    def test_cutMSFileByRecord(self):
        """
        Tests file cutting on a record basis. 
        
        The cut file is compared to a manually cut file which start and end 
        times will be read
        """
        mseed = LibMSEED()
        temp = os.path.join(self.path, 'BW.BGLD.__.EHE.D.2008.001')
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
        mseed = LibMSEED()
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        #Create 10 small files.
        fh = open(filename, 'rb')
        first_ten_records = fh.read(10 * 512)
        fh.close()
        file_list = []
        for _i in range(10):
            tempfile = NamedTemporaryFile().name
            file_list.append(tempfile)
            fh = open(tempfile, 'wb')
            fh.write(first_ten_records[_i * 512: (_i + 1) * 512])
            fh.close()
        # Randomize list.
        file_list.sort(key=lambda _x: random.random())
        # Init MSRecord and MSFileParam structure
        ms = MSStruct(filename)
        # Get the needed start- and endtime.
        info = mseed._getMSFileInfo(ms.f, ms.file)
        ms.f.seek(info['record_length'])
        starttime = ms.getStart()
        ms.f.seek(9 * info['record_length'])
        endtime = ms.getStart()
        del ms # for valgrind
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
        mseed = LibMSEED()
        filename = os.path.join(self.path, 'qualityflags.mseed')
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
        mseed = LibMSEED()
        filename = os.path.join(self.path, 'timingquality.mseed')
        tq = mseed.getTimingQuality(filename)
        self.assertEqual(tq, {'min': 0.0, 'max': 100.0, 'average': 50.0,
                              'median': 50.0, 'upper_quantile': 75.0,
                              'lower_quantile': 25.0})
        # No timing quality set should result in an empty dictionary.
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
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
        mseed = LibMSEED()
        # Mini-SEED filenames.
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_percent',
                           'gaps.mseed', 'qualityflags.mseed', 'test.mseed',
                           'timingquality.mseed']
        # Non Mini-SEED filenames.
        non_mseed_filenames = ['test_libmseed.py', '__init__.py',
                               'test_core.py']
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


    def test_getMSFileInfo(self):
        """
        Tests the getMSFileInfo method with known values.
        """
        mseed = LibMSEED()
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        # Simply reading the file.
        f = open(filename, 'rb')
        info = mseed._getMSFileInfo(f, filename)
        f.close()
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
        filename = os.path.join(self.path, 'test.mseed')
        f = open(filename, 'rb')
        info = mseed._getMSFileInfo(f, filename)
        f.close()
        self.assertEqual(info['filesize'], 8192)
        self.assertEqual(info['record_length'], 4096)
        self.assertEqual(info['number_of_records'], 2)
        self.assertEqual(info['excess_bytes'], 0)

    def test_ctypesArgtypes(self):
        """
        Test that ctypes argtypes are set for type checking
        """
        ArgumentError = C.ArgumentError
        cl = clibmseed
        args = [C.pointer(C.pointer(C.c_int())), 'a', 1, 1.5, 1, 0, 0, 0, 0]
        self.assertRaises(ArgumentError, cl.ms_readtraces, *args)
        self.assertRaises(TypeError, cl.ms_readtraces, *args[:-1])
        self.assertRaises(ArgumentError, cl.ms_readmsr_r, *args)
        self.assertRaises(TypeError, cl.ms_readmsr_r, *args[:-1])
        self.assertRaises(ArgumentError, cl.mst_printtracelist, *args[:5])
        self.assertRaises(ArgumentError, PyFile_FromFile, *args[:5])
        self.assertRaises(ArgumentError, cl.ms_find_reclen, *args[:4])
        args.append(1) # 10 argument function
        self.assertRaises(ArgumentError, cl.mst_packgroup, *args)
        args = ['hallo'] # one argument functions
        self.assertRaises(ArgumentError, cl.msr_starttime, *args)
        self.assertRaises(ArgumentError, cl.msr_endtime, *args)
        self.assertRaises(ArgumentError, cl.mst_init, *args)
        self.assertRaises(ArgumentError, cl.mst_free, *args)
        self.assertRaises(ArgumentError, cl.mst_initgroup, *args)
        self.assertRaises(ArgumentError, cl.mst_freegroup, *args)
        self.assertRaises(ArgumentError, cl.msr_init, *args)

    def test_readSingleRecordToMSR(self):
        """
        Tests readSingleRecordtoMSR against start and endtimes.

        Reference start and entimes are optained from the tracegroup.
        Both cases, with and without ms_p argument are tested.
        """
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        start, end = [1199145599915000L, 1199151207890000L]
        # start and endtime
        ms = MSStruct(filename)
        self.assertEqual(start, clibmseed.msr_starttime(ms.msr))
        ms.f.seek(ms.filePosFromRecNum(-1))
        ms.read(-1, 0, 1, 0)
        self.assertEqual(end, clibmseed.msr_endtime(ms.msr))
        del ms # for valgrind

    def test_readMSTracesViaRecords_thread_safety(self):
        """
        Tests for race conditions. Reading n_threads (currently 30) times
        the same mseed file in parallel and compare the results which must
        be all the same.
        
        Fails with readMSTracesViaRecords and passes with readMSTraces!
        """
        n_threads = 3
        mseed = LibMSEED()
        # Use a medium sized file.
        mseed_file = os.path.join(self.path,
                                  'BW.BGLD.__.EHE.D.2008.001.first_10_percent')
        # Read file into memory.
        f = open(mseed_file, 'rb')
        buffer = f.read()
        f.close()
        def test_function(_i, values):
            temp_file = os.path.join(self.path, 'temp_file_' + str(_i))
            # CHANGE FUNCTION TO BE TESTED HERE!
            values[_i] = mseed.readMSTracesViaRecords(temp_file)
        # Create the same file twenty times in a row.
        for _i in xrange(n_threads):
            temp_file = os.path.join(self.path, 'temp_file_' + str(_i))
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
            # Avoid infinite loop and leave after 120 seconds 
            # such a long time is needed for debugging with valgrind
            elif time.time() - start >= 120:
                msg = 'Not all threads finished!'
                raise Warning(msg)
                break
            else:
                time.sleep(0.1)
                continue
        # Compare all values which should be identical and clean up files
        for _i in xrange(n_threads - 1):
            self.assertEqual(values[_i][0][0],
                             values[_i + 1][0][0])
            np.testing.assert_array_equal(values[_i][0][1],
                                         values[_i + 1][0][1])
            temp_file = os.path.join(self.path, 'temp_file_' + str(_i))
            os.remove(temp_file)
        temp_file = os.path.join(self.path, 'temp_file_' + str(_i + 1))
        os.remove(temp_file)

    def test_unpackSteim2(self):
        """
        Test decompression of Steim2 strings. Remove 128 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        mseed = LibMSEED()
        steim2_file = os.path.join(self.path, 'steim2.mseed')
        data_string = open(steim2_file, 'rb').read()[128:] #128 Bytes header
        data = mseed.unpack_steim2(data_string, 5980, swapflag=self.swap, verbose=0)
        data_record = mseed.readMSTracesViaRecords(steim2_file)[0][1]
        np.testing.assert_array_equal(data, data_record)

    def test_unpackSteim1(self):
        """
        Test decompression of Steim1 strings. Remove 64 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        mseed = LibMSEED()
        steim1_file = os.path.join(self.path,
                                   'BW.BGLD.__.EHE.D.2008.001.first_record')
        data_string = open(steim1_file, 'rb').read()[64:] #64 Bytes header
        data = mseed.unpack_steim1(data_string, 412, swapflag=self.swap, verbose=0)
        data_record = mseed.readMSTracesViaRecords(steim1_file)[0][1]
        np.testing.assert_array_equal(data, data_record)

    def test_brokenLastRecord(self):
        """
        Test if Libmseed is able to read files with broken last record. Use
        both methods, readMSTracesViaRecords and readMSTraces
        """
        mseed = LibMSEED()
        file = os.path.join(self.path, "brokenlastrecord.mseed")
        # independent reading of the data
        data_string = open(file, 'rb').read()[128:] #128 Bytes header
        data = mseed.unpack_steim2(data_string, 5980, swapflag=self.swap, verbose=0)
        # test readMSTracesViaRecords
        data_record1 = mseed.readMSTracesViaRecords(file)[0][1]
        np.testing.assert_array_equal(data, data_record1)
        # test readMSTraces
        data_record2 = mseed.readMSTraces(file)[0][1]
        np.testing.assert_array_equal(data, data_record2)

    def test_oneSampleOverlap(self):
        """
        Both methods readMSTraces and readMSTracesViaRecords should recognize a
        single sample overlap.
        """
        # create a stream with one sample overlapping
        trace1 = Trace(data=np.zeros(1000))
        trace2 = Trace(data=np.zeros(10))
        trace2.stats.starttime = UTCDateTime(999)
        st = Stream([trace1, trace2])
        # write into MSEED
        tempfile = NamedTemporaryFile().name
        st.write(tempfile, format="MSEED")
        # read it again
        mseed = LibMSEED()
        trace_list = mseed.readMSTraces(tempfile)
        self.assertEquals(len(trace_list), 2)
        trace_list = mseed.readMSTracesViaRecords(tempfile)
        self.assertEquals(len(trace_list), 2)
        # clean up
        os.remove(tempfile)


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
