# -*- coding: utf-8 -*-
"""
The libmseed test suite.
"""

from StringIO import StringIO
from obspy.core import UTCDateTime, read
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.core.util import NamedTemporaryFile
from obspy.mseed import LibMSEED
from obspy.mseed.headers import PyFile_FromFile, HPTMODULUS, ENCODINGS, MSRecord
from obspy.mseed.libmseed import clibmseed, _MSStruct
import copy
import ctypes as C
import numpy as np
import os
import random
import sys
import unittest
import struct


class LibMSEEDTestCase(unittest.TestCase):
    """
    Test cases for the libmseed.
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(__file__)
        self.path = os.path.join(self.dir, 'data')
        # mseed steim compression is big endian
        if sys.byteorder == 'little':
            self.swap = 1
        else:
            self.swap = 0

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
                                  'BW.BGLD.__.EHE.D.2008.001.first_10_records')
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
        options. A re-encoded SEED file should still have the same values 
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

    def test_readingFileformatInformation(self):
        """
        Tests the reading of Mini-SEED file format information.
        """
        # Build encoding strings.
        encoding_strings = {}
        for key, value in ENCODINGS.iteritems():
            encoding_strings[value[0]] = key
        __libmseed__ = LibMSEED()
        # Test the encodings and byteorders.
        path = os.path.join(self.path, "encoding")
        files = ['float32_Float32_bigEndian.mseed',
                 'float32_Float32_littleEndian.mseed',
                 'float64_Float64_bigEndian.mseed',
                 'float64_Float64_littleEndian.mseed',
                 'fullASCII_bigEndian.mseed', 'fullASCII_littleEndian.mseed',
                 'int16_INT16_bigEndian.mseed',
                 'int16_INT16_littleEndian.mseed',
                 'int32_INT32_bigEndian.mseed',
                 'int32_INT32_littleEndian.mseed',
                 'int32_Steim1_bigEndian.mseed',
                 'int32_Steim1_littleEndian.mseed',
                 'int32_Steim2_bigEndian.mseed',
                 'int32_Steim2_littleEndian.mseed']
        for file in files:
            info = __libmseed__.getFileformatInformation(os.path.join(path,
                                                                      file))
            if not 'ASCII' in file:
                encoding = file.split('_')[1].upper()
                byteorder = file.split('_')[2].split('.')[0]
            else:
                encoding = 'ASCII'
                byteorder = file.split('_')[1].split('.')[0]
            if 'big' in byteorder:
                byteorder = 1
            else:
                byteorder = 0
            self.assertEqual(encoding_strings[encoding], info['encoding'])
            self.assertEqual(byteorder, info['byteorder'])
            # Also test the record length although it is equal for all files.
            self.assertEqual(256, info['reclen'])
        # No really good test files for the record length so just two files
        # with known record lengths are tested.
        info = __libmseed__.getFileformatInformation(os.path.join(self.path,
                                                      'timingquality.mseed'))
        self.assertEqual(info['reclen'], 512)
        info = __libmseed__.getFileformatInformation(os.path.join(self.path,
                                                      'steim2.mseed'))
        self.assertEqual(info['reclen'], 4096)

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

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and end time of a file.
        
        The values are compared with the readFileToTraceGroup() method which 
        parses the whole file. This will only work for files with only one
        trace and without any gaps or overlaps.
        """
        mseed = LibMSEED()
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_records',
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
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        tq = mseed.getTimingQuality(filename)
        self.assertEqual(tq, {})
        tq = mseed.getTimingQuality(filename, first_record=False)
        self.assertEqual(tq, {})

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
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_records',
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
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        # Simply reading the file.
        f = open(filename, 'rb')
        info = mseed._getMSFileInfo(f, filename)
        f.close()
        self.assertEqual(info['filesize'], 5120)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 10)
        self.assertEqual(info['excess_bytes'], 0)
        # Now with an open file. This should work regardless of the current
        # value of the file pointer and it should also not change the file
        # pointer.
        open_file = open(filename, 'rb')
        open_file.seek(1234)
        info = mseed._getMSFileInfo(open_file, filename)
        self.assertEqual(info['filesize'], 5120)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 10)
        self.assertEqual(info['excess_bytes'], 0)
        self.assertEqual(open_file.tell(), 1234)
        open_file.close()
        # Now test with a StringIO with the first ten percent.
        open_file = open(filename, 'rb')
        open_file_string = StringIO(open_file.read())
        open_file.close()
        open_file_string.seek(111)
        info = mseed._getMSFileInfo(open_file_string, filename)
        self.assertEqual(info['filesize'], 5120)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 10)
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
        self.assertRaises(ArgumentError, cl.ms_detect, *args[:4])
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

        Reference start and endtimes are obtained from the tracegroup.
        Both cases, with and without ms_p argument are tested.
        """
        filename = os.path.join(self.path,
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        start, end = [1199145599915000L, 1199145620510000L]
        # start and endtime
        ms = _MSStruct(filename, init_msrmsf=False)
        ms.read(-1, 0, 1, 0)
        self.assertEqual(start, clibmseed.msr_starttime(ms.msr))
        ms.offset = ms.filePosFromRecNum(-1)
        ms.read(-1, 0, 1, 0)
        self.assertEqual(end, clibmseed.msr_endtime(ms.msr))
        del ms # for valgrind

    def test_unpackSteim2(self):
        """
        Test decompression of Steim2 strings. Remove 128 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        mseed = LibMSEED()
        steim2_file = os.path.join(self.path, 'steim2.mseed')
        data_string = open(steim2_file, 'rb').read()[128:] #128 Bytes header
        data = mseed.unpack_steim2(data_string, 5980, swapflag=self.swap,
                                   verbose=0)
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
        data = mseed.unpack_steim1(data_string, 412, swapflag=self.swap,
                                   verbose=0)
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
        data = mseed.unpack_steim2(data_string, 5980, swapflag=self.swap,
                                   verbose=0)
        # test readMSTraces
        data_record2 = mseed.readMSTraces(file)[0][1]
        np.testing.assert_array_equal(data, data_record2)
        # test readMSTracesViaRecords
        data_record1 = mseed.readMSTracesViaRecords(file)[0][1]
        np.testing.assert_array_equal(data, data_record1)

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

    def test_bugWriteReadFloat32SEEDWin32(self):
        """
        Test case for issue #64.
        """
        # create stream object
        data = np.array([395.07809448, 395.0782, 1060.28112793, -1157.37487793,
                         - 1236.56237793, 355.07028198, -1181.42175293],
                        dtype=np.float32)
        st = Stream([Trace(data=data)])
        tempfile = NamedTemporaryFile().name
        st.write(tempfile, format="MSEED")
        # read temp file directly without libmseed
        bin_data = open(tempfile, "rb").read()
        bin_data = np.array(struct.unpack(">7f", bin_data[56:84]))
        np.testing.assert_array_equal(data, bin_data)
        # read via libmseed
        mseed = LibMSEED()
        # using readMSTraces
        trace_list = mseed.readMSTraces(tempfile, verbose=1)
        # using readMSTracesViaRecords
        trace_list2 = mseed.readMSTracesViaRecords(tempfile, verbose=1)
        # read via ObsPy
        st2 = read(tempfile)
        os.remove(tempfile)
        # test results
        np.testing.assert_array_equal(data, st2[0].data)
        np.testing.assert_array_equal(data, trace_list[0][1])
        np.testing.assert_array_equal(data, trace_list2[0][1])


    def test_msrParse(self):
        """
        Demonstrates how to actually read an msr_record from an
        Python object similar to StringIO. It can be usefull when directly
        transferring or receiving MiniSEED records from the web a database
        etc. The core code of this test is extracted from libmseed.py line
        266. If implementing this functinality, probably the best way is
        therefore to use the readMSTracesViaRecords function.
        """
        msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        msfile = os.path.join(self.path,
                              'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        pyobj = np.fromfile(msfile, dtype='b')
        
        errcode = clibmseed.msr_parse(pyobj.ctypes.data_as(C.POINTER(C.c_char)),
                                      len(pyobj), C.pointer(msr), -1, 1, 1)
        self.assertEquals(errcode, 0)
        chain = msr.contents
        header = LibMSEED()._convertMSRToDict(chain)
        delta = HPTMODULUS / float(header['samprate'])
        header['endtime'] = long(header['starttime'] + delta * \
                                  (header['numsamples'] - 1))
        # Access data directly as NumPy array.
        data = LibMSEED()._ctypesArray2NumpyArray(chain.datasamples,
                                                  chain.numsamples,
                                                  chain.sampletype)
        st = read(msfile)
        np.testing.assert_array_equal(data, st[0].data[:len(data)])


def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
