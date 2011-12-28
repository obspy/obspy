# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read, AttribDict
from obspy.core.util import NamedTemporaryFile
from obspy.mseed import util
from obspy.mseed.headers import clibmseed, ENCODINGS
from obspy.mseed.core import readMSEED, writeMSEED, isMSEED
from obspy.mseed.msstruct import _MSStruct
import copy
import numpy as np
import os
import unittest


class MSEEDReadingAndWritingTestCase(unittest.TestCase):
    """
    Test everything related to the general reading and writing of MiniSEED
    files.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_readHeadFileViaObsPy(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        stream = read(testfile, headonly=True, format='MSEED')
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(str(stream[0].data), '[]')
        # This is controlled by the stream[0].data attribute.
        self.assertEqual(stream[0].stats.npts, 11947)
        gapfile = os.path.join(self.path, 'data', 'gaps.mseed')
        # without given format -> autodetect using extension
        stream = read(gapfile, headonly=True)
        starttime = ['2007-12-31T23:59:59.915000Z',
                     '2008-01-01T00:00:04.035000Z',
                     '2008-01-01T00:00:10.215000Z',
                     '2008-01-01T00:00:18.455000Z']
        self.assertEqual(4, len(stream.traces))
        for _k, _i in enumerate(stream.traces):
            self.assertEqual(True, isinstance(_i, Trace))
            self.assertEqual(str(_i.data), '[]')
            self.assertEqual(str(_i.stats.starttime), starttime[_k])

    def test_readGappyFile(self):
        """
        Compares waveform data read by obspy.mseed with an ASCII dump.

        Checks the first 9 datasamples of each entry in trace_list of
        gaps.mseed. The values are assumed to be correct. The first values
        were created using Pitsa.

        XXX: This tests is a straight port from an old libmseed test. Redundant
        to some other tests.
        """
        mseed_file = os.path.join(self.path, 'data', unicode('gaps.mseed'))
        # list of known data samples
        starttime = [1199145599915000L, 1199145604035000L, 1199145610215000L,
                     1199145618455000L]
        datalist = [[-363, -382, -388, -420, -417, -397, -418, -390, -388],
                    [-427, -416, -393, -430, -426, -407, -401, -422, -439],
                    [-396, -399, -387, -384, -393, -380, -365, -394, -426],
                    [-389, -428, -409, -389, -388, -405, -390, -368, -368]]
        i = 0
        stream = readMSEED(mseed_file)
        for trace in stream:
            self.assertEqual('BGLD', trace.stats.station)
            self.assertEqual('EHE', trace.stats.channel)
            self.assertEqual(200, trace.stats.sampling_rate)
            self.assertEqual(starttime[i],
                util._convertDatetimeToMSTime(trace.stats.starttime))
            self.assertEqual(datalist[i], trace.data[0:9].tolist())
            i += 1
        del stream
        # Also test unicode filenames.
        mseed_filenames = [unicode('BW.BGLD.__.EHE.D.2008.001.first_record'),
                           unicode('qualityflags.mseed'),
                           unicode('test.mseed'),
                           unicode('timingquality.mseed')]
        samprate = [200.0, 200.0, 40.0, 200.0]
        station = ['BGLD', 'BGLD', 'HGN', 'BGLD']
        npts = [412, 412, 11947, 41604, 1]
        for i, _f in enumerate(mseed_filenames):
            filename = os.path.join(self.path, 'data', _f)
            stream = readMSEED(filename)
            self.assertEqual(samprate[i], stream[0].stats.sampling_rate)
            self.assertEqual(station[i], stream[0].stats.station)
            self.assertEqual(npts[i], stream[0].stats.npts)
            self.assertEqual(npts[i], len(stream[0].data))
        del stream

    def test_readAndWriteTraces(self):
        """
        Writes, reads and compares files created via obspy.mseed.

        This uses all possible encodings, record lengths and the byte order
        options. A re-encoded SEED file should still have the same values
        regardless of write options.
        Note: Test currently only tests the first trace
        """
        mseed_file = os.path.join(self.path, 'data', 'test.mseed')
        stream = readMSEED(mseed_file)
        # Define test ranges
        record_length_values = [2 ** i for i in range(8, 21)]
        encoding_values = {"ASCII": "|S1", "INT16": "int16", "INT32": "int32",
                           "FLOAT32": "float32", "FLOAT64": "float64",
                           "STEIM1": "int32", "STEIM2": "int32"}
        byteorder_values = ['>', '<']
        # Loop over every combination.
        for reclen in record_length_values:
            for byteorder in byteorder_values:
                for encoding in encoding_values.keys():
                    this_stream = copy.deepcopy(stream)
                    this_stream[0].data = \
                        np.require(this_stream[0].data,
                                   dtype=encoding_values[encoding])
                    temp_file = NamedTemporaryFile().name

                    writeMSEED(this_stream, temp_file, encoding=encoding,
                               byteorder=byteorder, reclen=reclen)
                    new_stream = readMSEED(temp_file)
                    # Assert the new stream still has the chosen attributes.
                    # This should mean that writing as well as reading them
                    # works.
                    self.assertEqual(new_stream[0].stats.mseed.byteorder,
                                     byteorder)
                    self.assertEqual(new_stream[0].stats.mseed.record_length,
                                     reclen)
                    self.assertEqual(new_stream[0].stats.mseed.encoding,
                                     encoding)

                    np.testing.assert_array_equal(this_stream[0].data,
                                                  new_stream[0].data)
                    os.remove(temp_file)

    def test_readingFileformatInformation(self):
        """
        Tests the reading of Mini-SEED file format information.
        """
        # Build encoding strings.
        encoding_strings = {}
        for key, value in ENCODINGS.iteritems():
            encoding_strings[value[0]] = key
        # Test the encodings and byteorders.
        path = os.path.join(self.path, "data", "encoding")
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
            info = util.getFileformatInformation(os.path.join(path, file))
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
        info = util.getFileformatInformation(os.path.join(self.path, 'data',
                                             'timingquality.mseed'))
        self.assertEqual(info['reclen'], 512)
        info = util.getFileformatInformation(os.path.join(self.path, 'data',
                                             'steim2.mseed'))
        self.assertEqual(info['reclen'], 4096)

    def test_readAndWriteFileWithGaps(self):
        """
        Tests reading and writing files with more than one trace.
        """
        filename = os.path.join(self.path, 'data', 'gaps.mseed')
        # Read file and test if all traces are being read.
        stream = readMSEED(filename)
        self.assertEqual(len(stream), 4)
        # Write File to temporary file.
        outfile = NamedTemporaryFile().name
        writeMSEED(copy.deepcopy(stream), outfile)
        # Read the same file again and compare it to the original file.
        new_stream = readMSEED(outfile)
        self.assertEqual(len(stream), len(new_stream))
        # Compare new_trace_list with trace_list
        for tr1, tr2 in zip(stream, new_stream):
            self.assertEqual(tr1.stats, tr2.stats)
            np.testing.assert_array_equal(tr1.data, tr2.data)
        os.remove(outfile)

    def test_isMSEED(self):
        """
        This tests the isMSEED method by just validating that each file in the
        data directory is a Mini-SEED file and each file in the working
        directory is not a Mini-SEED file.

        The filenames are hard coded so the test will not fail with future
        changes in the structure of the package.
        """
        # Mini-SEED filenames.
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_records',
                           'gaps.mseed', 'qualityflags.mseed', 'test.mseed',
                           'timingquality.mseed']
        # Non Mini-SEED filenames.
        non_mseed_filenames = ['test_mseed_reading_and_writing.py',
                               '__init__.py']
        # Loop over Mini-SEED files
        for _i in mseed_filenames:
            filename = os.path.join(self.path, 'data', _i)
            is_mseed = isMSEED(filename)
            self.assertTrue(is_mseed)
        # Loop over non Mini-SEED files
        for _i in non_mseed_filenames:
            filename = os.path.join(self.path, _i)
            is_mseed = isMSEED(filename)
            self.assertFalse(is_mseed)

    def test_readSingleRecordToMSR(self):
        """
        Tests readSingleRecordtoMSR against start and endtimes.

        Reference start and endtimes are obtained from the tracegroup.
        Both cases, with and without ms_p argument are tested.
        """
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        start, end = [1199145599915000L, 1199145620510000L]
        # start and endtime
        ms = _MSStruct(filename, init_msrmsf=False)
        ms.read(-1, 0, 1, 0)
        self.assertEqual(start, clibmseed.msr_starttime(ms.msr))
        ms.offset = ms.filePosFromRecNum(-1)
        ms.read(-1, 0, 1, 0)
        self.assertEqual(end, clibmseed.msr_endtime(ms.msr))
        del ms  # for valgrind

    def test_readFileViaMSEED(self):
        """
        Read file test via L{obspy.mseed.mseed.readMSEED}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        data = [2787, 2776, 2774, 2780, 2783]
        # Read the file directly to a Stream object.
        stream = readMSEED(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(stream[0].stats.get('location'), '00')
        self.assertEqual(stream[0].stats.npts, 11947)
        self.assertEqual(stream[0].stats['sampling_rate'], 40.0)
        self.assertEqual(stream[0].stats.get('channel'), 'BHZ')
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], data[_i])

    def test_readPartialTimewindowFromFile(self):
        """
        Uses obspy.mseed.mseed.readMSEED to read only read a certain time
        window of a file.
        """
        starttime = UTCDateTime('2007-12-31T23:59:59.915000Z')
        endtime = UTCDateTime('2008-01-01T00:00:20.510000Z')
        testfile = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        stream = readMSEED(testfile, starttime=starttime + 6,
                           endtime=endtime - 6)
        self.assertTrue(starttime < stream[0].stats.starttime)
        self.assertTrue(endtime > stream[0].stats.endtime)

    def test_readPartialWithOnlyStarttimeSet(self):
        """
        Uses obspy.mseed.mseed.readMSEED to read only the data starting with
        a certain time.
        """
        starttime = UTCDateTime('2007-12-31T23:59:59.915000Z')
        endtime = UTCDateTime('2008-01-01T00:00:20.510000Z')
        testfile = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        stream = readMSEED(testfile, starttime=starttime + 6)
        self.assertTrue(starttime < stream[0].stats.starttime)
        self.assertEqual(endtime, stream[0].stats.endtime)

    def test_readPartialWithOnlyEndtimeSet(self):
        """
        Uses obspy.mseed.mseed.readMSEED to read only the data ending before a
        certain time.
        """
        starttime = UTCDateTime('2007-12-31T23:59:59.915000Z')
        endtime = UTCDateTime('2008-01-01T00:00:20.510000Z')
        testfile = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        stream = readMSEED(testfile, endtime=endtime - 6)
        self.assertEqual(starttime, stream[0].stats.starttime)
        self.assertTrue(endtime > stream[0].stats.endtime)

    def test_readPartialFrameWithEmptyTimeRange(self):
        """
        Uses obspy.mseed.mseed.readMSEED to read a partial file with a
        timewindow outside of the actual data. Should return an empty Stream
        object.
        """
        starttime = UTCDateTime('2003-05-29T02:13:22.043400Z')
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        stream = readMSEED(testfile, starttime=starttime - 1E6,
                           endtime=starttime - 1E6 + 1)
        self.assertEqual(len(stream), 0)

    def test_readPartialWithSourceName(self):
        """
        Uses obspy.mseed.mseed.readMSEED to read only part of a file that
        matches certain sourcename patterns.
        """
        testfile = os.path.join(self.path, 'data', 'two_channels.mseed')
        st1 = readMSEED(testfile)
        self.assertEqual(st1[0].stats.channel, 'EHE')
        self.assertEqual(st1[1].stats.channel, 'EHZ')
        st2 = readMSEED(testfile, sourcename='*')
        self.assertEqual(st2[0].stats.channel, 'EHE')
        self.assertEqual(st2[1].stats.channel, 'EHZ')
        st3 = readMSEED(testfile, sourcename='*.EH*')
        self.assertEqual(st3[0].stats.channel, 'EHE')
        self.assertEqual(st3[1].stats.channel, 'EHZ')
        st4 = readMSEED(testfile, sourcename='*E')
        self.assertEqual(st4[0].stats.channel, 'EHE')
        self.assertEqual(len(st4), 1)
        st5 = readMSEED(testfile, sourcename='*.EHZ')
        self.assertEqual(st5[0].stats.channel, 'EHZ')
        self.assertEqual(len(st5), 1)
        st6 = readMSEED(testfile, sourcename='*.BLA')
        self.assertEqual(len(st6), 0)

    def test_readFromStringIO(self):
        """
        Tests reading from a MiniSEED file in an StringIO object.
        """

    def test_writeIntegers(self):
        """
        Write integer array via L{obspy.mseed.mseed.writeMSEED}.
        """
        tempfile = NamedTemporaryFile().name
        npts = 1000
        # data array of integers - float won't work!
        np.random.seed(815)  # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        # write
        writeMSEED(st, tempfile, format="MSEED")
        # read again
        stream = readMSEED(tempfile)
        os.remove(tempfile)
        stream.verify()
        np.testing.assert_array_equal(stream[0].data, data)

    def test_readMSTracesViaRecords_MultipleIds(self):
        """
        Tests a critical issue when the LibMSEED.readMSTracesViaRecords method
        is used (e.g. on Windows systems) and a start/endtime is set and the
        file has multiple ids.

        This is due to the fact that the readMSTraceViaRecords method uses the
        first and the last records of a file to take an educated guess about
        which records to actually read. This of course only works if all
        records have the same id and are chronologically ordered.

        I don't think there is an easy solution for it.
        """
        # The used file has ten records in successive order and then the first
        # record again with a different record id:
        # 2 Trace(s) in Stream:
        #     BW.BGLD..EHE | 2007-12-31T23:59:59.915000Z -
        #     2008-01-01T00:00:20.510000Z | 200.0 Hz, 4120 samples
        #     OB.BGLD..EHE | 2007-12-31T23:59:59.915000Z -
        #     2008-01-01T00:00:01.970000Z | 200.0 Hz, 412 samples
        #
        # Thus reading a small time window in between should contain at least
        # some samples.
        starttime = UTCDateTime(2008, 1, 1, 0, 0, 10)
        endtime = starttime + 5
        file = os.path.join(self.path, 'data',
                            'constructedFileToTestReadViaRecords.mseed')
        # Some samples should be in the time window.
        st = read(file, starttime=starttime, endtime=endtime)
        self.assertEqual(len(st), 1)
        samplecount = st[0].stats.npts
        # 5 seconds are 5s * 200Hz + 1 samples.
        self.assertEqual(samplecount, 1001)
        # Choose time outside of frame.
        st = read(file,
                starttime=UTCDateTime() - 10, endtime=UTCDateTime())
        # Should just result in an empty stream.
        self.assertEqual(len(st), 0)

    def test_writeAndReadDifferentRecordLengths(self):
        """
        Tests Mini-SEED writing and record lengths.
        """
        # libmseed instance.
        npts = 6000
        np.random.seed(815)  # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        record_lengths = [256, 512, 1024, 2048, 4096, 8192]
        # Loop over some record lengths.
        for rec_len in record_lengths:
            # Write it.
            tempfile = NamedTemporaryFile().name
            st.write(tempfile, format="MSEED", reclen=rec_len)
            # Open the file.
            file = open(tempfile, 'rb')
            info = util._getMSFileInfo(file, tempfile)
            file.close()
            # Test reading the two files.
            temp_st = read(tempfile)
            np.testing.assert_array_equal(data, temp_st[0].data)
            del temp_st
            os.remove(tempfile)
            # Check record length.
            self.assertEqual(info['record_length'], rec_len)
            # Check if filesize is a multiple of the record length.
            self.assertEqual(info['filesize'] % rec_len, 0)

    def test_readFullSEED(self):
        """
        Reads a full SEED volume.
        """
        files = os.path.join(self.path, 'data', 'fullseed.mseed')
        st = readMSEED(files)
        self.assertEquals(len(st), 3)
        self.assertEquals(len(st[0]), 602)
        self.assertEquals(len(st[1]), 623)
        self.assertEquals(len(st[2]), 610)

    def test_readWithWildCard(self):
        """
        Reads wildcard filenames.
        """
        files = os.path.join(self.path, 'data',
                             'BW.BGLD.__.EHE.D.2008.001.*_record')
        st = read(files)
        self.assertEquals(len(st), 3)
        st.merge()
        self.assertEquals(len(st), 1)

    def test_Header(self):
        """
        Tests whether the header is correctly written and read.
        """
        tempfile = NamedTemporaryFile().name
        np.random.seed(815)  # make test reproducable
        data = np.random.randint(-1000, 1000, 50).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location': 'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed': {'dataquality': 'D', 'record_length': 512,
                           'encoding': 'STEIM2', 'byteorder': '>'}}
        stats['starttime'] = UTCDateTime(2000, 1, 1)
        st = Stream([Trace(data=data, header=stats)])
        # Write it.
        st.write(tempfile, format="MSEED")
        # Read it again and delete the temporary file.
        stream = read(tempfile)
        os.remove(tempfile)
        stream.verify()
        # Loop over the attributes to be able to assert them because a
        # dictionary is not a stats dictionary.
        # This also assures that there are no additional keys.
        for key in stats.keys():
            self.assertEqual(stats[key], stream[0].stats[key])

    def test_readingAndWritingViaTheStatsAttribute(self):
        """
        Tests the writing with MSEED file attributes set via the attributes in
        trace.stats.mseed.
        """
        npts = 6000
        np.random.seed(815)  # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        # Test all possible combinations of record length, encoding and
        # byteorder.
        record_lengths = [256, 512, 1024, 2048, 4096, 8192]
        byteorders = ['>', '<']
        encodings = [value[0] for value in ENCODINGS.values()]
        np_encodings = {}
        # Special handling for ASCII encoded files.
        for value in ENCODINGS.values():
            if value[0] == 'ASCII':
                np_encodings[value[0]] = np.dtype("|S1")
            else:
                np_encodings[value[0]] = value[2]
        st = Stream([Trace(data=data)])
        st[0].stats.mseed = AttribDict()
        st[0].stats.mseed.dataquality = 'D'
        # Loop over all combinations.
        for reclen in record_lengths:
            for order in byteorders:
                for encoding in encodings:
                    # Create new stream and change stats.
                    stream = copy.deepcopy(st)
                    stream[0].stats.mseed.record_length = reclen
                    stream[0].stats.mseed.byteorder = order
                    stream[0].stats.mseed.encoding = encoding
                    # Convert the data so that it is compatible with the
                    # encoding.
                    stream[0].data = np.require(stream[0].data,
                                        np_encodings[encoding])
                    # Write it.
                    tempfile = NamedTemporaryFile().name
                    stream.write(tempfile, format="MSEED")
                    # Open the file.
                    stream2 = read(tempfile)
                    # Assert the stats.
                    self.assertEqual(stream[0].stats.mseed,
                                     stream2[0].stats.mseed)
                    del stream
                    del stream2
                    os.remove(tempfile)

    def test_readPartsOfFile(self):
        """
        Test reading only parts of an Mini-SEED file without unpacking or
        touching the rest.
        """
        temp = os.path.join(self.path, 'data', 'BW.BGLD.__.EHE.D.2008.001')
        file = temp + '.first_10_records'
        t = [UTCDateTime(2008, 1, 1, 0, 0, 1, 975000),
             UTCDateTime(2008, 1, 1, 0, 0, 4, 30000)]
        tr1 = read(file, starttime=t[0], endtime=t[1])[0]
        self.assertEqual(t[0], tr1.stats.starttime)
        self.assertEqual(t[1], tr1.stats.endtime)
        # initialize second record
        file2 = temp + '.second_record'
        tr2 = read(file2)[0]
        np.testing.assert_array_equal(tr1.data, tr2.data)

    def test_readWithGSE2Option(self):
        """
        Test that reading will still work if wrong option (of gse2)
        verify_chksum is given. This is important if the read method is
        called for an unknown file format.
        """
        file = os.path.join(self.path, 'data', 'BW.BGLD.__.EHE.D.2008.001'
                            '.second_record')
        tr = read(file, verify_chksum=True, starttime=None)[0]
        data = np.array([-397, -387, -368, -381, -388])
        np.testing.assert_array_equal(tr.data[0:5], data)
        self.assertEqual(412, len(tr.data))
        data = np.array([-406, -417, -403, -423, -413])
        np.testing.assert_array_equal(tr.data[-5:], data)

    def test_allDataTypesAndEndiansInMultipleFiles(self):
        """
        Tests writing all different types. This is an test which is independent
        of the read method. Only the data part is verified.
        """
        file = os.path.join(self.path, "data", \
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        tempfile = NamedTemporaryFile().name
        # Read the data and copy them
        st = read(file)
        data_copy = st[0].data.copy()
        # Float64, Float32, Int32, Int24, Int16, Char
        encodings = {5: "f8", 4: "f4", 3: "i4", 0: "S1", 1: "i2"}
        byteorders = {0: '<', 1: '>'}
        for byteorder, btype in byteorders.iteritems():
            for encoding, dtype in encodings.iteritems():
                # Convert data to floats and write them again
                st[0].data = data_copy.astype(dtype)
                st.write(tempfile, format="MSEED", encoding=encoding,
                         reclen=256, byteorder=byteorder)
                # Read the first record of data without header not using ObsPy
                s = open(tempfile, "rb").read()
                data = np.fromstring(s[56:256], dtype=btype + dtype)
                np.testing.assert_array_equal(data, st[0].data[:len(data)])
                # Read the binary chunk of data with ObsPy
                st2 = read(tempfile)
                np.testing.assert_array_equal(st2[0].data, st[0].data)
        os.remove(tempfile)

    def test_SavingSmallASCII(self):
        """
        Tests writing small ASCII strings.
        """
        tempfile = NamedTemporaryFile().name
        st = Stream()
        st.append(Trace(data=np.fromstring("A" * 8, "|S1")))
        st.write(tempfile, format="MSEED")
        os.remove(tempfile)

    def test_allDataTypesAndEndiansInSingleFile(self):
        """
        Tests all data and endian types into a single file.
        """
        tempfile = NamedTemporaryFile().name
        st1 = Stream()
        data = np.random.randint(-1000, 1000, 500)
        for dtype in ["i2", "i4", "f4", "f8", "S1"]:
            for enc in ["<", ">", "="]:
                st1.append(Trace(data=data.astype(np.dtype(enc + dtype))))
        st1.write(tempfile, format="MSEED")
        # read everything back (int16 gets converted into int32)
        st2 = read(tempfile)
        for dtype in ["i4", "i4", "f4", "f8", "S1"]:
            for enc in ["<", ">", "="]:
                tr = st2.pop(0).data
                self.assertEqual(tr.dtype.kind + str(tr.dtype.itemsize), dtype)
                # byte order is always native (=)
                np.testing.assert_array_equal(tr, data.astype("=" + dtype))
        os.remove(tempfile)

    def test_enforceSteim2WithSteim1asEncoding(self):
        """
        This tests whether the encoding kwarg overwrites the encoding in
        trace.stats.mseed.encoding.
        """
        file = os.path.join(self.path, "data", \
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        st = read(file)
        self.assertEqual(st[0].stats.mseed.encoding, 'STEIM1')
        tempfile = NamedTemporaryFile().name
        st.write(tempfile, format='MSEED', encoding='STEIM2')
        st2 = read(tempfile)
        os.remove(tempfile)
        self.assertEqual(st2[0].stats.mseed.encoding, 'STEIM2')

    def test_filesFromLibmseed(self):
        """
        Tests reading of files that are created by libmseed.

        This test also checks the files created by libmseed to some extend.
        """
        path = os.path.join(self.path, "data", "encoding")
        # Dictionary. The key is the filename, the value a tuple: dtype,
        # sampletype, encoding, content
        def_content = np.arange(1, 51, dtype='int32')
        files = {
            os.path.join(path, "smallASCII.mseed"):
                ('|S1', 'a', 0, np.fromstring('ABCDEFGH', dtype='|S1')),
            # Tests all ASCII letters.
            os.path.join(path, "fullASCII.mseed"):
                ('|S1', 'a', 0, np.fromstring(""" !"#$%&'()*+,-./""" + \
                   """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`""" + \
                   """abcdefghijklmnopqrstuvwxyz{|}~""", dtype='|S1')),
            # Note: int16 array will also be returned as int32.
            os.path.join(path, "int16_INT16.mseed"):
                ('int32', 'i', 1, def_content.astype('int16')),
            os.path.join(path, "int32_INT32.mseed"):
                ('int32', 'i', 3, def_content),
            os.path.join(path, "int32_Steim1.mseed"):
                ('int32', 'i', 10, def_content),
            os.path.join(path, "int32_Steim2.mseed"):
                ('int32', 'i', 11, def_content),
            os.path.join(path, "float32_Float32.mseed"):
                ('float32', 'f', 4, def_content.astype('float32')),
            os.path.join(path, "float64_Float64.mseed"):
                ('float64', 'd', 5, def_content.astype('float64'))
        }
        # Loop over all files and read them.
        for file in files.keys():
            # Check little and big Endian for each file.
            for _i in ('littleEndian', 'bigEndian'):
                cur_file = file[:-6] + '_' + _i + '.mseed'
                st = read(os.path.join(cur_file))
                # Check the array.
                np.testing.assert_array_equal(st[0].data, files[file][3])
                # Check the dtype.
                self.assertEqual(st[0].data.dtype, files[file][0])
                # Check byteorder. Should always be native byteorder. Byteorder
                # does not apply to ASCII arrays.
                if 'ASCII' in cur_file:
                    self.assertEqual(st[0].data.dtype.byteorder, '|')
                else:
                    self.assertEqual(st[0].data.dtype.byteorder, '=')
                del st
                # Read just the first record to check encoding. The sampletype
                # should follow from the encoding. But libmseed seems not to
                # read the sampletype when reading a file.
                ms = _MSStruct(cur_file, init_msrmsf=False)
                ms.read(-1, 0, 1, 0)
                # Check values.
                self.assertEqual(getattr(ms.msr.contents, 'encoding'),
                                 files[file][2])
                if _i == 'littleEndian':
                    self.assertEqual(getattr(ms.msr.contents, 'byteorder'), 0)
                else:
                    self.assertEqual(getattr(ms.msr.contents, 'byteorder'), 1)
                # Deallocate for debugging with valrgind
                del ms

    def test_writingMicroseconds(self):
        """
        Microseconds should be written.
        """
        file = os.path.join(self.path, 'data',
                            'BW.UH3.__.EHZ.D.2010.171.first_record')
        st = read(file)
        # Read and write the record again with different microsecond times
        for ms in [111111, 111110, 100000, 111100, 111000, 11111, 11110, 10000,
                   1111, 1110, 1000, 111, 110, 100, 11, 10, 1, 0,
                   999999, 999990, 900000, 999900, 999000, 99999, 99990, 90000,
                   9999, 9990, 999, 990, 99, 90, 9, 0, 100001, 900009]:
            st[0].stats.starttime = UTCDateTime(2010, 8, 7, 0, 8, 52, ms)
            tempfile = NamedTemporaryFile().name
            st.write(tempfile, format='MSEED', reclen=512)
            st2 = read(tempfile)
            os.remove(tempfile)
            # Should also be true for the stream objects.
            self.assertEqual(st[0].stats.starttime, st2[0].stats.starttime)
            # Should also be true for the stream objects.
            self.assertEqual(st[0].stats, st2[0].stats)

    def test_readingAndWritingDataquality(self):
        """
        Tests if the dataquality is written and read correctly. There is no
        corresponding test in test_libmseed.py as it is just more convenient to
        write it in here.
        """
        tempfile = NamedTemporaryFile().name
        np.random.seed(800)  # make test reproducable
        data = np.random.randint(-1000, 1000, 50).astype('int32')
        # Create 4 different traces with 4 different dataqualities.
        stats1 = {'network': 'BW', 'station': 'TEST', 'location': 'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed': {'dataquality': 'D'}}
        stats1['starttime'] = UTCDateTime(2000, 1, 1)
        stats2 = copy.deepcopy(stats1)
        stats2['mseed']['dataquality'] = 'R'
        stats2['location'] = 'B'
        stats3 = copy.deepcopy(stats1)
        stats3['mseed']['dataquality'] = 'Q'
        stats3['location'] = 'C'
        stats4 = copy.deepcopy(stats1)
        stats4['mseed']['dataquality'] = 'M'
        stats4['location'] = 'D'
        # Create the traces.
        tr1 = Trace(data=data, header=stats1)
        tr2 = Trace(data=data, header=stats2)
        tr3 = Trace(data=data, header=stats3)
        tr4 = Trace(data=data, header=stats4)
        st = Stream([tr1, tr2, tr3, tr4])
        # Write it.
        st.write(tempfile, format="MSEED")
        # Read it again and delete the temporary file.
        stream = read(tempfile)
        os.remove(tempfile)
        stream.verify()
        # Check all four dataqualities.
        for tr_old, tr_new in zip(st, stream):
            self.assertEqual(tr_old.stats.mseed.dataquality,
                             tr_new.stats.mseed.dataquality)

    def test_writingInvalidDataQuality(self):
        """
        Trying to write an invalid dataquality results in an error. Only D, R,
        Q and M are allowed.
        """
        tempfile = NamedTemporaryFile().name
        data = np.zeros(10)
        # Create 4 different traces with 4 different dataqualities.
        stats1 = {'network': 'BW', 'station': 'TEST', 'location': 'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed': {'dataquality': 'X'}}
        st = Stream([Trace(data=data, header=stats1)])
        # Write it.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED")
        # Delete the file if it has been written, i.e. the test failed.
        try:
            os.remove(tempfile)
        except:
            pass

    def test_isInvalidMSEED(self):
        """
        Tests isMSEED functionality.
        """
        # invalid blockette length in first blockette
        file = os.path.join(self.path, 'data', 'not.mseed')
        self.assertFalse(isMSEED(file))
        # just "000001V"
        file = os.path.join(self.path, 'data', 'not2.mseed')
        self.assertFalse(isMSEED(file))
        # just "000001V011"
        file = os.path.join(self.path, 'data', 'not3.mseed')
        self.assertFalse(isMSEED(file))
        # found blockette 010 but invalid record length
        file = os.path.join(self.path, 'data', 'not4.mseed')
        self.assertFalse(isMSEED(file))

    def test_isValidMSEED(self):
        """
        Tests isMSEED functionality.
        """
        # fullseed starting with blockette 010
        file = os.path.join(self.path, 'data', 'fullseed.mseed')
        self.assertTrue(isMSEED(file))
        # fullseed starting with blockette 008
        file = os.path.join(self.path, 'data', 'blockette008.mseed')
        self.assertTrue(isMSEED(file))
        # fullseed not starting with blockette 010 or 008
        file = os.path.join(self.path, 'data', 'fullseed.mseed')
        self.assertTrue(isMSEED(file))

    def test_bizarreFiles(self):
        """
        Tests reading some bizarre MSEED files.
        """
        st1 = read(os.path.join(self.path, "data", "bizarre",
                                "endiantest.be-header.be-data.mseed"))
        st2 = read(os.path.join(self.path, "data", "bizarre",
                                "endiantest.be-header.le-data.mseed"))
        st3 = read(os.path.join(self.path, "data", "bizarre",
                                "endiantest.le-header.be-data.mseed"))
        st4 = read(os.path.join(self.path, "data", "bizarre",
                                "endiantest.le-header.le-data.mseed"))
        for st in [st1, st2, st3, st4]:
            self.assertEqual(len(st), 1)
            self.assertEqual(st[0].id, "NL.HGN.00.BHZ")
            self.assertEqual(st[0].stats.starttime,
                             UTCDateTime("2003-05-29T02:13:22.043400Z"))
            self.assertEqual(st[0].stats.endtime,
                             UTCDateTime("2003-05-29T02:18:20.693400Z"))
            self.assertEqual(st[0].stats.npts, 11947)
            self.assertEqual(list(st[0].data[0:3]), [2787, 2776, 2774])

    def test_writeAndReadDifferentEncodings(self):
        """
        Writes and read a file with different encoding via the obspy.core
        methods.
        """
        npts = 1000
        np.random.seed(815)  # make test reproducable
        data = np.random.randn(npts).astype('float64') * 1e3 + .5
        st = Stream([Trace(data=data)])
        # Loop over some record lengths.
        for encoding, value in ENCODINGS.iteritems():
            seed_dtype = value[2]
            tempfile = NamedTemporaryFile().name
            # Write it once with the encoding key and once with the value.
            st[0].data = data.astype(seed_dtype)
            st.verify()
            st.write(tempfile, format="MSEED", encoding=encoding)
            st2 = read(tempfile)
            del st2[0].stats.mseed
            np.testing.assert_array_equal(st[0].data, st2[0].data)
            del st2
            ms = _MSStruct(tempfile)
            ms.read(-1, 1, 1, 0)
            self.assertEqual(ms.msr.contents.encoding, encoding)
            del ms  # for valgrind
            os.remove(tempfile)


def suite():
    return unittest.makeSuite(MSEEDReadingAndWritingTestCase,  'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
