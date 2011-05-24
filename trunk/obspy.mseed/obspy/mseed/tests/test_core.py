# -*- coding: utf-8 -*-

from copy import deepcopy
from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.core.util import AttribDict, NamedTemporaryFile
from obspy.mseed import LibMSEED
from obspy.mseed.core import readMSEED, isMSEED
from obspy.mseed.headers import ENCODINGS
from obspy.mseed.libmseed import _MSStruct
import numpy as np
import os
import unittest
import warnings
try:
    from unittest import skipIf
except ImportError:
    from obspy.core.util import skipIf


# some Python version don't support negative timestamps
NO_NEGATIVE_TIMESTAMPS = False
try:
    UTCDateTime(-50000)
except:
    NO_NEGATIVE_TIMESTAMPS = True


class CoreTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_readFileViaLibMSEED(self):
        """
        Read file test via L{obspy.mseed.core.readMSEED}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        testdata = [2787, 2776, 2774, 2780, 2783]
        # read
        stream = readMSEED(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(stream[0].stats.get('location'), '00')
        self.assertEqual(stream[0].stats.npts, 11947)
        self.assertEqual(stream[0].stats['sampling_rate'], 40.0)
        self.assertEqual(stream[0].stats.get('channel'), 'BHZ')
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])

    def test_readFileViaObsPy(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        testdata = [2787, 2776, 2774, 2780, 2783]
        # without given format -> auto detect file format
        stream = read(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(stream[0].stats.npts, 11947)
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])
        # with given format
        stream = read(testfile, format='MSEED')
        stream.verify()
        self.assertEqual(stream[0].stats.get('location'), '00')
        self.assertEqual(stream[0].stats.get('channel'), 'BHZ')
        self.assertEqual(stream[0].stats['sampling_rate'], 40.0)
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])
        # file with gaps
        gapfile = os.path.join(self.path, 'data', 'gaps.mseed')
        # without given format -> autodetect using extension
        stream = read(gapfile)
        stream.verify()
        self.assertEqual(4, len(stream.traces))
        for _i in stream.traces:
            self.assertEqual(True, isinstance(_i, Trace))

    def test_readHeadFileViaObsPy(self):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        stream = read(testfile, headonly=True, format='MSEED')
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(str(stream[0].data), '[]')
        self.assertEqual(stream[0].stats.npts, 11947)
        #
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

    def test_writeIntegersViaObsPy(self):
        """
        Write integer array via L{obspy.core.Stream}.
        """
        tempfile = NamedTemporaryFile().name
        npts = 1000
        # data array of integers - float won't work!
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        # write
        st.write(tempfile, format="MSEED")
        # read again
        stream = read(tempfile)
        os.remove(tempfile)
        stream.verify()
        self.assertEquals(stream[0].data.tolist(), data.tolist())

    @skipIf(NO_NEGATIVE_TIMESTAMPS, 'times before 1970 are not supported')
    def test_writeWithDateTimeBefore1970(self):
        """
        Write an stream via libmseed with a datetime before 1970.
        
        This test depends on the platform specific localtime()/gmtime()
        function. 
        """
        # create trace
        tr = Trace(data=np.empty(1000))
        tr.stats.starttime = UTCDateTime("1969-01-01T00:00:00")
        # write file
        tempfile = NamedTemporaryFile().name
        tr.write(tempfile, format="MSEED")
        # read again
        stream = read(tempfile)
        os.remove(tempfile)
        stream.verify()

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

    def test_readFullSEED(self):
        """
        Reads a full SEED volume.
        """
        files = os.path.join(self.path, 'data', 'fullseed.mseed')
        st = read(files)
        self.assertEquals(len(st), 3)
        self.assertEquals(len(st[0]), 602)
        self.assertEquals(len(st[1]), 623)
        self.assertEquals(len(st[2]), 610)

    def test_Header(self):
        """
        Tests whether the header is correctly written and read.
        """
        tempfile = NamedTemporaryFile().name
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, 50).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D', 'record_length' : 512,
                            'encoding' : 'STEIM2', 'byteorder' : '>'}}
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

    def test_writeAndReadDifferentRecordLengths(self):
        """
        Tests Mini-SEED writing and record lengths.
        """
        # libmseed instance.
        mseed = LibMSEED()
        npts = 6000
        np.random.seed(815) # make test reproducable
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
            info = mseed._getMSFileInfo(file, tempfile)
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

    def test_readingAndWritingViaTheStatsAttribute(self):
        """
        Tests the writing with MSEED file attributes set via the attributes in
        trace.stats.mseed.
        """
        __libmseed__ = LibMSEED()
        npts = 6000
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        # Test all possible combinations of record length, encoding and
        # byteorder.
        record_lengths = [256, 512, 1024, 2048, 4096, 8192]
        byteorders = ['>', '<']
        encodings = [value[0] for value in ENCODINGS.values()]
        np_encodings = {}
        for value in ENCODINGS.values():
            np_encodings[value[0]] = value[2]
        st = Stream([Trace(data=data)])
        st[0].stats.mseed = AttribDict()
        st[0].stats.mseed.dataquality = 'D'
        # Loop over all combinations.
        for reclen in record_lengths:
            for order in byteorders:
                for encoding in encodings:
                    # Create new stream and change stats.
                    stream = deepcopy(st)
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

    def test_invalidRecordLength(self):
        """
        An invalid record length should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        # Writing should fail with invalid record lengths.
        # Not a power of 2.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          reclen=1000)
        # Too small.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          reclen=8)
        # Not a number.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          reclen='A')
        os.remove(tempfile)

    def test_writeAndReadDifferentEncodings(self):
        """
        Writes and read a file with different encoding via the obspy.core
        methods.
        """
        # libmseed instance.
        npts = 1000
        np.random.seed(815) # make test reproducable
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
            del ms # for valgrind
            os.remove(tempfile)

    def test_invalidEncoding(self):
        """
        An invalid encoding should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        # Writing should fail with invalid record lengths.
        # Wrong number.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          encoding=2)
        # Wrong Text.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          encoding='FLOAT_64')
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
        byteorders = {0:'<', 1:'>'}
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

    def test_invalidDataType(self):
        """
        Writing data of type int64 and int16 are not supported.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815) # make test reproducable
        # int64
        data = np.random.randint(-1000, 1000, npts).astype('int64')
        st = Stream([Trace(data=data)])
        self.assertRaises(Exception, st.write, tempfile, format="MSEED")
        # int8
        data = np.random.randint(-1000, 1000, npts).astype('int8')
        st = Stream([Trace(data=data)])
        self.assertRaises(Exception, st.write, tempfile, format="MSEED")
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

    def test_writeWrongEncoding(self):
        """
        Test to write a floating point mseed file with encoding STEIM1.
        An exception should be raised.
        """
        file = os.path.join(self.path, "data", \
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        tempfile = NamedTemporaryFile().name
        # Read the data and convert them to float
        st = read(file)
        st[0].data = st[0].data.astype('float32') + .5
        # Type is not consistent float32 cannot be compressed with STEIM1,
        # therefore a exception should be raised.
        self.assertRaises(Exception, st.write, tempfile, format="MSEED",
                encoding=10)
        os.remove(tempfile)

    def test_writeWrongEncodingViaMseedStats(self):
        """
        Test to write a floating point mseed file with encoding STEIM1 with the
        encoding set in stats.mseed.encoding.  This will just raise a warning.
        
        Tests issue 256.
        """
        file = os.path.join(self.path, "data", "steim2.mseed")
        tempfile = NamedTemporaryFile().name
        # Read the data and convert them to float
        st = read(file)
        st[0].data = st[0].data.astype('float64') + .5
        # Type is not consistent float64 cannot be compressed with STEIM2,
        # therefore a warning should be raised.
        self.assertEqual(st[0].stats.mseed.encoding, 'STEIM2')
        # Test the warning.
        warnings.simplefilter('error', UserWarning)
        # The execution will actually stop once the warning has reached because
        # it now is an exception.
        self.assertRaises(UserWarning, st.write, tempfile, format="MSEED")
        warnings.filters.pop(0)
        # Actually write the file.
        warnings.simplefilter('ignore', UserWarning)
        st.write(tempfile, format="MSEED")
        warnings.filters.pop(0)
        # Read again and make some sanity checks.
        st2 = read(tempfile)
        os.remove(tempfile)
        self.assertEqual(st2[0].stats.mseed.encoding, 'FLOAT64')
        np.testing.assert_array_equal(st[0].data, st2[0].data)

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
            os.path.join(path, "smallASCII.mseed") : ('|S1', 'a', 0,
                        np.fromstring('ABCDEFGH', dtype='|S1')),
            # Tests all ASCII letters.
            os.path.join(path, "fullASCII.mseed") : ('|S1', 'a', 0,
               np.fromstring(
               """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV""" + \
               """WXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~""", dtype='|S1')),
            # Note: int16 array will also be returned as int32.
            os.path.join(path, "int16_INT16.mseed") : ('int32', 'i', 1,
                                                    def_content.astype('int16')),
            os.path.join(path, "int32_INT32.mseed") : ('int32', 'i', 3,
                                                    def_content),
            os.path.join(path, "int32_Steim1.mseed") : ('int32', 'i', 10,
                                                    def_content),
            os.path.join(path, "int32_Steim2.mseed") : ('int32', 'i', 11,
                                                    def_content),
            os.path.join(path, "float32_Float32.mseed") : ('float32', 'f', 4,
                                                def_content.astype('float32')),
            os.path.join(path, "float64_Float64.mseed") : ('float64', 'd', 5,
                                                def_content.astype('float64'))
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

    def test_wrongRecordLengthAsArgument(self):
        """
        Specifying a wrong record length should raise an error.
        """
        file = os.path.join(self.path, 'data', 'libmseed',
                            'float32_Float32_bigEndian.mseed')
        self.assertRaises(Exception, read, file, reclen=4096)

    def test_readQualityInformation(self):
        """
        Tests the reading of the quality informations if the flag is set.
        """
        # Two files. One with timing quality information and one with set
        # quality flags.
        timingqual = os.path.join(self.path, 'data', 'timingquality.mseed')
        qualityflags = os.path.join(self.path, 'data', 'qualityflags.mseed')
        t_st = read(timingqual, quality=True)
        q_st = read(qualityflags, quality=True)
        # The timingquality contains values from 0 to 100 in random order.
        qual = np.arange(101, dtype='float32')
        read_qual = t_st[0].stats.mseed.timing_quality
        read_qual = sorted(read_qual)
        np.testing.assert_array_equal(qual, read_qual)
        # Check for quality flags.
        # The test file contains 18 records. The first record has no set bit,
        # bit 0 of the second record is set, bit 1 of the third, ..., bit 7 of
        # the 9th record is set. The last nine records have 0 to 8 set bits,
        # starting with 0 bits, bit 0 is set, bits 0 and 1 are set...
        # Altogether the file contains 44 set bits.
        self.assertEqual(len(q_st), 18)
        self.assertEqual(q_st[0].stats.mseed.data_quality_flags_count,
                         [0] * 8)
        for _i in xrange(8):
            dummy = [0] * 8
            dummy[_i] = 1
            self.assertEqual(q_st[_i + 1].stats.mseed.data_quality_flags_count,
                             dummy)
        dummy = [0] * 8
        self.assertEqual(q_st[9].stats.mseed.data_quality_flags_count,
                         [0] * 8)
        for _i in xrange(8):
            dummy[_i] = 1
            self.assertEqual(
                q_st[_i + 10].stats.mseed.data_quality_flags_count, dummy)

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
        np.random.seed(800) # make test reproducable
        data = np.random.randint(-1000, 1000, 50).astype('int32')
        # Create 4 different traces with 4 different dataqualities.
        stats1 = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        stats1['starttime'] = UTCDateTime(2000, 1, 1)
        stats2 = deepcopy(stats1)
        stats2['mseed']['dataquality'] = 'R'
        stats2['location'] = 'B'
        stats3 = deepcopy(stats1)
        stats3['mseed']['dataquality'] = 'Q'
        stats3['location'] = 'C'
        stats4 = deepcopy(stats1)
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
        stats1 = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'X'}}
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
        st = read(file)
        self.assertEqual(len(st), 3)
        # fullseed starting with blockette 008
        file = os.path.join(self.path, 'data', 'blockette008.mseed')
        self.assertTrue(isMSEED(file))
        st = read(file)
        self.assertEqual(len(st), 1)
        # fullseed not starting with blockette 010 or 008
        file = os.path.join(self.path, 'data', 'fullseed.mseed')
        self.assertTrue(isMSEED(file))
        st = read(file)
        self.assertEqual(len(st), 3)

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

    def test_issue160(self):
        """
        Reading head of old fullseed file. Only the first 1024byte of the
        original file are provided.
        """
        file = os.path.join(self.path, 'data',
                            'RJOB.BW.EHZ.D.300806.0000.fullseed')
        tr_one = read(file)[0]
        tr_two = read(file, headonly=True)[0]
        ms = "AttribDict({'dataquality': 'D', 'record_length': 512, " + \
             "'encoding': 'STEIM1', 'byteorder': '>'})"
        for tr in tr_one, tr_two:
            self.assertEqual('BW.RJOB..EHZ', tr.id)
            self.assertEqual(ms, repr(tr.stats.mseed))
            self.assertEqual(412, tr.stats.npts)
            self.assertEqual(UTCDateTime(2006, 8, 30, 0, 0, 2, 815000),
                             tr.stats.endtime)

    def test_issue217(self):
        """
        Reading a MiniSEED file without sequence numbers and a record length
        of 1024.
        """
        file = os.path.join(self.path, 'data',
                            'reclen_1024_without_sequence_numbers.mseed')
        tr = read(file)[0]
        ms = "AttribDict({'dataquality': 'D', 'record_length': 1024, " + \
             "'encoding': 'STEIM1', 'byteorder': '>'})"
        self.assertEqual('XX.STF1..HHN', tr.id)
        self.assertEqual(ms, repr(tr.stats.mseed))
        self.assertEqual(932, tr.stats.npts)
        self.assertEqual(UTCDateTime(2007, 5, 31, 22, 45, 46, 720000),
                         tr.stats.endtime)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
