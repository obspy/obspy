# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.core.util import NamedTemporaryFile
from obspy.mseed import libmseed
from obspy.mseed.libmseed import MSStruct
from obspy.mseed.core import readMSEED
import inspect
import numpy as np
import os
import unittest


class CoreTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(inspect.getsourcefile(self.__class__))

    def tearDown(self):
        pass

    def test_readFileViaLibMSEED(self):
        """
        Read file test via L{obspy.mseed.core.readMSEED}.
        """
        testfile = os.path.join(self.path, 'data', 'test.mseed')
        testdata = [2787, 2776, 2774, 2780, 2783]
        # read
        stream = readMSEED(testfile)
        stream._verify()
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
        stream._verify()
        self.assertEqual(stream[0].stats.network, 'NL')
        self.assertEqual(stream[0].stats['station'], 'HGN')
        self.assertEqual(stream[0].stats.npts, 11947)
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])
        # with given format
        stream = read(testfile, format='MSEED')
        stream._verify()
        self.assertEqual(stream[0].stats.get('location'), '00')
        self.assertEqual(stream[0].stats.get('channel'), 'BHZ')
        self.assertEqual(stream[0].stats['sampling_rate'], 40.0)
        for _i in xrange(5):
            self.assertEqual(stream[0].data[_i], testdata[_i])
        # file with gaps
        gapfile = os.path.join(self.path, 'data', 'gaps.mseed')
        # without given format -> autodetect using extension
        stream = read(gapfile)
        stream._verify()
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
        self.assertEqual(4, len(stream.traces))
        for _i in stream.traces:
            self.assertEqual(True, isinstance(_i, Trace))
            self.assertEqual(str(_i.data), '[]')

    def test_writeIntegersViaObsPy(self):
        """
        Write integer array via L{obspy.core.Stream}.
        """
        tempfile = NamedTemporaryFile().name
        npts = 1000
        # data array of integers - float won't work!
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'',
                 'channel': 'EHE', 'npts': npts, 'sampling_rate': 200.0}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
        # write
        st.write(tempfile, format="MSEED")
        # read again
        stream = read(tempfile)
        os.remove(tempfile)
        stream._verify()
        self.assertEquals(stream[0].data.tolist(), data.tolist())

    def test_readWithWildCard(self):
        """
        Reads wildcard filenames.
        """
        files = os.path.join(self.path, 'data', 'BW.BGLD.__.EHE.D.2008.001.*')
        st = read(files)
        self.assertEquals(len(st), 4)
        st.merge()
        self.assertEquals(len(st), 1)

    def test_readFullSEED(self):
        """
        Reads a full SEED volume.
        """
        files = os.path.join(self.path, 'data', 'ArclinkRequest_340397.fseed')
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
                 'mseed' : {'dataquality' : 'D'}}
        stats['starttime'] = UTCDateTime(2000, 1, 1)
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
        # Write it.
        st.write(tempfile, format="MSEED")
        # Read it again and delete the temporary file.
        stream = read(tempfile)
        os.remove(tempfile)
        stream._verify()
        # Loop over the attributes to be able to assert them because a
        # dictionary is not a stats dictionary.
        # This also assures that there are no additional keys.
        for key in stats.keys():
            self.assertEqual(stats[key], stream[0].stats[key])
        # Test the dataquality key extra.
        self.assertEqual(stream[0].stats.mseed.dataquality, 'D')

    def test_writeAndReadDifferentRecordLengths(self):
        """
        Tests Mini-SEED writing and record lengths.
        """
        # libmseed instance.
        mseed = libmseed()
        npts = 6000
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
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

    def test_invalidRecordLength(self):
        """
        An invalid record length should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
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

    def test_writeAndReadDifferentEncodings(self):
        """
        Writes and read a file with different encoding via the obspy.core
        methods.
        """
        # libmseed instance.
        npts = 1000
        np.random.seed(815) # make test reproducable
        data = np.random.randn(npts).astype('float64') * 1e3 + .5
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        tr = Trace(data=data, header=stats)
        st = Stream([tr])
        encodings = {'ASCII': (0, "a"), 'INT32': (3, "i"),
                     'FLOAT32': (4, "f"), 'FLOAT64': (5, "d"),
                     'STEIM1': (10, "i"), 'STEIM2' : (11, "i")}
        # Loop over some record lengths.
        for key in encodings.keys():
            tempfile = NamedTemporaryFile().name
            tempfile2 = NamedTemporaryFile().name
            # Write it once with the encoding key and once with the value.
            st[0].data = data.astype(encodings[key][1])
            st._verify()
            st.write(tempfile, format="MSEED", encoding=key)
            st.write(tempfile2, format="MSEED", encoding=key)
            # Test reading the two files.
            temp_st1 = read(tempfile)
            temp_st2 = read(tempfile2)
            np.testing.assert_array_equal(st[0].data, temp_st1[0].data)
            np.testing.assert_array_equal(st[0].data, temp_st2[0].data)
            del temp_st1#, temp_st2
            # Check the encodings.
            for file in [tempfile, tempfile2]:
                ms = MSStruct(file)
                ms.read(-1, 1, 1, 0)
                self.assertEqual(ms.msr.contents.encoding, encodings[key][0])
                del ms # for valgrind
                os.remove(file)

    def test_invalidEncoding(self):
        """
        An invalid encoding should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815) # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
        # Writing should fail with invalid record lengths.
        # Wrong number.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          encoding=2)
        # Wrong Text.
        self.assertRaises(ValueError, st.write, tempfile, format="MSEED",
                          encoding='FLOAT_64')

    def test_readPartsOfFile(self):
        """
        Test reading only parts of an mseed file without unpacking or
        touching the rest.
        """
        temp = os.path.join(self.path, 'data', 'BW.BGLD.__.EHE.D.2008.001')
        file = temp + '.first_10_percent'
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
        called for an unkown file format.
        """
        file = os.path.join(self.path, 'data', 'BW.BGLD.__.EHE.D.2008.001'
        '.second_record')
        tr = read(file, verify_chksum=True, starttime=None)[0]
        data = np.array([-397, -387, -368, -381, -388])
        np.testing.assert_array_equal(tr.data[0:5], data)
        self.assertEqual(412, len(tr.data))
        data = np.array([-406, -417, -403, -423, -413])
        np.testing.assert_array_equal(tr.data[-5:], data)

    def test_writeAllTypes(self):
        """
        Tests writing all different types. This is an test which is independent of
        the read method. Only the data part is verified.
        """
        file = os.path.join(self.path, "data", \
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        tempfile = NamedTemporaryFile().name
        # Read the data and copy them
        st = read(file)
        data_copy = st[0].data.copy()
        # Float64, Float32, Int32, Int24, Int16, Char
        encodings = {5: "<f8", 4: "<f4", 3: "<i4", 0: "|S1"} #, 1: "<i2"
        for encoding, dtype in encodings.iteritems():
            # Convert data to floats and write them again
            st[0].data = data_copy.astype(dtype)
            st.write(tempfile, format="MSEED", encoding=encoding,
                     reclen=256, byteorder=0)
            # Read the binary chunk of data without header not using ObsPy
            s = open(tempfile, "rb").read()
            data = np.fromstring(s[56:256], dtype=dtype)
            # Test if both are the same
            np.testing.assert_array_equal(data, st[0].data[:len(data)])


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
