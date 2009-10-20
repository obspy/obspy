# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.mseed.core import readMSEED
from obspy.core.util import NamedTemporaryFile
from obspy.mseed import libmseed
import inspect
import numpy as N
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
        data = N.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'',
                 'channel': 'EHE', 'npts': npts, 'sampling_rate': 200.0}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
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
        data = N.random.randint(-1000, 1000, 50).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (50 - 1) * 0.005
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
        data = N.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
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
            N.testing.assert_array_equal(data, temp_st[0].data)
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
        data = N.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
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
        mseed = libmseed()
        npts = 1000
        data = N.random.randint(10, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
        tr = Trace(data=data, header=stats)
        tr._verify()
        st = Stream([tr])
        st._verify()
        encodings = {'INT16' : 1, 'INT32' : 3, 'STEIM1' : 10, 'STEIM2' :11}
        # Loop over some record lengths.
        for key in encodings.keys():
            tempfile = NamedTemporaryFile().name
            tempfile2 = NamedTemporaryFile().name
            # Write it once with the encoding key and once with the value.
            st.write(tempfile, format="MSEED", encoding=encodings[key])
            st.write(tempfile2, format="MSEED", encoding=key)
            # Check the encodings.
            msr1, msf1 = mseed.readSingleRecordToMSR(tempfile)
            msr2, msf2 = mseed.readSingleRecordToMSR(tempfile2)
            # Test reading the two files.
            temp_st1 = read(tempfile)
            temp_st2 = read(tempfile2)
            N.testing.assert_array_equal(data, temp_st1[0].data)
            N.testing.assert_array_equal(data, temp_st2[0].data)
            del temp_st1
            del temp_st2
            # Assert encodings.
            self.assertEqual(msr1.contents.encoding, encodings[key])
            self.assertEqual(msr2.contents.encoding, encodings[key])
            # Close file handler
            mseed.clear(msf1, msr1)
            mseed.clear(msf2, msr2)
            # Delete temp files.
            os.remove(tempfile)
            os.remove(tempfile2)

    def test_invalidEncoding(self):
        """
        An invalid encoding should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        data = N.random.randint(-1000, 1000, npts).astype('int32')
        stats = {'network': 'BW', 'station': 'TEST', 'location':'A',
                 'channel': 'EHE', 'npts': len(data), 'sampling_rate': 200.0,
                 'mseed' : {'dataquality' : 'D'}}
        start = UTCDateTime(2000, 1, 1)
        stats['starttime'] = start
        stats['endtime'] = start + (npts - 1) * 0.005
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


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
