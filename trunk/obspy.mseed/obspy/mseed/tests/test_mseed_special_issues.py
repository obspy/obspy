# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, Stream, Trace, read
from obspy.core.util import NamedTemporaryFile
from obspy.mseed import util
from obspy.mseed.headers import clibmseed, PyFile_FromFile
from obspy.mseed.core import readMSEED, writeMSEED
from obspy.mseed.msstruct import _MSStruct
from struct import unpack
import ctypes as C
import numpy as np
import os
import platform
import sys
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


class MSEEDSpecialIssueTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        # mseed steim compression is big endian
        if sys.byteorder == 'little':
            self.swap = 1
        else:
            self.swap = 0

    def test_invalidRecordLength(self):
        """
        An invalid record length should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815)  # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        # Writing should fail with invalid record lengths.
        # Not a power of 2.
        self.assertRaises(ValueError, writeMSEED, st, tempfile, format="MSEED",
                          reclen=1000)
        # Too small.
        self.assertRaises(ValueError, writeMSEED, st, tempfile, format="MSEED",
                          reclen=8)
        # Not a number.
        self.assertRaises(ValueError, writeMSEED, st, tempfile, format="MSEED",
                          reclen='A')
        os.remove(tempfile)

    def test_invalidEncoding(self):
        """
        An invalid encoding should raise an exception.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815)  # make test reproducable
        data = np.random.randint(-1000, 1000, npts).astype('int32')
        st = Stream([Trace(data=data)])
        # Writing should fail with invalid record lengths.
        # Wrong number.
        self.assertRaises(ValueError, writeMSEED, st, tempfile, format="MSEED",
                          encoding=2)
        # Wrong Text.
        self.assertRaises(ValueError, writeMSEED, st, tempfile, format="MSEED",
                          encoding='FLOAT_64')
        os.remove(tempfile)

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
        args.append(1)  # 10 argument function
        self.assertRaises(ArgumentError, cl.mst_packgroup, *args)
        args = ['hallo']  # one argument functions
        self.assertRaises(ArgumentError, cl.msr_starttime, *args)
        self.assertRaises(ArgumentError, cl.msr_endtime, *args)
        self.assertRaises(ArgumentError, cl.mst_init, *args)
        self.assertRaises(ArgumentError, cl.mst_free, *args)
        self.assertRaises(ArgumentError, cl.mst_initgroup, *args)
        self.assertRaises(ArgumentError, cl.mst_freegroup, *args)
        self.assertRaises(ArgumentError, cl.msr_init, *args)

    def test_brokenLastRecord(self):
        """
        Test if Libmseed is able to read files with broken last record. Use
        both methods, readMSTracesViaRecords and readMSTraces
        """
        file = os.path.join(self.path, "data", "brokenlastrecord.mseed")
        # independent reading of the data
        data_string = open(file, 'rb').read()[128:]  # 128 Bytes header
        data = util._unpackSteim2(data_string, 5980, swapflag=self.swap,
                                  verbose=0)
        # test readMSTraces
        data_record = readMSEED(file)[0].data
        np.testing.assert_array_equal(data, data_record)

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
        writeMSEED(st, tempfile, format="MSEED")
        # read it again
        new_stream = readMSEED(tempfile)
        self.assertEquals(len(new_stream), 2)
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
        writeMSEED(st, tempfile, format="MSEED")
        # read temp file directly without libmseed
        bin_data = open(tempfile, "rb").read()
        bin_data = np.array(unpack(">7f", bin_data[56:84]))
        np.testing.assert_array_equal(data, bin_data)
        # read via ObsPy
        st2 = readMSEED(tempfile)
        os.remove(tempfile)
        # test results
        np.testing.assert_array_equal(data, st2[0].data)

    @skipIf(NO_NEGATIVE_TIMESTAMPS,
            'times before 1970 are not supported on this operation system')
    def test_writeWithDateTimeBefore1970(self):
        """
        Write an stream via libmseed with a datetime before 1970.

        This test depends on the platform specific localtime()/gmtime()
        function.
        """
        # XXX: skip Windows systems
        if platform.system() == 'Windows':
            return
        # create trace
        tr = Trace(data=np.empty(1000))
        tr.stats.starttime = UTCDateTime("1969-01-01T00:00:00")
        # write file
        tempfile = NamedTemporaryFile().name
        writeMSEED(Stream([tr]), tempfile, format="MSEED")
        # read again
        stream = readMSEED(tempfile)
        os.remove(tempfile)
        stream.verify()

    def test_invalidDataType(self):
        """
        Writing data of type int64 and int16 are not supported.
        """
        npts = 6000
        tempfile = NamedTemporaryFile().name
        np.random.seed(815)  # make test reproducable
        # int64
        data = np.random.randint(-1000, 1000, npts).astype('int64')
        st = Stream([Trace(data=data)])
        self.assertRaises(Exception, st.write, tempfile, format="MSEED")
        # int8
        data = np.random.randint(-1000, 1000, npts).astype('int8')
        st = Stream([Trace(data=data)])
        self.assertRaises(Exception, st.write, tempfile, format="MSEED")
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
        encoding set in stats.mseed.encoding.
        This will just raise a warning.
        """
        file = os.path.join(self.path, "data", \
                            "BW.BGLD.__.EHE.D.2008.001.first_record")
        tempfile = NamedTemporaryFile().name
        # Read the data and convert them to float
        st = read(file)
        st[0].data = st[0].data.astype('float32') + .5
        # Type is not consistent float32 cannot be compressed with STEIM1,
        # therefore a warning should be raised.
        self.assertEqual(st[0].stats.mseed.encoding, 'STEIM1')
        warnings.simplefilter('error', UserWarning)
        self.assertRaises(UserWarning, st.write, tempfile, format="MSEED")
        warnings.filters.pop()
        os.remove(tempfile)

    def test_wrongRecordLengthAsArgument(self):
        """
        Specifying a wrong record length should raise an error.
        """
        file = os.path.join(self.path, 'data', 'libmseed',
                            'float32_Float32_bigEndian.mseed')
        self.assertRaises(Exception, read, file, reclen=4096)

    def test_readQualityInformationWarns(self):
        """
        Reading the quality information while reading the data files is no more
        supported in newer obspy.mseed versions. Check that a warning is
        raised.
        Similar functionality is included in obspy.mseed.util.
        """
        timingqual = os.path.join(self.path, 'data', 'timingquality.mseed')
        warnings.simplefilter('error', DeprecationWarning)
        # This should not raise a warning.
        read(timingqual)
        # This should warn.
        self.assertRaises(DeprecationWarning, read, timingqual, quality=True)
        warnings.filters.pop()

    def test_issue160(self):
        """
        Tests issue #160.

        Reading the head of old fullseed file. Only the first 1024 byte of the
        original file are provided.
        """
        file = os.path.join(self.path, 'data',
                            'RJOB.BW.EHZ.D.300806.0000.fullseed')
        tr_one = read(file)[0]
        tr_two = read(file, headonly=True)[0]
        ms = "AttribDict({'dataquality': 'D', 'record_length': 512, " + \
             "'byteorder': '>', 'encoding': 'STEIM1'})"
        for tr in tr_one, tr_two:
            self.assertEqual('BW.RJOB..EHZ', tr.id)
            self.assertEqual(ms, repr(tr.stats.mseed))
            self.assertEqual(412, tr.stats.npts)
            self.assertEqual(UTCDateTime(2006, 8, 30, 0, 0, 2, 815000),
                             tr.stats.endtime)

    def test_issue217(self):
        """
        Tests issue #217.

        Reading a MiniSEED file without sequence numbers and a record length of
        1024.
        """
        file = os.path.join(self.path, 'data',
                            'reclen_1024_without_sequence_numbers.mseed')
        tr = read(file)[0]
        ms = "AttribDict({'dataquality': 'D', 'record_length': 1024, " + \
             "'byteorder': '>', 'encoding': 'STEIM1'})"
        self.assertEqual('XX.STF1..HHN', tr.id)
        self.assertEqual(ms, repr(tr.stats.mseed))
        self.assertEqual(932, tr.stats.npts)
        self.assertEqual(UTCDateTime(2007, 5, 31, 22, 45, 46, 720000),
                         tr.stats.endtime)

    def test_issue296(self):
        """
        Tests issue #296.
        """
        tempfile = NamedTemporaryFile().name
        # 1 - transform to np.float64 values
        st = read()
        for tr in st:
            tr.data = tr.data.astype('float64')
        # write a single trace automatically detecting encoding
        st[0].write(tempfile, format="MSEED")
        # write a single trace automatically detecting encoding
        st.write(tempfile, format="MSEED")
        # write a single trace with encoding 5
        st[0].write(tempfile, format="MSEED", encoding=5)
        # write a single trace with encoding 5
        st.write(tempfile, format="MSEED", encoding=5)
        # 2 - transform to np.float32 values
        st = read()
        for tr in st:
            tr.data = tr.data.astype('float32')
        # write a single trace automatically detecting encoding
        st[0].write(tempfile, format="MSEED")
        # write a single trace automatically detecting encoding
        st.write(tempfile, format="MSEED")
        # write a single trace with encoding 4
        st[0].write(tempfile, format="MSEED", encoding=4)
        # write a single trace with encoding 4
        st.write(tempfile, format="MSEED", encoding=4)
        # 3 - transform to np.int32 values
        st = read()
        for tr in st:
            tr.data = tr.data.astype('int32')
        # write a single trace automatically detecting encoding
        st[0].write(tempfile, format="MSEED")
        # write a single trace automatically detecting encoding
        st.write(tempfile, format="MSEED")
        # write a single trace with encoding 3
        st[0].write(tempfile, format="MSEED", encoding=3)
        # write the whole stream with encoding 3
        st.write(tempfile, format="MSEED", encoding=3)
        # write a single trace with encoding 10
        st[0].write(tempfile, format="MSEED", encoding=10)
        # write the whole stream with encoding 10
        st.write(tempfile, format="MSEED", encoding=10)
        # write a single trace with encoding 11
        st[0].write(tempfile, format="MSEED", encoding=11)
        # write the whole stream with encoding 11
        st.write(tempfile, format="MSEED", encoding=11)
        # 4 - transform to np.int16 values
        st = read()
        for tr in st:
            tr.data = tr.data.astype('int16')
        # write a single trace automatically detecting encoding
        st[0].write(tempfile, format="MSEED")
        # write a single trace automatically detecting encoding
        st.write(tempfile, format="MSEED")
        # write a single trace with encoding 1
        st[0].write(tempfile, format="MSEED", encoding=1)
        # write the whole stream with encoding 1
        st.write(tempfile, format="MSEED", encoding=1)
        # 5 - transform to ASCII values
        st = read()
        for tr in st:
            tr.data = tr.data.astype('|S1')
        # write a single trace automatically detecting encoding
        st[0].write(tempfile, format="MSEED")
        # write a single trace automatically detecting encoding
        st.write(tempfile, format="MSEED")
        # write a single trace with encoding 0
        st[0].write(tempfile, format="MSEED", encoding=0)
        # write the whole stream with encoding 0
        st.write(tempfile, format="MSEED", encoding=0)
        # cleanup
        os.remove(tempfile)

    def test_issue289(self):
        """
        Tests issue #289.

        Reading MiniSEED using start-/endtime outside of data should result in
        an empty Stream object.
        """
        # 1
        file = os.path.join(self.path, 'data', 'steim2.mseed')
        st = read(file, starttime=UTCDateTime() - 10, endtime=UTCDateTime())
        self.assertEqual(len(st), 0)
        # 2
        file = os.path.join(self.path, 'data', 'fullseed.mseed')
        st = read(file, starttime=UTCDateTime() - 10, endtime=UTCDateTime())
        self.assertEqual(len(st), 0)

    def test_issue312(self):
        """
        Tests issue #312

        The blkt_link struct was defined wrong.
        """
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        # start and endtime
        ms = _MSStruct(filename)
        ms.read(-1, 0, 1, 0)
        blkt_link = ms.msr.contents.blkts.contents
        # The first blockette usually begins after 48 bytes. In the test file
        # it does.
        self.assertEqual(blkt_link.blktoffset, 48)
        # The first blockette is blockette 1000 in this file.
        self.assertEqual(blkt_link.blkt_type, 1000)
        # Only one blockette.
        self.assertEqual(blkt_link.next_blkt, 0)
        # Blockette data is 8 bytes - 4 bytes for the blockette header.
        self.assertEqual(blkt_link.blktdatalen, 4)
        del ms


def suite():
    return unittest.makeSuite(MSEEDSpecialIssueTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
