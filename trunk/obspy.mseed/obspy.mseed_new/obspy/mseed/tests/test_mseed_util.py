# -*- coding: utf-8 -*-

from StringIO import StringIO
from obspy.core import UTCDateTime
from obspy.mseed import util
from obspy.mseed.core import readMSEED
import numpy as np
import os
import random
import sys
import unittest


class MSEEDUtilTestCase(unittest.TestCase):
    """
    Tests suite for util module of obspy.mseed.
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        # mseed steim compression is big endian
        if sys.byteorder == 'little':
            self.swap = 1
        else:
            self.swap = 0

    def test_convertDatetime(self):
        """
        Tests all time conversion methods.
        """
        # These values are created using the Linux "date -u -d @TIMESTRING"
        # command. These values are assumed to be correct.
        timesdict = {
            1234567890: UTCDateTime(2009, 2, 13, 23, 31, 30),
            1111111111: UTCDateTime(2005, 3, 18, 1, 58, 31),
            1212121212: UTCDateTime(2008, 5, 30, 4, 20, 12),
            1313131313: UTCDateTime(2011, 8, 12, 6, 41, 53),
            100000: UTCDateTime(1970, 1, 2, 3, 46, 40),
            100000.111112: UTCDateTime(1970, 1, 2, 3, 46, 40, 111112),
            200000000: UTCDateTime(1976, 5, 3, 19, 33, 20)
        }
        # Loop over timesdict.
        for ts, dt in timesdict.iteritems():
            self.assertEqual(dt, util._convertMSTimeToDatetime(ts * 1000000L))
            self.assertEqual(ts * 1000000L, util._convertDatetimeToMSTime(dt))
        # Additional sanity tests.
        # Today.
        now = UTCDateTime()
        self.assertEqual(now, util._convertMSTimeToDatetime(
                              util._convertDatetimeToMSTime(now)))
        # Some random date.
        random.seed(815)  # make test reproducable
        timestring = random.randint(0, 2000000) * 1e6
        self.assertEqual(timestring, util._convertDatetimeToMSTime(
                        util._convertMSTimeToDatetime(timestring)))

    def test_getMSFileInfo(self):
        """
        Tests the util._getMSFileInfo method with known values.
        """
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        # Simply reading the file.
        f = open(filename, 'rb')
        info = util._getMSFileInfo(f, filename)
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
        info = util._getMSFileInfo(open_file, filename)
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
        info = util._getMSFileInfo(open_file_string, filename)
        self.assertEqual(info['filesize'], 5120)
        self.assertEqual(info['record_length'], 512)
        self.assertEqual(info['number_of_records'], 10)
        self.assertEqual(info['excess_bytes'], 0)
        self.assertEqual(open_file_string.tell(), 111)
        # One more file containing two records.
        filename = os.path.join(self.path, 'data', 'test.mseed')
        f = open(filename, 'rb')
        info = util._getMSFileInfo(f, filename)
        f.close()
        self.assertEqual(info['filesize'], 8192)
        self.assertEqual(info['record_length'], 4096)
        self.assertEqual(info['number_of_records'], 2)
        self.assertEqual(info['excess_bytes'], 0)

    def test_getDataQualityFlagsCount(self):
        """
        This test reads a self-made Mini-SEED file with set Data Quality Bits.
        A real test file would be better as this test tests a file that was
        created by the inverse method that reads the bits.
        """
        filename = os.path.join(self.path, 'data', 'qualityflags.mseed')
        # Read quality flags.
        flags = util.getDataQualityFlagsCount(filename)
        # The test file contains 18 records. The first record has no set bit,
        # bit 0 of the second record is set, bit 1 of the third, ..., bit 7 of
        # the 9th record is set. The last nine records have 0 to 8 set bits,
        # starting with 0 bits, bit 0 is set, bits 0 and 1 are set...
        # Altogether the file contains 44 set bits.
        self.assertEqual(flags, [9, 8, 7, 6, 5, 4, 3, 2])
        # No set quality flags should result in a list of zeros.
        filename = os.path.join(self.path, 'data', 'test.mseed')
        flags = util.getDataQualityFlagsCount(filename)
        self.assertEqual(len(flags),  8)
        self.assertEqual(sum(flags),  0)

    def test_getStartAndEndTime(self):
        """
        Tests getting the start- and endtime of a file.

        The values are compared with the results of reading the full files.
        """
        mseed_filenames = ['BW.BGLD.__.EHE.D.2008.001.first_10_records',
                           'test.mseed', 'timingquality.mseed']
        for _i in mseed_filenames:
            filename = os.path.join(self.path, 'data', _i)
            # Get the start- and end time.
            (start, end) = util.getStartAndEndTime(filename)
            # Parse the whole file.
            stream = readMSEED(filename)
            self.assertEqual(start, stream[0].stats.starttime)
            self.assertEqual(end, stream[0].stats.endtime)

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
        filename = os.path.join(self.path, 'data', 'timingquality.mseed')
        tq = util.getTimingQuality(filename)
        self.assertEqual(tq, {'min': 0.0, 'max': 100.0, 'average': 50.0,
                              'median': 50.0, 'upper_quantile': 75.0,
                              'lower_quantile': 25.0})
        # No timing quality set should result in an empty dictionary.
        filename = os.path.join(self.path, 'data',
                                'BW.BGLD.__.EHE.D.2008.001.first_10_records')
        tq = util.getTimingQuality(filename)
        self.assertEqual(tq, {})
        tq = util.getTimingQuality(filename, first_record=False)
        self.assertEqual(tq, {})

    def test_unpackSteim1(self):
        """
        Test decompression of Steim1 strings. Remove 64 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        steim1_file = os.path.join(self.path, 'data',
                                   'BW.BGLD.__.EHE.D.2008.001.first_record')
        # 64 Bytes header.
        data_string = open(steim1_file, 'rb').read()[64:]
        data = util._unpackSteim1(data_string, 412, swapflag=self.swap,
                                   verbose=0)
        data_record = readMSEED(steim1_file)[0].data
        np.testing.assert_array_equal(data, data_record)

    def test_unpackSteim2(self):
        """
        Test decompression of Steim2 strings. Remove 128 Bytes of header
        by hand, see SEEDManual_V2.4.pdf page 100.
        """
        steim2_file = os.path.join(self.path, 'data', 'steim2.mseed')
        # 128 Bytes header.
        data_string = open(steim2_file, 'rb').read()[128:]
        data = util._unpackSteim2(data_string, 5980, swapflag=self.swap,
                                   verbose=0)
        data_record = readMSEED(steim2_file)[0].data
        np.testing.assert_array_equal(data, data_record)


def suite():
    return unittest.makeSuite(MSEEDUtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
