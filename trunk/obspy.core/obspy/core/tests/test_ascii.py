# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from obspy.core.ascii import readSLIST, readTSPAIR, isSLIST, isTSPAIR, \
    writeTSPAIR, writeSLIST
from obspy.core.util import NamedTemporaryFile
import numpy as np
import os
import unittest


class ASCIITestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_isSLISTFile(self):
        """
        Testing SLIST file format.
        """
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        self.assertEqual(isSLIST(testfile), True)
        testfile = os.path.join(self.path, 'data', 'slist_2_traces.ascii')
        self.assertEqual(isSLIST(testfile), True)
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        self.assertEqual(isSLIST(testfile), False)
        # not existing file should fail
        testfile = os.path.join(self.path, 'data', 'xyz')
        self.assertEqual(isSLIST(testfile), False)

    def test_readSLISTFileSingleTrace(self):
        """
        Read SLIST file test via obspy.core.ascii.readSLIST.
        """
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        # read
        stream = readSLIST(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_readSLISTFileMultipleTraces(self):
        """
        Read SLIST file test via obspy.core.ascii.readSLIST.
        """
        testfile = os.path.join(self.path, 'data', 'slist_2_traces.ascii')
        # read
        stream = readSLIST(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHE')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 630)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)

    def test_readSLISTFileHeadOnly(self):
        """
        Read SLIST file test via obspy.core.ascii.readSLIST.
        """
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        # read
        stream = readSLIST(testfile, headonly=True)
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(len(stream[0].data), 0)

    def test_readSLISTFileEncoding(self):
        """
        Read SLIST file test via obspy.core.ascii.readSLIST.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'slist_float.ascii')
        stream = readSLIST(testfile)
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 12)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # unknown encoding
        testfile = os.path.join(self.path, 'data', 'slist_unknown.ascii')
        self.assertRaises(NotImplementedError, readSLIST, testfile)

    def test_isTSPAIRFile(self):
        """
        Testing TSPAIR file format.
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        self.assertEqual(isTSPAIR(testfile), True)
        testfile = os.path.join(self.path, 'data', 'tspair_2_traces.ascii')
        self.assertEqual(isTSPAIR(testfile), True)
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
        self.assertEqual(isTSPAIR(testfile), False)
        # not existing file should fail
        testfile = os.path.join(self.path, 'data', 'xyz')
        self.assertEqual(isTSPAIR(testfile), False)

    def test_readTSPAIRFileSingleTrace(self):
        """
        Read TSPAIR file test via obspy.core.ascii.readTSPAIR.
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        # read
        stream = readTSPAIR(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_readTSPAIRFileMultipleTraces(self):
        """
        Read TSPAIR file test via obspy.core.ascii.readTSPAIR.
        """
        testfile = os.path.join(self.path, 'data', 'tspair_2_traces.ascii')
        # read
        stream = readTSPAIR(testfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 635)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        self.assertEqual(stream[1].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)
        # second trace
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHE')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 630)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)

    def test_readTSPAIRHeadOnly(self):
        """
        Read TSPAIR file test via obspy.core.ascii.readTSPAIR.
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        # read
        stream = readTSPAIR(testfile, headonly=True)
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 635)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        self.assertEqual(len(stream[0].data), 0)

    def test_readTSPAIRFileEncoding(self):
        """
        Read TSPAIR file test via obspy.core.ascii.readTSPAIR.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'tspair_float.ascii')
        stream = readTSPAIR(testfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 12)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # unknown encoding
        testfile = os.path.join(self.path, 'data', 'tspair_unknown.ascii')
        self.assertRaises(NotImplementedError, readTSPAIR, testfile)

    def test_writeTSPAIR(self):
        """
        Write TSPAIR file test via obspy.core.ascii.writeTSPAIR.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'tspair_float.ascii')
        stream_orig = readTSPAIR(testfile)
        tmpfile = NamedTemporaryFile().name
        # write
        writeTSPAIR(stream_orig, tmpfile)
        # read again
        stream = readTSPAIR(tmpfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 12)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # compare raw header
        lines_orig = open(testfile, 'rt').readlines()
        lines_new = open(tmpfile, 'rt').readlines()
        self.assertEqual(lines_orig[0], lines_new[0])
        # clean up
        os.remove(tmpfile)

    def test_writeTSPAIRFileMultipleTraces(self):
        """
        Write TSPAIR file test via obspy.core.ascii.writeTSPAIR.
        """
        testfile = os.path.join(self.path, 'data', 'tspair_2_traces.ascii')
        stream_orig = readTSPAIR(testfile)
        tmpfile = NamedTemporaryFile().name
        # write
        writeTSPAIR(stream_orig, tmpfile)
        # look at the raw data
        lines = open(tmpfile, 'rt').readlines()
        self.assertTrue(lines[0].startswith('TIMESERIES'))
        self.assertTrue('TSPAIR' in lines[0])
        self.assertEqual(lines[1], '2008-01-15T00:00:00.025000  185\n')
        # read again
        stream = readTSPAIR(tmpfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHE')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 630)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 635)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)
        # clean up
        os.remove(tmpfile)

    def test_writeSLIST(self):
        """
        Write SLIST file test via obspy.core.ascii.writeTSPAIR.
        """
        # float32
        testfile = os.path.join(self.path, 'data', 'slist_float.ascii')
        stream_orig = readSLIST(testfile)
        tmpfile = NamedTemporaryFile().name
        # write
        writeSLIST(stream_orig, tmpfile)
        # look at the raw data
        lines = open(tmpfile, 'rt').readlines()
        self.assertEquals(lines[0].strip(),
            'TIMESERIES XX_TEST__BHZ_R, 12 samples, 40 sps, ' + \
            '2008-01-15T00:00:00.025000, SLIST, FLOAT, Counts')
        self.assertEquals(lines[1].strip(),
            '185.009995\t181.020004\t185.029999\t189.039993\t194.050003\t' + \
            '205.059998')
        # read again
        stream = readSLIST(tmpfile)
        stream.verify()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHZ')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 12)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        data = [185.01, 181.02, 185.03, 189.04, 194.05, 205.06,
                209.07, 214.08, 222.09, 225.98, 226.99, 219.00]
        np.testing.assert_array_almost_equal(stream[0].data, data, decimal=2)
        # compare raw header
        lines_orig = open(testfile, 'rt').readlines()
        lines_new = open(tmpfile, 'rt').readlines()
        self.assertEqual(lines_orig[0], lines_new[0])
        # clean up
        os.remove(tmpfile)

    def test_writeSLISTFileMultipleTraces(self):
        """
        Write SLIST file test via obspy.core.ascii.writeTSPAIR.
        """
        testfile = os.path.join(self.path, 'data', 'slist_2_traces.ascii')
        stream_orig = readSLIST(testfile)
        tmpfile = NamedTemporaryFile().name
        # write
        writeSLIST(stream_orig, tmpfile)
        # look at the raw data
        lines = open(tmpfile, 'rt').readlines()
        self.assertTrue(lines[0].startswith('TIMESERIES'))
        self.assertTrue('SLIST' in lines[0])
        self.assertEqual(lines[1].strip(), '185\t181\t185\t189\t194\t205')
        # read again
        stream = readSLIST(tmpfile)
        stream.verify()
        # sort traces to ensure comparable results
        stream.sort()
        self.assertEqual(stream[0].stats.network, 'XX')
        self.assertEqual(stream[0].stats.station, 'TEST')
        self.assertEqual(stream[0].stats.location, '')
        self.assertEqual(stream[0].stats.channel, 'BHE')
        self.assertEqual(stream[0].stats.sampling_rate, 40.0)
        self.assertEqual(stream[0].stats.npts, 630)
        self.assertEqual(stream[0].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[0].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[0].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[0].data[-4:], data)
        # second trace
        self.assertEqual(stream[1].stats.network, 'XX')
        self.assertEqual(stream[1].stats.station, 'TEST')
        self.assertEqual(stream[1].stats.location, '')
        self.assertEqual(stream[1].stats.channel, 'BHZ')
        self.assertEqual(stream[1].stats.sampling_rate, 40.0)
        self.assertEqual(stream[1].stats.npts, 635)
        self.assertEqual(stream[1].stats.starttime,
                         UTCDateTime("2008-01-15T00:00:00.025000"))
        self.assertEqual(stream[1].stats.calib, 1.0e-00)
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [761, 755, 748, 746]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)
        # clean up
        os.remove(tmpfile)


def suite():
    return unittest.makeSuite(ASCIITestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
