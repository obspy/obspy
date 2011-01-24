# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from obspy.core.ascii import readSLIST, readTSPAIR, isSLIST, isTSPAIR
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
        testfile = os.path.join(self.path, 'data', 'slist2.ascii')
        self.assertEqual(isSLIST(testfile), True)
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
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
        testfile = os.path.join(self.path, 'data', 'slist2.ascii')
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

    def test_isTSPAIRFile(self):
        """
        Testing TSPAIR file format. 
        """
        testfile = os.path.join(self.path, 'data', 'tspair.ascii')
        self.assertEqual(isTSPAIR(testfile), True)
        testfile = os.path.join(self.path, 'data', 'tspair2.ascii')
        self.assertEqual(isTSPAIR(testfile), True)
        testfile = os.path.join(self.path, 'data', 'slist.ascii')
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
        testfile = os.path.join(self.path, 'data', 'tspair2.ascii')
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
        self.assertEqual(stream[0].stats.mseed.dataquality, 'R')
        # check first 4 samples
        data = [185, 181, 185, 189]
        np.testing.assert_array_almost_equal(stream[1].data[0:4], data)
        # check last 4 samples
        data = [781, 785, 778, 772]
        np.testing.assert_array_almost_equal(stream[1].data[-4:], data)


def suite():
    return unittest.makeSuite(ASCIITestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
