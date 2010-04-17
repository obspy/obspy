# -*- coding: utf-8 -*-

from obspy.db.util import parseMappingData, createPreview
from obspy.core import UTCDateTime, Trace
import numpy as np
import unittest


class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.db.util.
    """

    def test_parseMappingData(self):
        """
        Tests for function parseMappingData.
        """
        #1
        data = ["BW.MANZ.00.EHE GE.ROTZ..EHZ 1970-01-01 2007-12-31",
                "BW.MANZ.00.EHE GE.ROTZ..EHZ 2008-01-01",
                " ",
                ".MANZ.00.EHE GE.ROTZ..EHZ",
                "# comment",
                "BW...EHE GE.ROTZ..EHZ"]
        results = parseMappingData(data)
        self.assertEquals(len(results['.MANZ.00.EHE']), 1)
        self.assertEquals(results['.MANZ.00.EHE'][0]['network'], 'GE')
        self.assertEquals(results['.MANZ.00.EHE'][0]['station'], 'ROTZ')
        self.assertEquals(results['.MANZ.00.EHE'][0]['location'], '')
        self.assertEquals(results['.MANZ.00.EHE'][0]['channel'], 'EHZ')
        self.assertEquals(results['.MANZ.00.EHE'][0]['starttime'], None)
        self.assertEquals(results['.MANZ.00.EHE'][0]['endtime'], None)
        self.assertEquals(len(results['BW.MANZ.00.EHE']), 2)
        self.assertEquals(len(results['BW...EHE']), 1)
        #2 invalid ids
        data = ["BWMANZ00EHE GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["BW.MANZ.00EHE GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["BW.MANZ.00.EHE. GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["XXX.MANZ.00.EHE GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["BW.XXXXXX.00.EHE GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["BW.MANZ.XXX.EHE GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["BW.MANZ.00.XXXX GE.ROTZ..EHZ"]
        self.assertRaises(Exception, parseMappingData, data)
        #3 invalid date/times
        data = ["BW.MANZ.00.EHE GE.ROTZ..EHZ 2008 2009"]
        self.assertRaises(Exception, parseMappingData, data)
        data = ["BW.MANZ.00.EHE GE.ROTZ..EHZ 2009-01-01 2008-01-01"]
        self.assertRaises(Exception, parseMappingData, data)

    def test_createPreview(self):
        """
        Test for creating preview.
        """
        #1
        trace = Trace(data=np.array([0] * 28 + [0, 1] * 30 + [-1, 1] * 29))
        trace.stats.starttime = UTCDateTime(32)
        preview = createPreview(trace, delta=60.0)
        self.assertEqual(preview.stats.starttime, UTCDateTime(60))
        self.assertEqual(preview.stats.endtime, UTCDateTime(120))
        self.assertEqual(preview.stats.delta, 60.0)
        np.testing.assert_array_equal(preview.data, np.array([1, 2]))
        #2
        trace = Trace(data=np.arange(0, 30))
        preview = createPreview(trace, delta=60.0)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(0))
        self.assertEqual(preview.stats.delta, 60.0)
        np.testing.assert_array_equal(preview.data, np.array([29]))
        #2
        trace = Trace(data=np.arange(0, 60))
        preview = createPreview(trace, delta=60.0)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(0))
        self.assertEqual(preview.stats.delta, 60.0)
        np.testing.assert_array_equal(preview.data, np.array([59]))
        #3
        trace = Trace(data=np.arange(0, 90))
        preview = createPreview(trace, delta=60.0)
        self.assertEqual(preview.stats.starttime, UTCDateTime(0))
        self.assertEqual(preview.stats.endtime, UTCDateTime(60))
        self.assertEqual(preview.stats.delta, 60.0)
        np.testing.assert_array_equal(preview.data, np.array([59, 29]))



def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
