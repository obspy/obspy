# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from obspy.db.util import parseMappingData
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
        self.assertEqual(len(results['.MANZ.00.EHE']), 1)
        self.assertEqual(results['.MANZ.00.EHE'][0]['network'], 'GE')
        self.assertEqual(results['.MANZ.00.EHE'][0]['station'], 'ROTZ')
        self.assertEqual(results['.MANZ.00.EHE'][0]['location'], '')
        self.assertEqual(results['.MANZ.00.EHE'][0]['channel'], 'EHZ')
        self.assertEqual(results['.MANZ.00.EHE'][0]['starttime'], None)
        self.assertEqual(results['.MANZ.00.EHE'][0]['endtime'], None)
        self.assertEqual(len(results['BW.MANZ.00.EHE']), 2)
        self.assertEqual(len(results['BW...EHE']), 1)
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


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
