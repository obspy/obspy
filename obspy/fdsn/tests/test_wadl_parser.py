# -*- coding: utf-8 -*-
"""
The obspy.fdsn.wadl_parser test suite.
"""
from obspy import UTCDateTime

from obspy.fdsn.wadl_parser import WADLParser
import os
import unittest


class WADLParserTestCase(unittest.TestCase):
    """
    Test cases for obspy.fdsn.wadl_parser.WADL_Parser.
    """
    def setUp(self):
        # directory where the test files are located
        self.data_path = os.path.join(os.path.dirname(__file__), "data")

    def test_dataselect_wadl_parsing(self):
        """
        Tests the parsing of a dataselect wadl.
        """
        filename = os.path.join(self.data_path, "dataselect.wadl")
        with open(filename, "rt") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        self.assertTrue("starttime" in params)
        self.assertTrue("endtime" in params)
        self.assertTrue("network" in params)
        self.assertTrue("station" in params)
        self.assertTrue("location" in params)
        self.assertTrue("channel" in params)
        self.assertTrue("quality" in params)
        self.assertTrue("minimumlength" in params)
        self.assertTrue("quality" in params)
        self.assertTrue("longestonly" in params)

        self.assertEqual(params["starttime"]["type"], UTCDateTime)
        self.assertEqual(params["starttime"]["required"], True)

        self.assertEqual(params["endtime"]["type"], UTCDateTime)
        self.assertEqual(params["endtime"]["required"], True)

        self.assertEqual(params["network"]["type"], str)
        self.assertEqual(params["station"]["type"], str)
        self.assertEqual(params["location"]["type"], str)
        self.assertEqual(params["channel"]["type"], str)

        self.assertEqual(sorted(params["quality"]["options"]),
                         sorted(["D", "R", "Q", "M", "B"]))

    def test_event_wadl_parsing(self):
        """
        Tests the parsing of an event wadl.
        """
        filename = os.path.join(self.data_path, "event.wadl")
        with open(filename, "rt") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        # The WADL contains some short forms. In the parameters dictionary
        # these should be converted to the long forms.
        self.assertTrue("starttime" in params)
        self.assertTrue("endtime" in params)
        self.assertTrue("minlatitude" in params)
        self.assertTrue("maxlatitude" in params)
        self.assertTrue("minlongitude" in params)
        self.assertTrue("maxlongitude" in params)
        self.assertTrue("minmagnitude" in params)
        self.assertTrue("maxmagnitude" in params)
        self.assertTrue("magnitudetype" in params)
        self.assertTrue("catalog" in params)

        self.assertTrue("contributor" in params)
        self.assertTrue("maxdepth" in params)
        self.assertTrue("mindepth" in params)
        self.assertTrue("latitude" in params)
        self.assertTrue("longitude" in params)

        self.assertTrue("maxradius" in params)
        self.assertTrue("minradius" in params)
        self.assertTrue("orderby" in params)
        self.assertTrue("updatedafter" in params)

        self.assertTrue("eventid" in params)
        self.assertTrue("originid" in params)
        self.assertTrue("includearrivals" in params)
        self.assertTrue("includeallmagnitudes" in params)
        self.assertTrue("includeallorigins" in params)
        self.assertTrue("limit" in params)
        self.assertTrue("offset" in params)
        self.assertTrue("format" in params)

        # The nodata attribute should not be parsed.
        self.assertFalse("nodata" in params)

        self.assertEqual(
            params["magnitudetype"]["doc_title"],
            "type of Magnitude used to test minimum and maximum limits "
            "(case insensitive)")
        self.assertEqual(params["magnitudetype"]["doc"],
                         "Examples: Ml,Ms,mb,Mw\"")

    def test_station_wadl_parsing(self):
        """
        Tests the parsing of a station wadl.
        """
        filename = os.path.join(self.data_path, "station.wadl")
        with open(filename, "rt") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        self.assertTrue("starttime" in params)
        self.assertTrue("endtime" in params)
        self.assertTrue("startbefore" in params)
        self.assertTrue("startafter" in params)
        self.assertTrue("endbefore" in params)
        self.assertTrue("endafter" in params)
        self.assertTrue("network" in params)
        self.assertTrue("station" in params)
        self.assertTrue("location" in params)
        self.assertTrue("channel" in params)
        self.assertTrue("minlatitude" in params)
        self.assertTrue("maxlatitude" in params)
        self.assertTrue("latitude" in params)
        self.assertTrue("minlongitude" in params)
        self.assertTrue("maxlongitude" in params)
        self.assertTrue("longitude" in params)
        self.assertTrue("minradius" in params)
        self.assertTrue("maxradius" in params)
        self.assertTrue("level" in params)
        self.assertTrue("includerestricted" in params)
        self.assertTrue("includeavailability" in params)
        self.assertTrue("updatedafter" in params)
        self.assertTrue("matchtimeseries" in params)
        self.assertTrue("format" in params)

        # The nodata attribute should not be parsed.
        self.assertFalse("nodata" in params)

        self.assertEqual(
            params["endbefore"]["doc_title"],
            "limit to stations ending before the specified time")
        self.assertEqual(
            params["endbefore"]["doc"],
            "Examples: endbefore=2012-11-29 or 2012-11-29T00:00:00 or "
            "2012-11-29T00:00:00.000")


def suite():
    return unittest.makeSuite(WADLParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
