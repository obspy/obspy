#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.fdsn.wadl_parser test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

from obspy import UTCDateTime
from obspy.clients.fdsn.wadl_parser import WADLParser


class WADLParserTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.fdsn.wadl_parser.WADL_Parser.
    """
    def setUp(self):
        # directory where the test files are located
        self.data_path = os.path.join(os.path.dirname(__file__), "data")

    def _parse_wadl_file(self, filename):
        """
        Parses wadl, returns WADLParser and any catched warnings.
        """
        filename = os.path.join(self.data_path, filename)
        with open(filename, "rb") as fh:
            wadl_string = fh.read()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            parser = WADLParser(wadl_string)
        return parser, w

    def test_dataselect_wadl_parsing(self):
        """
        Tests the parsing of a dataselect wadl.
        """
        filename = os.path.join(self.data_path, "dataselect.wadl")
        with open(filename, "rb") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        self.assertIn("starttime", params)
        self.assertIn("endtime", params)
        self.assertIn("network", params)
        self.assertIn("station", params)
        self.assertIn("location", params)
        self.assertIn("channel", params)
        self.assertIn("quality", params)
        self.assertIn("minimumlength", params)
        self.assertIn("quality", params)
        self.assertIn("longestonly", params)

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

        # Check that the default values did get read correctly.
        self.assertEqual(params["quality"]["default_value"], "B")
        self.assertEqual(params["minimumlength"]["default_value"], 0.0)
        self.assertEqual(params["longestonly"]["default_value"], False)

    def test_event_wadl_parsing(self):
        """
        Tests the parsing of an event wadl.
        """
        parser, w = self._parse_wadl_file("event.wadl")
        self.assertEqual(len(w), 0)

        params = parser.parameters

        # The WADL contains some short forms. In the parameters dictionary
        # these should be converted to the long forms.
        self.assertIn("starttime", params)
        self.assertIn("endtime", params)
        self.assertIn("minlatitude", params)
        self.assertIn("maxlatitude", params)
        self.assertIn("minlongitude", params)
        self.assertIn("maxlongitude", params)
        self.assertIn("minmagnitude", params)
        self.assertIn("maxmagnitude", params)
        self.assertIn("magnitudetype", params)
        self.assertIn("catalog", params)

        self.assertIn("contributor", params)
        self.assertIn("maxdepth", params)
        self.assertIn("mindepth", params)
        self.assertIn("latitude", params)
        self.assertIn("longitude", params)

        self.assertIn("maxradius", params)
        self.assertIn("minradius", params)
        self.assertIn("orderby", params)
        self.assertIn("updatedafter", params)

        self.assertIn("eventid", params)
        self.assertIn("originid", params)
        self.assertIn("includearrivals", params)
        self.assertIn("includeallmagnitudes", params)
        self.assertIn("includeallorigins", params)
        self.assertIn("limit", params)
        self.assertIn("offset", params)

        # The nodata attribute should not be parsed.
        self.assertFalse("nodata" in params)
        # Same for the format attribute.
        self.assertFalse("format" in params)

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
        parser, w = self._parse_wadl_file(filename)
        params = parser.parameters

        self.assertIn("starttime", params)
        self.assertIn("endtime", params)
        self.assertIn("startbefore", params)
        self.assertIn("startafter", params)
        self.assertIn("endbefore", params)
        self.assertIn("endafter", params)
        self.assertIn("network", params)
        self.assertIn("station", params)
        self.assertIn("location", params)
        self.assertIn("channel", params)
        self.assertIn("minlatitude", params)
        self.assertIn("maxlatitude", params)
        self.assertIn("latitude", params)
        self.assertIn("minlongitude", params)
        self.assertIn("maxlongitude", params)
        self.assertIn("longitude", params)
        self.assertIn("minradius", params)
        self.assertIn("maxradius", params)
        self.assertIn("level", params)
        self.assertIn("includerestricted", params)
        self.assertIn("includeavailability", params)
        self.assertIn("updatedafter", params)
        self.assertIn("matchtimeseries", params)

        # The nodata attribute should not be parsed.
        self.assertFalse("nodata" in params)
        # Same for the format attribute.
        self.assertFalse("format" in params)

        self.assertEqual(
            params["endbefore"]["doc_title"],
            "limit to stations ending before the specified time")
        self.assertEqual(
            params["endbefore"]["doc"],
            "Examples: endbefore=2012-11-29 or 2012-11-29T00:00:00 or "
            "2012-11-29T00:00:00.000")

    def test_reading_wadls_without_type(self):
        """
        Tests the reading of WADL files that have no type.
        """
        filename = os.path.join(self.data_path, "station_no_types.wadl")
        parser, w = self._parse_wadl_file(filename)
        params = parser.parameters

        # Assert that types have been assigned.
        self.assertEqual(params["starttime"]["type"], UTCDateTime)
        self.assertEqual(params["endtime"]["type"], UTCDateTime)
        self.assertEqual(params["startbefore"]["type"], UTCDateTime)
        self.assertEqual(params["startafter"]["type"], UTCDateTime)
        self.assertEqual(params["endbefore"]["type"], UTCDateTime)
        self.assertEqual(params["endafter"]["type"], UTCDateTime)
        self.assertEqual(params["network"]["type"], str)
        self.assertEqual(params["station"]["type"], str)
        self.assertEqual(params["location"]["type"], str)
        self.assertEqual(params["channel"]["type"], str)
        self.assertEqual(params["minlatitude"]["type"], float)
        self.assertEqual(params["maxlatitude"]["type"], float)
        self.assertEqual(params["latitude"]["type"], float)
        self.assertEqual(params["minlongitude"]["type"], float)
        self.assertEqual(params["maxlongitude"]["type"], float)
        self.assertEqual(params["longitude"]["type"], float)
        self.assertEqual(params["minradius"]["type"], float)
        self.assertEqual(params["maxradius"]["type"], float)
        self.assertEqual(params["level"]["type"], str)
        self.assertEqual(params["includerestricted"]["type"], bool)
        self.assertEqual(params["includeavailability"]["type"], bool)
        self.assertEqual(params["updatedafter"]["type"], UTCDateTime)

        # Now read a dataselect file with no types.
        filename = os.path.join(self.data_path, "dataselect_no_types.wadl")
        with open(filename, "rb") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        # Assert that types have been assigned.
        self.assertEqual(params["starttime"]["type"], UTCDateTime)
        self.assertEqual(params["endtime"]["type"], UTCDateTime)
        self.assertEqual(params["network"]["type"], str)
        self.assertEqual(params["station"]["type"], str)
        self.assertEqual(params["location"]["type"], str)
        self.assertEqual(params["channel"]["type"], str)
        self.assertEqual(params["quality"]["type"], str)
        self.assertEqual(params["minimumlength"]["type"], float)
        self.assertEqual(params["longestonly"]["type"], bool)

    def test_usgs_event_wadl_parsing(self):
        """
        Tests the parsing of an event wadl.
        """
        filename = os.path.join(self.data_path, "usgs_event.wadl")
        with open(filename, "rb") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        # The WADL contains some short forms. In the parameters dictionary
        # these should be converted to the long forms.
        self.assertIn("starttime", params)
        self.assertIn("endtime", params)
        self.assertIn("minlatitude", params)
        self.assertIn("maxlatitude", params)
        self.assertIn("minlongitude", params)
        self.assertIn("maxlongitude", params)
        self.assertIn("minmagnitude", params)
        self.assertIn("maxmagnitude", params)
        self.assertIn("magnitudetype", params)
        self.assertIn("catalog", params)

        self.assertIn("contributor", params)
        self.assertIn("maxdepth", params)
        self.assertIn("mindepth", params)
        self.assertIn("latitude", params)
        self.assertIn("longitude", params)

        self.assertIn("maxradius", params)
        self.assertIn("minradius", params)
        self.assertIn("orderby", params)
        self.assertIn("updatedafter", params)

        self.assertIn("eventid", params)
        self.assertIn("includearrivals", params)
        self.assertIn("includeallmagnitudes", params)
        self.assertIn("includeallorigins", params)
        self.assertIn("limit", params)
        self.assertIn("offset", params)

    def test_parsing_dataselect_wadls_with_missing_attributes(self):
        """
        Some WADL file miss required attributes. In this case a warning will be
        raised.

        Update: FDSN-WS 1.1 has optional arguments which are no longer
        required!
        """
        # This dataselect WADL misses the quality, minimumlength, and
        # longestonly parameters.
        filename = os.path.join(self.data_path,
                                "dataselect_missing_attributes.wadl")
        with open(filename, "rb") as fh:
            wadl_string = fh.read()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            parser = WADLParser(wadl_string)
            # No warning should be raised due to update to FDSN-WS 1.1.
            self.assertEqual(len(w), 0)

        # Assert that some other parameters are still existent.
        params = parser.parameters
        self.assertIn("starttime", params)
        self.assertIn("endtime", params)
        self.assertIn("network", params)
        self.assertIn("station", params)
        self.assertIn("location", params)
        self.assertIn("channel", params)

    def test_parsing_event_wadls_with_missing_attributes(self):
        """
        Some WADL file miss required attributes. In this case a warning will be
        raised.

        Update: FDSN-WS 1.1 has optional arguments which are no longer
        required!
        """
        # This event WADL misses the includeallorigins and the updatedafter
        # parameters.
        filename = os.path.join(self.data_path,
                                "event_missing_attributes.wadl")
        with open(filename, "rb") as fh:
            wadl_string = fh.read()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            parser = WADLParser(wadl_string)
            # No warning should be raised due to update to FDSN-WS 1.1.
            self.assertEqual(len(w), 0)

        # Assert that some other parameters are still existent.
        params = parser.parameters
        self.assertIn("starttime", params)
        self.assertIn("endtime", params)
        self.assertIn("minlatitude", params)
        self.assertIn("maxlatitude", params)
        self.assertIn("minlongitude", params)
        self.assertIn("maxlongitude", params)
        self.assertIn("minmagnitude", params)
        self.assertIn("maxmagnitude", params)
        self.assertIn("magnitudetype", params)
        self.assertIn("catalog", params)

    def test_parsing_current_wadls_iris(self):
        """
        Test parsing real world wadls provided by servers as of 2014-01-07.
        """
        parser, w = self._parse_wadl_file("2014-01-07_iris_event.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['catalog', 'contributor', 'endtime', 'eventid',
                    'includeallmagnitudes', 'includeallorigins',
                    'includearrivals', 'latitude', 'limit', 'longitude',
                    'magtype', 'maxdepth', 'maxlatitude', 'maxlongitude',
                    'maxmagnitude', 'maxradius', 'mindepth', 'minlatitude',
                    'minlongitude', 'minmagnitude', 'minradius', 'offset',
                    'orderby', 'originid', 'starttime', 'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_iris_station.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endafter', 'endbefore', 'endtime',
                    'includeavailability', 'includerestricted', 'latitude',
                    'level', 'location', 'longitude', 'matchtimeseries',
                    'maxlatitude', 'maxlongitude', 'maxradius', 'minlatitude',
                    'minlongitude', 'minradius', 'network', 'startafter',
                    'startbefore', 'starttime', 'station', 'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_iris_dataselect.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endtime', 'location', 'longestonly',
                    'minimumlength', 'network', 'quality', 'starttime',
                    'station']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

    def test_parsing_current_wadls_usgs(self):
        """
        Test parsing real world wadls provided by servers as of 2014-01-07.
        """
        parser, w = self._parse_wadl_file("2014-01-07_usgs_event.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['alertlevel', 'callback', 'catalog', 'contributor',
                    'endtime', 'eventid', 'eventtype', 'includeallmagnitudes',
                    'includeallorigins', 'includearrivals', 'kmlanimated',
                    'kmlcolorby', 'latitude', 'limit', 'longitude',
                    'magnitudetype', 'maxcdi', 'maxdepth', 'maxgap',
                    'maxlatitude', 'maxlongitude', 'maxmagnitude', 'maxmmi',
                    'maxradius', 'maxsig', 'mincdi', 'mindepth', 'minfelt',
                    'mingap', 'minlatitude', 'minlongitude', 'minmagnitude',
                    'minmmi', 'minradius', 'minsig', 'offset', 'orderby',
                    'producttype', 'reviewstatus', 'starttime', 'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

    def test_parsing_current_wadls_seismicportal(self):
        """
        Test parsing real world wadls provided by servers as of 2014-02-16.
        """
        parser, w = \
            self._parse_wadl_file("2014-02-16_seismicportal_event.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['callback', 'catalog', 'contributor', 'endtime', 'eventid',
                    'includeallmagnitudes', 'includeallorigins',
                    'includearrivals', 'latitude', 'limit', 'longitude',
                    'magtype', 'maxdepth', 'maxlatitude', 'maxlongitude',
                    'maxmagnitude', 'maxradius', 'mindepth', 'minlatitude',
                    'minlongitude', 'minmagnitude', 'minradius', 'offset',
                    'orderby', 'starttime', 'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

    def test_parsing_current_wadls_resif(self):
        """
        Test parsing real world wadls provided by servers as of 2014-01-07.
        """
        parser, w = self._parse_wadl_file("2014-01-07_resif_station.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endafter', 'endbefore', 'endtime',
                    'includeavailability', 'includerestricted', 'latitude',
                    'level', 'location', 'longitude', 'maxlatitude',
                    'maxlongitude', 'maxradius', 'minlatitude', 'minlongitude',
                    'minradius', 'network', 'startafter', 'startbefore',
                    'starttime', 'station', 'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_resif_dataselect.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endtime', 'location', 'longestonly',
                    'minimumlength', 'network', 'quality', 'starttime',
                    'station']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

    def test_parsing_current_wadls_ncedc(self):
        """
        Test parsing real world wadls provided by servers as of 2014-01-07.
        """
        parser, w = self._parse_wadl_file("2014-01-07_ncedc_event.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['catalog', 'contributor', 'endtime', 'eventid',
                    'includeallmagnitudes', 'includearrivals',
                    'includemechanisms', 'latitude', 'limit', 'longitude',
                    'magnitudetype', 'maxdepth', 'maxlatitude', 'maxlongitude',
                    'maxmagnitude', 'maxradius', 'mindepth', 'minlatitude',
                    'minlongitude', 'minmagnitude', 'minradius', 'offset',
                    'orderby', 'starttime']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_ncedc_station.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endafter', 'endbefore', 'endtime',
                    'includeavailability', 'latitude', 'level', 'location',
                    'longitude', 'maxlatitude', 'maxlongitude', 'maxradius',
                    'minlatitude', 'minlongitude', 'minradius', 'network',
                    'startafter', 'startbefore', 'starttime', 'station',
                    'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_ncedc_dataselect.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endtime', 'location', 'network', 'starttime',
                    'station']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

    def test_parsing_current_wadls_ethz(self):
        """
        Test parsing real world wadls provided by servers as of 2014-01-07.
        """
        parser, w = self._parse_wadl_file("2014-01-07_ethz_event.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['contributor', 'endtime', 'eventid', 'formatted',
                    'includeallorigins', 'includearrivals', 'includecomments',
                    'includemagnitudes', 'includepicks', 'latitude', 'limit',
                    'longitude', 'magnitudetype', 'maxdepth', 'maxlatitude',
                    'maxlongitude', 'maxmagnitude', 'maxradius', 'mindepth',
                    'minlatitude', 'minlongitude', 'minmagnitude', 'minradius',
                    'offset', 'orderby', 'output', 'starttime', 'updatedafter']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_ethz_station.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endafter', 'endbefore', 'endtime', 'formatted',
                    'includerestricted', 'latitude', 'level', 'location',
                    'longitude', 'maxlatitude', 'maxlongitude', 'maxradius',
                    'minlatitude', 'minlongitude', 'minradius', 'network',
                    'output', 'startafter', 'startbefore', 'starttime',
                    'station']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)

        parser, w = self._parse_wadl_file("2014-01-07_ethz_dataselect.wadl")
        params = parser.parameters
        # Check parsed parameters
        expected = ['channel', 'endtime', 'location', 'network', 'quality',
                    'starttime', 'station']
        self.assertEqual(sorted(params.keys()), expected)
        self.assertEqual(len(w), 0)


def suite():
    return unittest.makeSuite(WADLParserTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
