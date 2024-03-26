#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.fdsn.wadl_parser test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings

from obspy import UTCDateTime
from obspy.clients.fdsn.wadl_parser import WADLParser


class TestWADLParser():
    """
    Test cases for obspy.clients.fdsn.wadl_parser.WADL_Parser.
    """
    def _parse_wadl_file(self, path):
        """
        Parses wadl, returns WADLParser and any catched warnings.
        """
        with open(path, "rb") as fh:
            wadl_string = fh.read()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            parser = WADLParser(wadl_string)
        return parser, w

    def test_dataselect_wadl_parsing(self, testdata):
        """
        Tests the parsing of a dataselect wadl.
        """
        with open(testdata["dataselect.wadl"], "rb") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        assert "starttime" in params
        assert "endtime" in params
        assert "network" in params
        assert "station" in params
        assert "location" in params
        assert "channel" in params
        assert "quality" in params
        assert "minimumlength" in params
        assert "quality" in params
        assert "longestonly" in params

        assert params["starttime"]["type"] == UTCDateTime
        assert params["starttime"]["required"]

        assert params["endtime"]["type"] == UTCDateTime
        assert params["endtime"]["required"]

        assert params["network"]["type"] == str
        assert params["station"]["type"] == str
        assert params["location"]["type"] == str
        assert params["channel"]["type"] == str

        assert sorted(params["quality"]["options"]) == \
            sorted(["D", "R", "Q", "M", "B"])

        # Check that the default values did get read correctly.
        assert params["quality"]["default_value"] == "B"
        assert params["minimumlength"]["default_value"] == 0.0
        assert not params["longestonly"]["default_value"]

    def test_event_wadl_parsing(self, testdata):
        """
        Tests the parsing of an event wadl.
        """
        parser, w = self._parse_wadl_file(testdata["event.wadl"])
        assert len(w) == 0

        params = parser.parameters

        # The WADL contains some short forms. In the parameters dictionary
        # these should be converted to the long forms.
        assert "starttime" in params
        assert "endtime" in params
        assert "minlatitude" in params
        assert "maxlatitude" in params
        assert "minlongitude" in params
        assert "maxlongitude" in params
        assert "minmagnitude" in params
        assert "maxmagnitude" in params
        assert "magnitudetype" in params
        assert "catalog" in params

        assert "contributor" in params
        assert "maxdepth" in params
        assert "mindepth" in params
        assert "latitude" in params
        assert "longitude" in params

        assert "maxradius" in params
        assert "minradius" in params
        assert "orderby" in params
        assert "updatedafter" in params

        assert "eventid" in params
        assert "originid" in params
        assert "includearrivals" in params
        assert "includeallmagnitudes" in params
        assert "includeallorigins" in params
        assert "limit" in params
        assert "offset" in params

        # The nodata attribute should not be parsed.
        assert not ("nodata" in params)

        assert params["magnitudetype"]["doc_title"] == \
            "type of Magnitude used to test minimum and maximum limits " \
            "(case insensitive)"
        assert params["magnitudetype"]["doc"] == "Examples: Ml,Ms,mb,Mw\""

    def test_availability_wadl_parsing(self):
        """
        Tests the parsing of an availability wadl.
        """
        parser, w = self._parse_wadl_file("availability.wadl")
        params = parser.parameters

        assert "starttime" in params
        assert "endtime" in params
        assert "network" in params
        assert "station" in params
        assert "location" in params
        assert "channel" in params
        assert "quality" in params
        assert "merge" in params
        assert "orderby" in params
        assert "limit" in params
        assert "includerestricted" in params
        assert "mergegaps" in params
        assert "show" in params
        assert "format" in params

        # The nodata attribute should not be parsed.
        assert not ("nodata" in params)

    def test_station_wadl_parsing(self):
        """
        Tests the parsing of a station wadl.
        """
        parser, w = self._parse_wadl_file(testdata["station.wadl"])
        params = parser.parameters

        assert "starttime" in params
        assert "endtime" in params
        assert "startbefore" in params
        assert "startafter" in params
        assert "endbefore" in params
        assert "endafter" in params
        assert "network" in params
        assert "station" in params
        assert "location" in params
        assert "channel" in params
        assert "minlatitude" in params
        assert "maxlatitude" in params
        assert "latitude" in params
        assert "minlongitude" in params
        assert "maxlongitude" in params
        assert "longitude" in params
        assert "minradius" in params
        assert "maxradius" in params
        assert "level" in params
        assert "includerestricted" in params
        assert "includeavailability" in params
        assert "updatedafter" in params
        assert "matchtimeseries" in params

        # The nodata attribute should not be parsed.
        assert not ("nodata" in params)

        assert params["endbefore"]["doc_title"] == \
            "limit to stations ending before the specified time"
        assert params["endbefore"]["doc"] == \
            "Examples: endbefore=2012-11-29 or 2012-11-29T00:00:00 or " \
            "2012-11-29T00:00:00.000"

    def test_reading_wadls_without_type(self, testdata):
        """
        Tests the reading of WADL files that have no type.
        """
        parser, w = self._parse_wadl_file(testdata["station_no_types.wadl"])
        params = parser.parameters

        # Assert that types have been assigned.
        assert params["starttime"]["type"] == UTCDateTime
        assert params["endtime"]["type"] == UTCDateTime
        assert params["startbefore"]["type"] == UTCDateTime
        assert params["startafter"]["type"] == UTCDateTime
        assert params["endbefore"]["type"] == UTCDateTime
        assert params["endafter"]["type"] == UTCDateTime
        assert params["network"]["type"] == str
        assert params["station"]["type"] == str
        assert params["location"]["type"] == str
        assert params["channel"]["type"] == str
        assert params["minlatitude"]["type"] == float
        assert params["maxlatitude"]["type"] == float
        assert params["latitude"]["type"] == float
        assert params["minlongitude"]["type"] == float
        assert params["maxlongitude"]["type"] == float
        assert params["longitude"]["type"] == float
        assert params["minradius"]["type"] == float
        assert params["maxradius"]["type"] == float
        assert params["level"]["type"] == str
        assert params["includerestricted"]["type"] == bool
        assert params["includeavailability"]["type"] == bool
        assert params["updatedafter"]["type"] == UTCDateTime

        # Now read a dataselect file with no types.
        with open(testdata["dataselect_no_types.wadl"], "rb") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        # Assert that types have been assigned.
        assert params["starttime"]["type"] == UTCDateTime
        assert params["endtime"]["type"] == UTCDateTime
        assert params["network"]["type"] == str
        assert params["station"]["type"] == str
        assert params["location"]["type"] == str
        assert params["channel"]["type"] == str
        assert params["quality"]["type"] == str
        assert params["minimumlength"]["type"] == float
        assert params["longestonly"]["type"] == bool

    def test_usgs_event_wadl_parsing(self, testdata):
        """
        Tests the parsing of an event wadl.
        """
        with open(testdata["usgs_event.wadl"], "rb") as fh:
            wadl_string = fh.read()
        parser = WADLParser(wadl_string)
        params = parser.parameters

        # The WADL contains some short forms. In the parameters dictionary
        # these should be converted to the long forms.
        assert "starttime" in params
        assert "endtime" in params
        assert "minlatitude" in params
        assert "maxlatitude" in params
        assert "minlongitude" in params
        assert "maxlongitude" in params
        assert "minmagnitude" in params
        assert "maxmagnitude" in params
        assert "magnitudetype" in params
        assert "catalog" in params

        assert "contributor" in params
        assert "maxdepth" in params
        assert "mindepth" in params
        assert "latitude" in params
        assert "longitude" in params

        assert "maxradius" in params
        assert "minradius" in params
        assert "orderby" in params
        assert "updatedafter" in params

        assert "eventid" in params
        assert "includearrivals" in params
        assert "includeallmagnitudes" in params
        assert "includeallorigins" in params
        assert "limit" in params
        assert "offset" in params

    def test_parsing_dataselect_wadls_with_missing_attributes(self, testdata):
        """
        Some WADL file miss required attributes. In this case a warning will be
        raised.

        Update: FDSN-WS 1.1 has optional arguments which are no longer
        required!
        """
        # This dataselect WADL misses the quality, minimumlength, and
        # longestonly parameters.
        with open(testdata["dataselect_missing_attributes.wadl"], "rb") as fh:
            wadl_string = fh.read()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            parser = WADLParser(wadl_string)
            # No warning should be raised due to update to FDSN-WS 1.1.
            assert len(w) == 0

        # Assert that some other parameters are still existent.
        params = parser.parameters
        assert "starttime" in params
        assert "endtime" in params
        assert "network" in params
        assert "station" in params
        assert "location" in params
        assert "channel" in params

    def test_parsing_event_wadls_with_missing_attributes(self, testdata):
        """
        Some WADL file miss required attributes. In this case a warning will be
        raised.

        Update: FDSN-WS 1.1 has optional arguments which are no longer
        required!
        """
        # This event WADL misses the includeallorigins and the updatedafter
        # parameters.
        with open(testdata["event_missing_attributes.wadl"], "rb") as fh:
            wadl_string = fh.read()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            parser = WADLParser(wadl_string)
            # No warning should be raised due to update to FDSN-WS 1.1.
            assert len(w) == 0

        # Assert that some other parameters are still existent.
        params = parser.parameters
        assert "starttime" in params
        assert "endtime" in params
        assert "minlatitude" in params
        assert "maxlatitude" in params
        assert "minlongitude" in params
        assert "maxlongitude" in params
        assert "minmagnitude" in params
        assert "maxmagnitude" in params
        assert "magnitudetype" in params
        assert "catalog" in params

    def test_parsing_current_wadls(self, testdata):
        """
        Test parsing real world wadls provided by servers as of 2014-01-07.
        """
        filenames = ["2014-01-07_iris_event.wadl",
                     "2014-01-07_iris_station.wadl",
                     "2014-01-07_iris_dataselect.wadl",
                     "2014-01-07_usgs_event.wadl",
                     "2014-02-16_seismicportal_event.wadl",
                     "2014-01-07_resif_station.wadl",
                     "2014-01-07_resif_dataselect.wadl",
                     "2014-01-07_ncedc_event.wadl",
                     "2014-01-07_ncedc_station.wadl",
                     "2014-01-07_ncedc_dataselect.wadl",
                     "2014-01-07_ethz_event.wadl",
                     "2014-01-07_ethz_station.wadl",
                     "2014-01-07_ethz_dataselect.wadl",
                     ]
        expected_list = [
            ['catalog', 'contributor', 'endtime', 'eventid', 'format',
             'includeallmagnitudes', 'includeallorigins',
             'includearrivals', 'latitude', 'limit', 'longitude',
             'magtype', 'maxdepth', 'maxlatitude', 'maxlongitude',
             'maxmagnitude', 'maxradius', 'mindepth', 'minlatitude',
             'minlongitude', 'minmagnitude', 'minradius', 'offset',
             'orderby', 'originid', 'starttime', 'updatedafter'],
            ['channel', 'endafter', 'endbefore', 'endtime', 'format',
             'includeavailability', 'includerestricted', 'latitude',
             'level', 'location', 'longitude', 'matchtimeseries',
             'maxlatitude', 'maxlongitude', 'maxradius', 'minlatitude',
             'minlongitude', 'minradius', 'network', 'startafter',
             'startbefore', 'starttime', 'station', 'updatedafter'],
            ['channel', 'endtime', 'location', 'longestonly',
             'minimumlength', 'network', 'quality', 'starttime',
             'station'],
            ['alertlevel', 'callback', 'catalog', 'contributor',
             'endtime', 'eventid', 'eventtype',
             'format', 'includeallmagnitudes',
             'includeallorigins', 'includearrivals', 'kmlanimated',
             'kmlcolorby', 'latitude', 'limit', 'longitude',
             'magnitudetype', 'maxcdi', 'maxdepth', 'maxgap',
             'maxlatitude', 'maxlongitude', 'maxmagnitude', 'maxmmi',
             'maxradius', 'maxsig', 'mincdi', 'mindepth', 'minfelt',
             'mingap', 'minlatitude', 'minlongitude', 'minmagnitude',
             'minmmi', 'minradius', 'minsig', 'offset', 'orderby',
             'producttype', 'reviewstatus', 'starttime', 'updatedafter'],
            ['callback', 'catalog', 'contributor', 'endtime', 'eventid',
             'format', 'includeallmagnitudes', 'includeallorigins',
             'includearrivals', 'latitude', 'limit', 'longitude',
             'magtype', 'maxdepth', 'maxlatitude', 'maxlongitude',
             'maxmagnitude', 'maxradius', 'mindepth', 'minlatitude',
             'minlongitude', 'minmagnitude', 'minradius', 'offset',
             'orderby', 'starttime', 'updatedafter'],
            ['channel', 'endafter', 'endbefore', 'endtime',
             'includeavailability', 'includerestricted', 'latitude',
             'level', 'location', 'longitude', 'maxlatitude',
             'maxlongitude', 'maxradius', 'minlatitude', 'minlongitude',
             'minradius', 'network', 'startafter', 'startbefore',
             'starttime', 'station', 'updatedafter'],
            ['channel', 'endtime', 'location', 'longestonly',
             'minimumlength', 'network', 'quality', 'starttime',
             'station'],
            ['catalog', 'contributor', 'endtime', 'eventid', 'format',
             'includeallmagnitudes', 'includearrivals',
             'includemechanisms', 'latitude', 'limit', 'longitude',
             'magnitudetype', 'maxdepth', 'maxlatitude', 'maxlongitude',
             'maxmagnitude', 'maxradius', 'mindepth', 'minlatitude',
             'minlongitude', 'minmagnitude', 'minradius', 'offset',
             'orderby', 'starttime'],
            ['channel', 'endafter', 'endbefore', 'endtime', 'format',
             'includeavailability', 'latitude', 'level', 'location',
             'longitude', 'maxlatitude', 'maxlongitude', 'maxradius',
             'minlatitude', 'minlongitude', 'minradius', 'network',
             'startafter', 'startbefore', 'starttime', 'station',
             'updatedafter'],
            ['channel', 'endtime', 'location', 'network', 'starttime',
             'station'],
            ['contributor', 'endtime', 'eventid', 'formatted',
             'includeallorigins', 'includearrivals', 'includecomments',
             'includemagnitudes', 'includepicks', 'latitude', 'limit',
             'longitude', 'magnitudetype', 'maxdepth', 'maxlatitude',
             'maxlongitude', 'maxmagnitude', 'maxradius', 'mindepth',
             'minlatitude', 'minlongitude', 'minmagnitude', 'minradius',
             'offset', 'orderby', 'output', 'starttime', 'updatedafter'],
            ['channel', 'endafter', 'endbefore', 'endtime', 'formatted',
             'includerestricted', 'latitude', 'level', 'location',
             'longitude', 'maxlatitude', 'maxlongitude', 'maxradius',
             'minlatitude', 'minlongitude', 'minradius', 'network',
             'output', 'startafter', 'startbefore', 'starttime',
             'station'],
            ['channel', 'endtime', 'location', 'network', 'quality',
             'starttime', 'station'],
            ]

        for filename, expected in zip(filenames, expected_list):
            parser, w = self._parse_wadl_file(testdata[filename])
            params = parser.parameters
            # Check parsed parameters
            assert sorted(params.keys()) == expected
            assert len(w) == 0
