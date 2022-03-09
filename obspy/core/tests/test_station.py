#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the station handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import pytest

from obspy import read_inventory, UTCDateTime
from obspy.core.inventory import Station
from obspy.core.util import CatchAndAssertWarnings
from obspy.core.util.testing import WarningsCapture


@pytest.mark.usefixtures('ignore_numpy_errors')
class TestStation:
    """
    Tests the for :class:`~obspy.core.inventory.station.Station` class.
    """
    def test_station_response_plot(self, image_path):
        """
        Tests the response plot.
        """
        sta = read_inventory()[0][0]
        with WarningsCapture():
            sta.plot(0.05, channel="*[NE]", outfile=image_path)

    def test_response_plot_degrees(self, image_path):
        """
        Tests the response plot.
        """
        sta = read_inventory()[0][0]
        with WarningsCapture():
            sta.plot(0.05, channel="*[NE]", plot_degrees=True,
                     outfile=image_path)

    def test_len(self):
        """
        Tests the __len__ property.
        """
        sta = read_inventory()[0][0]

        assert len(sta) == len(sta.channels)
        assert len(sta) == 12

    def test_station_select(self):
        """
        Tests the select() method on station objects.
        """
        sta = read_inventory()[0][0]

        # Basic assertions to make sure the test data does not change.
        assert len(sta) == 12
        assert sta.code == "FUR"
        out = sorted(["%s.%s" % (_i.location_code, _i.code) for _i in sta])
        expected = (['.BHE', '.BHN', '.BHZ', '.HHE', '.HHN', '.HHZ',
                     '.LHE', '.LHN', '.LHZ', '.VHE', '.VHN', '.VHZ'])
        assert out == expected

        assert sta[0].code == "HHZ"
        # Manually set the end-date of the first one.
        sta[0].end_date = UTCDateTime(2010, 1, 1)

        # If nothing is given, nothing should change.
        sta_2 = sta.select()
        assert len(sta_2) == 12
        assert sta_2.code == "FUR"

        # Only select vertical channels.
        sta_2 = sta.select(channel="*Z")
        assert len(sta_2) == 4
        assert sta_2.code == "FUR"
        out = sorted(["%s.%s" % (_i.location_code, _i.code) for _i in sta_2])
        assert out == ['.BHZ', '.HHZ', '.LHZ', '.VHZ']

        # Only BH channels.
        sta_2 = sta.select(channel="BH?")
        assert len(sta_2) == 3
        assert sta_2.code == "FUR"

        out = sorted(["%s.%s" % (_i.location_code, _i.code) for _i in sta_2])
        assert out == ['.BHE', '.BHN', '.BHZ']

        # All location codes.
        sta_2 = sta.select(location="*")
        assert len(sta_2) == 12
        assert sta_2.code == "FUR"

        sta_2 = sta.select(location="")
        assert len(sta_2) == 12
        assert sta_2.code == "FUR"

        # None exist with this code.
        sta_2 = sta.select(location="10")
        assert len(sta_2) == 0
        assert sta_2.code == "FUR"

        # The time parameter selects channels active at that particular
        # time. All channels start 2006-12-16 and only the first ends in
        # 2010-1-1. All others don't have an end-date set.
        assert len(sta.select(time=UTCDateTime(2005, 1, 1))) == 0
        assert len(sta.select(time=UTCDateTime(2007, 1, 1))) == 12
        assert len(sta.select(time=UTCDateTime(2006, 12, 15))) == 0
        assert len(sta.select(time=UTCDateTime(2006, 12, 17))) == 12
        assert len(sta.select(time=UTCDateTime(2012, 1, 1))) == 11

        # Test starttime parameter.
        assert len(sta.select(starttime=UTCDateTime(2005, 1, 1))) == 12
        assert len(sta.select(starttime=UTCDateTime(2009, 1, 1))) == 12
        assert len(sta.select(starttime=UTCDateTime(2011, 1, 1))) == 11
        assert len(sta.select(starttime=UTCDateTime(2016, 1, 1))) == 11

        # Test endtime parameter.
        assert len(sta.select(endtime=UTCDateTime(2005, 1, 1))) == 0
        assert len(sta.select(endtime=UTCDateTime(2009, 1, 1))) == 12
        assert len(sta.select(endtime=UTCDateTime(2011, 1, 1))) == 12
        assert len(sta.select(endtime=UTCDateTime(2016, 1, 1))) == 12

        # Sampling rate parameter.
        assert len(sta.select(sampling_rate=33.0)) == 0
        assert len(sta.select(sampling_rate=100.0)) == 3
        assert len(sta.select(sampling_rate=20.0)) == 3
        assert len(sta.select(sampling_rate=1.0)) == 3
        assert len(sta.select(sampling_rate=0.1)) == 3

        out = sorted(["%s.%s" % (_i.location_code, _i.code) for _i
                     in sta.select(sampling_rate=100.0)])
        assert out == ['.HHE', '.HHN', '.HHZ']

        # Check tolerances.
        assert len(sta.select(sampling_rate=33.0 + 1E-6)) == 0
        assert len(sta.select(sampling_rate=100.0 + 1E-6)) == 3
        assert len(sta.select(sampling_rate=20.0 - 1E-6)) == 3
        assert len(sta.select(sampling_rate=1.0 + 1E-6)) == 3
        assert len(sta.select(sampling_rate=0.1 - 1E-6)) == 3

        # Artificially set different coordinates for a channel of RJOB.
        sta = read_inventory()[1][0]
        sta[0].latitude = 47.9
        sta[0].longitude = 12.9
        out = sta.select(
            minlatitude=47.8, maxlatitude=48,
            minlongitude=12.8, maxlongitude=13)
        assert len(out) == 1
        assert len(sta.select(
            latitude=47.95, longitude=12.95, maxradius=0.1)) == 1
        assert len(sta.select(
            latitude=47.95, longitude=12.95, minradius=0.1)) == 2
        assert len(sta.select(
            latitude=47.95, longitude=12.95,
            minradius=0.08, maxradius=0.1)) == 0

    def test_warn_identifier_invalid_uri_syntax(self):
        """
        Tests the warning on Identifiers getting set with an invalid URI (not
        having scheme-colon-path)
        """
        sta = Station(code='A', latitude=1, longitude=1, elevation=1)
        invalid_uri = "this-has-no-URI-scheme-and-no-colon"
        msg = f"Given string seems to not be a valid URI: '{invalid_uri}'"
        with CatchAndAssertWarnings(expected=[(UserWarning, msg)]):
            sta.identifiers = [invalid_uri]
