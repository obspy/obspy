#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.fdsn.download_helpers test suite.

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

import obspy
from obspy.core.compatibility import mock
from obspy.fdsn.download_helpers import domain, Restrictions
from obspy.fdsn.download_helpers.utils import filter_channel_priority, \
    get_stationxml_filename, get_mseed_filename, \
    get_stationxml_contents
from obspy.fdsn.download_helpers.download_status import Channel, \
    TimeInterval, STATUS


class DomainTestCase(unittest.TestCase):
    """
    Test case for the domain definitions.
    """
    def test_rectangular_domain(self):
        """
        Test the rectangular domain.
        """
        dom = domain.RectangularDomain(-10, 10, -20, 20)
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {
            "minlatitude": -10,
            "maxlatitude": 10,
            "minlongitude": -20,
            "maxlongitude": 20})

        # The rectangular domain is completely defined by the query parameters.
        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_circular_domain(self):
        """
        Test the circular domain.
        """
        dom = domain.CircularDomain(10, 20, 30, 40)
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {
            "latitude": 10,
            "longitude": 20,
            "minradius": 30,
            "maxradius": 40})

        # The circular domain is completely defined by the query parameters.
        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_global_domain(self):
        """
        Test the global domain.
        """
        dom = domain.GlobalDomain()
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {})

        # Obviously every point is in the domain.
        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_subclassing_without_abstract_method(self):
        """
        Subclassing without implementing the get_query_parameters method
        results in a TypeError upon instantiation time.
        """
        class NewDom(domain.Domain):
            pass

        self.assertRaises(TypeError, NewDom)

    def test_instantiating_root_domain_object_fails(self):
        """
        Trying to create a root domain object should fail.
        """
        self.assertRaises(TypeError, domain.Domain)


class DownloadHelpersUtilTestCase(unittest.TestCase):
    """
    Test cases for utility functionality for the download helpers.
    """
    def __init__(self, *args, **kwargs):
        super(DownloadHelpersUtilTestCase, self).__init__(*args, **kwargs)
        self.path = os.path.dirname(__file__)
        self.data = os.path.join(self.path, "data")

    def test_channel_priority_filtering(self):
        """
        Tests the channel priority filtering.
        """
        c1 = Channel("", "BHE")
        c2 = Channel("10", "SHE")
        c3 = Channel("00", "BHZ")
        c4 = Channel("", "HHE")
        channels = [c1, c2, c3, c4]

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "HH[Z,N,E]", "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]",
                "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c4])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c1, c3])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "BH*", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c1, c3])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=[
                "BH[N,Z]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [c3])

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["S*", "BH*"])
        self.assertEqual(filtered_channels, [c2])

        # Different ways to not filter.
        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=["*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(
            channels, key="channel", priorities=None)
        self.assertEqual(filtered_channels, channels)

    def test_location_priority_filtering(self):
        """
        Tests the channel priority filtering.
        """
        c1 = Channel("", "BHE")
        c2 = Channel("10", "SHE")
        c3 = Channel("00", "BHZ")
        c4 = Channel("", "HHE")
        channels = [c1, c2, c3, c4]

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["*0"])
        self.assertEqual(filtered_channels, [c2, c3])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["00"])
        self.assertEqual(filtered_channels, [c3])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=[""])
        self.assertEqual(filtered_channels, [c1, c4])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["1?"])
        self.assertEqual(filtered_channels, [c2])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["", "*0"])
        self.assertEqual(filtered_channels, [c1, c4])

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["*0", ""])
        self.assertEqual(filtered_channels, [c2, c3])

        # Different ways to not filter.
        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=["*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(
            channels, key="location", priorities=None)
        self.assertEqual(filtered_channels, channels)


    # def test_station_list_nearest_neighbour_filter(self):
    #     """
    #     Test the filtering based on geographical distance.
    #     """
    #     # Only the one at depth 200 should be removed as it is the only one
    #     # that has two neighbours inside the filter radius.
    #     stations = [
    #         Station("11", "11", 0, 0, 0, [], None),
    #         Station("22", "22", 0, 0, 200, [], None),
    #         Station("22", "22", 0, 0, 400, [], None),
    #         Station("33", "33", 0, 0, 2000, [], None),
    #     ]
    #     filtered_stations = filter_stations(stations, 250)
    #     self.assertEqual(filtered_stations, [
    #         Station("11", "11", 0, 0, 0, [], None),
    #         Station("22", "22", 0, 0, 400, [], None),
    #         Station("33", "33", 0, 0, 2000, [], None)])
    #
    #     # The two at 200 and 250 m depth should be removed.
    #     stations = [
    #         Station("11", "11", 0, 0, 0, [], None),
    #         Station("22", "22", 0, 0, 200, [], None),
    #         Station("22", "22", 0, 0, 250, [], None),
    #         Station("22", "22", 0, 0, 400, [], None),
    #         Station("33", "33", 0, 0, 2000, [], None)]
    #     filtered_stations = filter_stations(stations, 250)
    #     self.assertEqual(filtered_stations, [
    #         Station("11", "11", 0, 0, 0, [], None),
    #         Station("22", "22", 0, 0, 400, [], None),
    #         Station("33", "33", 0, 0, 2000, [], None)])
    #
    #     # Set the distance to 1 degree and check the longitude behaviour at
    #     # the longitude wraparound point.
    #     stations = [
    #         Station("11", "11", 0, 0, 0, [], None),
    #         Station("22", "22", 0, 90, 0, [], None),
    #         Station("33", "33", 0, 180, 0, [], None),
    #         Station("44", "44", 0, -90, 0, [], None),
    #         Station("55", "55", 0, -180, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     # Only 4 stations should remain and either the one at 0,180 or the
    #     # one at 0, -180 should have been removed as they are equal.
    #     self.assertEqual(len(filtered_stations), 4)
    #     self.assertTrue(Station("11", "11", 0, 0, 0, [], None)
    #                     in filtered_stations)
    #     self.assertTrue(Station("22", "22", 0, 90, 0, [], None)
    #                     in filtered_stations)
    #     self.assertTrue(Station("44", "44", 0, -90, 0, [], None)
    #                     in filtered_stations)
    #     self.assertTrue((Station("33", "33", 0, 180, 0, [], None)
    #                      in filtered_stations) or
    #                     (Station("55", "55", 0, -180, 0, [], None)
    #                      in filtered_stations))
    #
    #     # Test filtering around the longitude wraparound.
    #     stations = [
    #         Station("11", "11", 0, 180, 0, [], None),
    #         Station("22", "22", 0, 179.2, 0, [], None),
    #         Station("33", "33", 0, 180.8, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(filtered_stations, [
    #         Station("22", "22", 0, 179.2, 0, [], None),
    #         Station("33", "33", 0, 180.8, 0, [], None)])
    #
    #     # Test the conversion of lat/lng to meter distances.
    #     stations = [
    #         Station("11", "11", 0, 180, 0, [], None),
    #         Station("22", "22", 0, -180, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 1)
    #     stations = [
    #         Station("11", "11", 0, 180, 0, [], None),
    #         Station("22", "22", 0, -179.5, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 1)
    #     stations = [
    #         Station("11", "11", 0, 180, 0, [], None),
    #         Station("22", "22", 0, -179.1, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 1)
    #     stations = [
    #         Station("11", "11", 0, 180, 0, [], None),
    #         Station("22", "22", 0, 178.9, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 2)
    #
    #     # Also test the latitude settings.
    #     stations = [
    #         Station("11", "11", 0, -90, 0, [], None),
    #         Station("22", "22", 0, -90, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 1)
    #     stations = [
    #         Station("11", "11", 0, -90, 0, [], None),
    #         Station("22", "22", 0, -89.5, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 1)
    #     stations = [
    #         Station("11", "11", 0, -90, 0, [], None),
    #         Station("22", "22", 0, -89.1, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 1)
    #     stations = [
    #         Station("11", "11", 0, -90, 0, [], None),
    #         Station("22", "22", 0, -88.9, 0, [], None)]
    #     filtered_stations = filter_stations(stations, 111000)
    #     self.assertEqual(len(filtered_stations), 2)

    def test_merge_station_lists(self):
        """
        Tests the merging of two stations.
        """
        list_one = [
            Station("11", "11", 0, 0, 0, [], None),
            Station("11", "11", 0, 0, 500, [], None),
            Station("11", "11", 0, 0, 1500, [], None),
        ]
        list_two = [
            Station("11", "11", 0, 0, 10, [], None),
            Station("11", "11", 0, 0, 505, [], None),
            Station("11", "11", 0, 0, 1505, [], None),
        ]
        new_list = filter_based_on_interstation_distance(list_one, list_two, 20)
        self.assertEqual(new_list, list_one)
        new_list = filter_based_on_interstation_distance(list_one, list_two, 2)
        self.assertEqual(new_list, list_one + list_two)
        new_list = filter_based_on_interstation_distance(list_one, list_two, 8)
        self.assertEqual(new_list, list_one + [
            Station("11", "11", 0, 0, 10, [], None)])

    def test_stationxml_filename_helper(self):
        """
        Tests the get_stationxml_filename() function.
        """
        c1 = ("", "BHE")
        c2 = ("10", "SHE")
        starttime = obspy.UTCDateTime(2012, 1, 1)
        endtime = obspy.UTCDateTime(2012, 1, 2)
        channels = [c1, c2]

        # A normal string is considered a path.
        self.assertEqual(get_stationxml_filename(
            "FOLDER", network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime),
            os.path.join("FOLDER", "BW.FURT.xml"))
        self.assertEqual(get_stationxml_filename(
            "stations", network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime),
            os.path.join("stations", "BW.FURT.xml"))

        # Passing a format string causes it to be used.
        self.assertEqual(get_stationxml_filename(
            "{network}_{station}.xml", network="BW", station="FURT",
            channels=channels, starttime=starttime, endtime=endtime),
            "BW_FURT.xml")
        self.assertEqual(get_stationxml_filename(
            "TEMP/{network}/{station}.xml", network="BW", station="FURT",
            channels=channels, starttime=starttime, endtime=endtime),
            "TEMP/BW/FURT.xml")

        # A passed function will be executed. A string should just be returned.
        def get_name(network, station, channels, starttime, endtime):
            return "network" + "__" + station
        self.assertEqual(get_stationxml_filename(
            get_name, network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime), "network__FURT")

        # A dictionary with certain keys are also acceptable.
        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": [c2],
                    "filename": "test.xml"}
        self.assertEqual(get_stationxml_filename(
            get_name, network="BW", station="FURT", channels=channels,
            starttime=starttime, endtime=endtime),
            {"missing_channels": [c1], "available_channels": [c2],
             "filename": "test.xml"})

        # Missing keys raise.
        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": [c2]}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        # Wrong value types should also raise.
        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": [c2],
                    "filename": True}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": True,
                    "available_channels": [c2],
                    "filename": "test.xml"}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        def get_name(network, station, channels, starttime, endtime):
            return {"missing_channels": [c1],
                    "available_channels": True,
                    "filename": "test.xml"}
        self.assertRaises(ValueError, get_stationxml_filename, get_name,
                          "BW", "FURT", channels, starttime, endtime)

        # It will raise a type error, if the function does not return the
        # proper type.
        self.assertRaises(TypeError, get_stationxml_filename, lambda x: 1,
                          "BW", "FURT", starttime, endtime)

    def test_mseed_filename_helper(self):
        """
        Tests the get_mseed_filename() function.
        """
        starttime = obspy.UTCDateTime(2014, 1, 2, 3, 4, 5)
        endtime = obspy.UTCDateTime(2014, 2, 3, 4, 5, 6)

        # A normal string is considered a path.
        self.assertEqual(
            get_mseed_filename("FOLDER", network="BW", station="FURT",
                               location="", channel="BHE",
                               starttime=starttime, endtime=endtime),
            os.path.join(
                "FOLDER", "BW.FURT..BHE__2014-01-02T03-04-05Z__"
                "2014-02-03T04-05-06Z.mseed"))
        self.assertEqual(
            get_mseed_filename("waveforms", network="BW", station="FURT",
                               location="00", channel="BHE",
                               starttime=starttime, endtime=endtime),
            os.path.join("waveforms", "BW.FURT.00.BHE__2014-01-02T03-04-05Z__"
                         "2014-02-03T04-05-06Z.mseed"))

        # Passing a format string causes it to be used.
        self.assertEqual(get_mseed_filename(
            "{network}_{station}_{location}_{channel}_"
            "{starttime}_{endtime}.ms", network="BW", station="FURT",
            location="", channel="BHE", starttime=starttime, endtime=endtime),
            "BW_FURT__BHE_2014-01-02T03-04-05Z_2014-02-03T04-05-06Z.ms")
        self.assertEqual(get_mseed_filename(
            "{network}_{station}_{location}_{channel}_"
            "{starttime}_{endtime}.ms", network="BW", station="FURT",
            location="00", channel="BHE", starttime=starttime,
            endtime=endtime),
            "BW_FURT_00_BHE_2014-01-02T03-04-05Z_2014-02-03T04-05-06Z.ms")

        # A passed function will be executed.
        def get_name(network, station, location, channel, starttime, endtime):
            if network == "AH":
                return True
            return "network" + "__" + station + location + channel

        # Returning a filename is possible.
        self.assertEqual(
            get_mseed_filename(get_name, network="BW", station="FURT",
                               location="", channel="BHE",
                               starttime=starttime, endtime=endtime),
            "network__FURTBHE")
        # 'True' can also be returned. This indicates that the file already
        # exists.
        self.assertEqual(
            get_mseed_filename(get_name, network="AH", station="FURT",
                               location="", channel="BHE",
                               starttime=starttime, endtime=endtime), True)

        # It will raise a type error, if the function does not return the
        # proper type.
        self.assertRaises(TypeError, get_mseed_filename, lambda x: 1,
                          "BW", "FURT", "", "BHE")

    @mock.patch("os.makedirs")
    def test_attach_miniseed_filenames(self, patch):
        """
        Tests the attach_miniseed_filenames function. Also serves as an
        integration test for the get_mseed_filename() function.
        """
        channels = [Channel("", "BHE"), Channel("", "BHN"), Channel("", "BHZ")]
        stations = [Station("BW", "ALTM", 0, 0, 0, channels=channels),
                    Station("BW", "FURT", 0, 0, 0, channels=channels)]

        # Simple folder setting.
        new_stations = attach_miniseed_filenames(stations, "waveforms")
        self.assertEqual(len(new_stations["stations_to_download"]), 2)
        self.assertEqual(new_stations["existing_miniseed_filenames"], [])
        self.assertEqual(new_stations["ignored_channel_count"], 0)
        for stat in new_stations["stations_to_download"]:
            self.assertEqual(
                [_i.mseed_filename for _i in stat.channels],
                ["waveforms/BW.FURT..BH%s.mseed" %
                 _i for _i in ["E", "N", "Z"]])
        # 6 channels thus 6 directories should have been created.
        self.assertEqual(patch.call_count, 6)
        patch.reset_mock()

        # String template.
        new_stations = attach_miniseed_filenames(
            stations,  "A/{network}/{station}/{location}{channel}.mseed")
        self.assertEqual(len(new_stations["stations_to_download"]), 2)
        self.assertEqual(new_stations["existing_miniseed_filenames"], [])
        self.assertEqual(new_stations["ignored_channel_count"], 0)
        for stat in new_stations["stations_to_download"]:
            self.assertEqual(
                [_i.mseed_filename for _i in stat.channels],
                ["A/BW/FURT/BH%s.mseed" %
                 _i for _i in ["E", "N", "Z"]])
        # Once again 6 channels.
        self.assertEqual(patch.call_count, 6)
        patch.reset_mock()

        # A function returning just returning True should result in all
        # channels being ignored.
        def get_name(*args):
            return True
        new_stations = attach_miniseed_filenames(stations, get_name)
        self.assertEqual(new_stations["stations_to_download"], [])
        self.assertEqual(new_stations["existing_miniseed_filenames"], [])
        self.assertEqual(new_stations["ignored_channel_count"], 6)

    def test_get_stationxml_contents(self):
        """
        Tests the fast get_stationxml_contents() function.
        """
        filename = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(self.data))), "station", "tests", "data",
            "AU.MEEK.xml")
        # Read with ObsPy and the fast variant.
        inv = obspy.read_inventory(filename)
        # Consistency test.
        self.assertEqual(len(inv.networks), 1)
        contents = get_stationxml_contents(filename)
        net = inv[0]
        sta = net[0]
        cha = sta[0]
        self.assertEqual(
            contents,
            [ChannelAvailability(net.code, sta.code, cha.location_code,
                                 cha.code, cha.start_date, cha.end_date,
                                 filename)])

    def test_attach_stationxml_filenames(self):
        """
        Test the attaching of filenames to the stations objects.
        """
        channels = [Channel("", "BHE"), Channel("", "BHN"), Channel("", "BHZ")]
        stations = [Station("BW", "ALTM", 0, 0, 0, channels=channels),
                    Station("BW", "FURT", 0, 0, 0, channels=channels)]

    def test_restrictions_object(self):
        """
        Tests the restrictions object.
        """
        start = obspy.UTCDateTime(2014, 1, 1)
        res = Restrictions(starttime=start, endtime=start + 10)

        # No chunklength means it should just return one item.
        chunks = list(res)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], (start, start + 10))

        # One with chunklength should return the chunked pieces.
        res = Restrictions(starttime=start, endtime=start + 10,
                           chunklength_in_sec=1)
        chunks = list(res)
        self.assertEqual(len(chunks), 10)
        self.assertEqual(
            [_i[0] for _i in chunks],
            [start + _i * 1 for _i in range(10)])
        self.assertEqual(
            [_i[1] for _i in chunks],
            [start + _i * 1 for _i in range(1, 11)])
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[-1][1], start + 10)

        # Make sure the last piece is cut if it needs to be.
        start = obspy.UTCDateTime(2012, 1, 1)
        end = obspy.UTCDateTime(2012, 2, 1)
        res = Restrictions(starttime=start, endtime=end,
                           chunklength_in_sec=86400 * 10)
        chunks = list(res)
        self.assertEqual(chunks, [
            (start, start + 86400 * 10),
            (start + 86400 * 10, start + 86400 * 20),
            (start + 86400 * 20, start + 86400 * 30),
            (start + 86400 * 30, end)])

    def test_channel_str_representation(self):
        """
        Test the string representations of channel objects.
        """
        # Single interval.
        intervals = [TimeInterval(
            obspy.UTCDateTime(2012, 1, 1), obspy.UTCDateTime(2012, 2, 1),
            filename=None, status=None)]
        c = Channel(location="", channel="BHE", intervals=intervals)
        self.assertEqual(str(c), (
            "Channel(location='', channel=BHE, intervals=[\n"
            "\tTimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 2, 1, 0, 0), filename=None, status=None)\n"
            "])"))


class TimeIntervalTestCase(unittest.TestCase):
    """
    Test cases for the TimeInterval class.
    """
    def test_repr(self):
        st = obspy.UTCDateTime(2012, 1, 1)
        et = obspy.UTCDateTime(2012, 1, 2)
        ti = TimeInterval(st, et)
        self.assertEqual(
            repr(ti),
            "TimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 1, 2, 0, 0), filename=None, status='none')")

        st = obspy.UTCDateTime(2012, 1, 1)
        et = obspy.UTCDateTime(2012, 1, 2)
        ti = TimeInterval(st, et, filename="dummy.txt")
        self.assertEqual(
            repr(ti),
            "TimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 1, 2, 0, 0), filename='dummy.txt', "
            "status='none')")

        st = obspy.UTCDateTime(2012, 1, 1)
        et = obspy.UTCDateTime(2012, 1, 2)
        ti = TimeInterval(st, et, filename="dummy.txt", status=STATUS.IGNORE)
        self.assertEqual(
            repr(ti),
            "TimeInterval(start=UTCDateTime(2012, 1, 1, 0, 0), "
            "end=UTCDateTime(2012, 1, 2, 0, 0), filename='dummy.txt', "
            "status='ignore')")


class StationTestCase(unittest.TestCase):
    """
    Test cases for the Station class.
    """
    def test_temporal_bounds(self):
        """
        Tests the temporal bounds retrieval.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c = Channel(location="", channel="BHZ", intervals=time_intervals)
        self.assertEqual(c.temporal_bounds, (st, st + 10 * 60))

    def test_wants_station_information(self):
        """
        Tests the wants station information property.
        """
        st = obspy.UTCDateTime(2015, 1, 1)
        time_intervals = [
            TimeInterval(st + _i * 60, st + (_i + 1) * 60) for _i in range(10)]
        c = Channel(location="", channel="BHZ", intervals=time_intervals)

        # Right now all intervals have status NONE.
        self.assertFalse(c.needs_station_file)

        # As soon as at least one interval has status DOWNLOADED or EXISTS,
        # a station file is required.
        c.intervals[1].status = STATUS.EXISTS
        self.assertTrue(c.needs_station_file)
        c.intervals[1].status = STATUS.DOWNLOADED
        self.assertTrue(c.needs_station_file)

        # Any other status does not trigger the need to download.
        c.intervals[1].status = STATUS.DOWNLOAD_REJECTED
        self.assertFalse(c.needs_station_file)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(unittest.makeSuite(DomainTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(DownloadHelpersUtilTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(TimeIntervalTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(StationTestCase, 'test'))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
