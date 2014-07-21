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

from obspy.fdsn.download_helpers import domain
from obspy.fdsn.download_helpers.utils import filter_channel_priority, \
    filter_stations, merge_stations, Station, Channel, \
    filter_channels_based_on_count, get_stationxml_filename, get_mseed_filename


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

        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_global_domain(self):
        """
        Test the global domain.
        """
        dom = domain.GlobalDomain()
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {})

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
    def test_channel_priority_filtering(self):
        """
        Tests the channel priority filtering.
        """
        channels = ["BHE", "SHE", "BHZ", "HHE"]

        filtered_channels = filter_channel_priority(channels, priorities=[
            "HH[Z,N,E]", "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["HHE"])

        filtered_channels = filter_channel_priority(channels, priorities=[
            "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["BHE", "BHZ"])

        filtered_channels = filter_channel_priority(channels, priorities=[
            "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [])

        filtered_channels = filter_channel_priority(channels, priorities=[
            "*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = filter_channel_priority(channels, priorities=[
            "BH*", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["BHE", "BHZ"])

        filtered_channels = filter_channel_priority(channels, priorities=[
            "BH[N,Z]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["BHZ"])

        filtered_channels = filter_channel_priority(channels, priorities=[
            "S*", "BH*"])
        self.assertEqual(filtered_channels, ["SHE"])

        # Different ways to not filter.
        filtered_channels = filter_channel_priority(channels, priorities=[
            "*"])
        self.assertEqual(filtered_channels, ["BHE", "SHE", "BHZ", "HHE"])

        filtered_channels = filter_channel_priority(channels,
                                                    priorities=None)
        self.assertEqual(filtered_channels, ["BHE", "SHE", "BHZ", "HHE"])

    def test_station_list_nearest_neighbour_filter(self):
        """
        Test the filtering based on geographical distance.
        """
        # Only the one at depth 200 should be removed as it is the only one
        # that has two neighbours inside the filter radius.
        stations = [
            Station("11", "11", 0, 0, 0, [], None),
            Station("22", "22", 0, 0, 200, [], None),
            Station("22", "22", 0, 0, 400, [], None),
            Station("33", "33", 0, 0, 2000, [], None),
        ]
        filtered_stations = filter_stations(stations, 250)
        self.assertEqual(filtered_stations, [
            Station("11", "11", 0, 0, 0, [], None),
            Station("22", "22", 0, 0, 400, [], None),
            Station("33", "33", 0, 0, 2000, [], None)])

        # The two at 200 and 250 m depth should be removed.
        stations = [
            Station("11", "11", 0, 0, 0, [], None),
            Station("22", "22", 0, 0, 200, [], None),
            Station("22", "22", 0, 0, 250, [], None),
            Station("22", "22", 0, 0, 400, [], None),
            Station("33", "33", 0, 0, 2000, [], None)]
        filtered_stations = filter_stations(stations, 250)
        self.assertEqual(filtered_stations, [
            Station("11", "11", 0, 0, 0, [], None),
            Station("22", "22", 0, 0, 400, [], None),
            Station("33", "33", 0, 0, 2000, [], None)])

        # Set the distance to 1 degree and check the longitude behaviour at
        # the longitude wraparound point.
        stations = [
            Station("11", "11", 0, 0, 0, [], None),
            Station("22", "22", 0, 90, 0, [], None),
            Station("33", "33", 0, 180, 0, [], None),
            Station("44", "44", 0, -90, 0, [], None),
            Station("55", "55", 0, -180, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        # Only 4 stations should remain and either the one at 0,180 or the
        # one at 0, -180 should have been removed as they are equal.
        self.assertEqual(len(filtered_stations), 4)
        self.assertTrue(Station("11", "11", 0, 0, 0, [], None)
                        in filtered_stations)
        self.assertTrue(Station("22", "22", 0, 90, 0, [], None)
                        in filtered_stations)
        self.assertTrue(Station("44", "44", 0, -90, 0, [], None)
                        in filtered_stations)
        self.assertTrue((Station("33", "33", 0, 180, 0, [], None)
                         in filtered_stations) or
                        (Station("55", "55", 0, -180, 0, [], None)
                         in filtered_stations))

        # Test filtering around the longitude wraparound.
        stations = [
            Station("11", "11", 0, 180, 0, [], None),
            Station("22", "22", 0, 179.2, 0, [], None),
            Station("33", "33", 0, 180.8, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(filtered_stations, [
            Station("22", "22", 0, 179.2, 0, [], None),
            Station("33", "33", 0, 180.8, 0, [], None)])

        # Test the conversion of lat/lng to meter distances.
        stations = [
            Station("11", "11", 0, 180, 0, [], None),
            Station("22", "22", 0, -180, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 1)
        stations = [
            Station("11", "11", 0, 180, 0, [], None),
            Station("22", "22", 0, -179.5, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 1)
        stations = [
            Station("11", "11", 0, 180, 0, [], None),
            Station("22", "22", 0, -179.1, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 1)
        stations = [
            Station("11", "11", 0, 180, 0, [], None),
            Station("22", "22", 0, 178.9, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 2)

        # Also test the latitude settings.
        stations = [
            Station("11", "11", 0, -90, 0, [], None),
            Station("22", "22", 0, -90, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 1)
        stations = [
            Station("11", "11", 0, -90, 0, [], None),
            Station("22", "22", 0, -89.5, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 1)
        stations = [
            Station("11", "11", 0, -90, 0, [], None),
            Station("22", "22", 0, -89.1, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 1)
        stations = [
            Station("11", "11", 0, -90, 0, [], None),
            Station("22", "22", 0, -88.9, 0, [], None)]
        filtered_stations = filter_stations(stations, 111000)
        self.assertEqual(len(filtered_stations), 2)

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
        new_list = merge_stations(list_one, list_two, 20)
        self.assertEqual(new_list, list_one)
        new_list = merge_stations(list_one, list_two, 2)
        self.assertEqual(new_list, list_one + list_two)
        new_list = merge_stations(list_one, list_two, 8)
        self.assertEqual(new_list, list_one + [
            Station("11", "11", 0, 0, 10, [], None)])

    def test_filter_channels_based_on_count(self):
        """
        Tests the filter_channels_based_on_count() utilility function.
        """
        # Each location has three channels, except location "20". Therefore
        # all channels from location "20" should be removed
        channels = [
            Channel("", "EHE"),
            Channel("", "EHZ"),
            Channel("", "EHN"),
            Channel("10", "EHE"),
            Channel("10", "EHZ"),
            Channel("10", "EHN"),
            Channel("20", "EHE"),
            Channel("20", "EHZ")]
        self.assertEqual(
            sorted(filter_channels_based_on_count(channels),
                   key=lambda x: x.location + "." + x.channel),
            [Channel("", "EHE"), Channel("", "EHN"), Channel("", "EHZ"),
             Channel("10", "EHE"), Channel("10", "EHN"), Channel("10", "EHZ")]
        )

        # Both locations only once, leave both.
        channels = [Channel("", "EHE"), Channel("10", "EHE")]
        self.assertEqual(
            sorted(filter_channels_based_on_count(channels),
                   key=lambda x: x.location + "." + x.channel),
            [Channel("", "EHE"), Channel("10", "EHE")])

        # Only channels from one location. leave all.
        channels = [Channel("", "EHE"), Channel("", "EHN")]
        self.assertEqual(
            sorted(filter_channels_based_on_count(channels),
                   key=lambda x: x.location + "." + x.channel),
            [Channel("", "EHE"), Channel("", "EHN")])

        # Only one channel.
        channels = [Channel("", "EHE")]
        self.assertEqual(filter_channels_based_on_count(channels),
                         [Channel("", "EHE")])

        # Twice only one channel and one two channels per location. Leave
        # the one with two channels.
        channels = [
            Channel("", "EHE"),
            Channel("", "EHZ"),
            Channel("10", "EHE"),
            Channel("20", "EHE")]
        self.assertEqual(
            sorted(filter_channels_based_on_count(channels),
                   key=lambda x: x.location + "." + x.channel),
            [Channel("", "EHE"), Channel("", "EHZ")])

    def test_stationxml_filename_helper(self):
        """
        Tests the get_stationxml_filename() function.
        """
        # Passing a format string causes it to be used.
        self.assertEqual(
            get_stationxml_filename("{network}_{station}.xml",
                                    network="BW", station="FURT"),
            "BW_FURT.xml")

        # A normal string is considered a path.
        self.assertEqual(
            get_stationxml_filename("FOLDER", network="BW", station="FURT"),
            os.path.join("FOLDER", "BW.FURT.xml"))

        # A passed function will be executed.
        def get_name(network, station):
            return "network" + "__" + station
        self.assertEqual(
            get_stationxml_filename(get_name, network="BW", station="FURT"),
            "network__FURT")

        # It will raise a type error, if the function does not return the
        # proper type.
        self.assertRaises(TypeError, get_stationxml_filename, lambda x: 1,
                          "BW", "FURT")

    def test_mseed_filename_helper(self):
        """
        Tests the get_mseed_filename() function.
        """
        # Passing a format string causes it to be used.
        self.assertEqual(
            get_mseed_filename("{network}_{station}_{location}_{channel}.ms",
                               network="BW", station="FURT", location="",
                               channel="BHE"), "BW_FURT__BHE.ms")

        # A normal string is considered a path.
        self.assertEqual(
            get_mseed_filename("FOLDER", network="BW", station="FURT",
                               location="", channel="BHE"),
            os.path.join("FOLDER", "BW.FURT..BHE.mseed"))

        # A passed function will be executed.
        def get_name(network, station, location, channel):
            return "network" + "__" + station + location + channel
        self.assertEqual(
            get_mseed_filename(get_name, network="BW", station="FURT",
                               location="", channel="BHE"),
            "network__FURTBHE")

        # It will raise a type error, if the function does not return the
        # proper type.
        self.assertRaises(TypeError, get_mseed_filename, lambda x: 1,
                          "BW", "FURT", "", "BHE")


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(unittest.makeSuite(DomainTestCase, 'test'))
    testsuite.addTest(unittest.makeSuite(DownloadHelpersUtilTestCase, 'test'))
    return testsuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
