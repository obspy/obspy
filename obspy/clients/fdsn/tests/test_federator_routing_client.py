#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections
import unittest
from unittest import mock

from packaging.version import parse as parse_version
import pytest

import obspy
from obspy.clients.fdsn.routing.federator_routing_client import \
    FederatorRoutingClient


_DummyResponse = collections.namedtuple("_DummyResponse", ["content"])
pytestmark = pytest.mark.network


class FederatorRoutingClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = FederatorRoutingClient()
        self._cls = ("obspy.clients.fdsn.routing.federator_routing_client."
                     "FederatorRoutingClient")

    def test_get_service_version(self):
        # At the time of test writing the version is 1.1.1. Here we just
        # make sure it is larger.
        self.assertGreaterEqual(
            parse_version(self.client.get_service_version()),
            parse_version("1.1.1"))

    def test_response_splitting(self):
        data = """
RANDOM_KEY=true

DATACENTER=GEOFON,http://geofon.gfz-potsdam.de
DATASELECTSERVICE=http://geofon.gfz-potsdam1.de/fdsnws/dataselect/1/
STATIONSERVICE=http://geofon.gfz-potsdam2.de/fdsnws/station/1/
AF CER -- BHE 2007-03-15T00:47:00 2599-12-31T23:59:59
AF CER -- BHN 2007-03-15T00:47:00 2599-12-31T23:59:59


DATACENTER=INGV,http://www.ingv.it
DATASELECTSERVICE=http://webservices1.rm.ingv.it/fdsnws/dataselect/1/
STATIONSERVICE=http://webservices2.rm.ingv.it/fdsnws/station/1/
EVENTSERVICE=http://webservices.rm.ingv.it/fdsnws/event/1/
AC PUK -- HHE 2009-05-29T00:00:00 2009-12-22T00:00:00
        """
        self.assertEqual(
            FederatorRoutingClient._split_routing_response(data, "dataselect"),
            {"http://geofon.gfz-potsdam1.de": (
                "AF CER -- BHE 2007-03-15T00:47:00 2599-12-31T23:59:59\n"
                "AF CER -- BHN 2007-03-15T00:47:00 2599-12-31T23:59:59"),
             "http://webservices1.rm.ingv.it": (
                "AC PUK -- HHE 2009-05-29T00:00:00 2009-12-22T00:00:00")})
        self.assertEqual(
            FederatorRoutingClient._split_routing_response(data, "station"),
            {"http://geofon.gfz-potsdam2.de": (
                "AF CER -- BHE 2007-03-15T00:47:00 2599-12-31T23:59:59\n"
                "AF CER -- BHN 2007-03-15T00:47:00 2599-12-31T23:59:59"),
                "http://webservices2.rm.ingv.it": (
                    "AC PUK -- HHE 2009-05-29T00:00:00 2009-12-22T00:00:00")})

        # Error handling.
        with self.assertRaises(ValueError) as e:
            FederatorRoutingClient._split_routing_response(data, "random")
        self.assertEqual(e.exception.args[0],
                         "Service must be 'dataselect' or 'station'.")

    def test_response_splitting_fdsnws_subdomain(self):
        data = """
DATACENTER=NOA,http://bbnet.gein.noa.gr/HL/
DATASELECTSERVICE=http://eida.gein.noa.gr/fdsnws/dataselect/1/
STATIONSERVICE=http://eida.gein.noa.gr/fdsnws/station/1/
HP LTHK * * 2017-10-20T00:00:00 2599-12-31T23:59:59

DATACENTER=RASPISHAKE,http://raspberryshake.net/
DATASELECTSERVICE=http://fdsnws.raspberryshakedata.com/fdsnws/dataselect/1/
STATIONSERVICE=http://fdsnws.raspberryshakedata.com/fdsnws/station/1/
EVENTSERVICE=http://fdsnws.raspberryshakedata.com/fdsnws/event/1/
AM RA14E * * 2017-10-20T00:00:00 2599-12-31T23:59:59
        """
        self.assertEqual(
            FederatorRoutingClient._split_routing_response(data, "station"),
            {"http://eida.gein.noa.gr":
                "HP LTHK * * 2017-10-20T00:00:00 2599-12-31T23:59:59",
             "http://fdsnws.raspberryshakedata.com":
                 "AM RA14E * * 2017-10-20T00:00:00 2599-12-31T23:59:59"})

    def test_get_waveforms(self):
        """
        This just dispatches to the get_waveforms_bulk() method - so no need
        to also test it explicitly.
        """
        with mock.patch(self._cls + ".get_waveforms_bulk") as p:
            p.return_value = "1234"
            st = self.client.get_waveforms(
                network="XX", station="XXXXX", location="XX",
                channel="XXX", starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 2),
                latitude=1.0, longitude=2.0,
                longestonly=True, minimumlength=2)
        self.assertEqual(st, "1234")
        self.assertEqual(p.call_count, 1)
        self.assertEqual(
            p.call_args[0][0][0],
            ["XX", "XXXXX", "XX", "XXX", obspy.UTCDateTime(2017, 1, 1),
             obspy.UTCDateTime(2017, 1, 2)])
        # SNCLs + times should be filtered out.
        self.assertEqual(p.call_args[1],
                         {"longestonly": True,
                          "minimumlength": 2, "latitude": 1.0,
                          "longitude": 2.0})

        # Don't pass in the SNCLs.
        with mock.patch(self._cls + ".get_waveforms_bulk") as p:
            p.return_value = "1234"
            st = self.client.get_waveforms(
                starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 2),
                latitude=1.0, longitude=2.0,
                longestonly=True, minimumlength=2)
        self.assertEqual(st, "1234")
        self.assertEqual(p.call_count, 1)
        self.assertEqual(
            p.call_args[0][0][0],
            ["*", "*", "*", "*", obspy.UTCDateTime(2017, 1, 1),
             obspy.UTCDateTime(2017, 1, 2)])
        self.assertEqual(p.call_args[1],
                         {"longestonly": True,
                          "minimumlength": 2, "latitude": 1.0,
                          "longitude": 2.0})

    def test_get_waveforms_bulk(self):
        # Some mock routing response.
        content = """
DATACENTER=GEOFON,http://geofon.gfz-potsdam.de
DATASELECTSERVICE=http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/
STATIONSERVICE=http://geofon.gfz-potsdam.de/fdsnws/station/1/
AF CER -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00
AF CVNA -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00

DATACENTER=IRISDMC,http://ds.iris.edu
DATASELECTSERVICE=http://service.iris.edu/fdsnws/dataselect/1/
STATIONSERVICE=http://service.iris.edu/fdsnws/station/1/
EVENTSERVICE=http://service.iris.edu/fdsnws/event/1/
SACPZSERVICE=http://service.iris.edu/irisws/sacpz/1/
RESPSERVICE=http://service.iris.edu/irisws/resp/1/
AF CNG -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00
AK CAPN -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00
        """
        if hasattr(content, "encode"):
            content = content.encode()

        with mock.patch(self._cls + "._download") as p1, \
                mock.patch(self._cls + "._download_waveforms") as p2:
            p1.return_value = _DummyResponse(content=content)
            p2.return_value = "1234"

            st = self.client.get_waveforms_bulk(
                [["A*", "C*", "", "LHZ", obspy.UTCDateTime(2017, 1, 1),
                  obspy.UTCDateTime(2017, 1, 2)]],
                longestonly=True, minimumlength=2)
        self.assertEqual(st, "1234")

        self.assertEqual(p1.call_count, 1)
        self.assertEqual(p1.call_args[0][0],
                         "http://service.iris.edu/irisws/fedcatalog/1/query")
        self.assertEqual(p1.call_args[1]["data"], (
            b"format=request\n"
            b"A* C* -- LHZ 2017-01-01T00:00:00.000000 "
            b"2017-01-02T00:00:00.000000"))

        self.assertEqual(p2.call_args[0][0], {
            "http://geofon.gfz-potsdam.de": (
                "AF CER -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00\n"
                "AF CVNA -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00"),
            "http://service.iris.edu": (
                "AF CNG -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00\n"
                "AK CAPN -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00")})
        self.assertEqual(p2.call_args[1],
                         {"longestonly": True, "minimumlength": 2})

    def test_get_waveforms_error_handling(self):
        # Some parameters should not be passed explicitly.
        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms_bulk([[
                "AA", "BB", "", "LHZ", obspy.UTCDateTime(2016, 1, 1),
                obspy.UTCDateTime(2016, 1, 2)]], network="BB")
        self.assertEqual(
            e.exception.args[0],
            "`network` must not be part of the optional parameters in a bulk "
            "request.")

    def test_get_stations(self):
        """
        This just dispatches to the get_waveforms_bulk() method - so no need
        to also test it explicitly.
        """
        with mock.patch(self._cls + ".get_stations_bulk") as p:
            p.return_value = "1234"
            st = self.client.get_stations(
                network="XX", station="XXXXX", location="XX",
                channel="XXX", starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 2),
                latitude=1.0, longitude=2.0,
                maximumradius=1.0, level="network")
        self.assertEqual(st, "1234")
        self.assertEqual(p.call_count, 1)
        self.assertEqual(
            p.call_args[0][0][0],
            ["XX", "XXXXX", "XX", "XXX", obspy.UTCDateTime(2017, 1, 1),
             obspy.UTCDateTime(2017, 1, 2)])
        # SNCLs + times should be filtered out.
        self.assertEqual(p.call_args[1],
                         {"latitude": 1.0, "longitude": 2.0,
                          "maximumradius": 1.0, "level": "network"})

        # Don't pass in the SNCLs.
        with mock.patch(self._cls + ".get_stations_bulk") as p:
            p.return_value = "1234"
            st = self.client.get_stations(
                starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 2),
                latitude=1.0, longitude=2.0,
                maximumradius=1.0, level="network")
        self.assertEqual(st, "1234")
        self.assertEqual(p.call_count, 1)
        self.assertEqual(
            p.call_args[0][0][0],
            ["*", "*", "*", "*", obspy.UTCDateTime(2017, 1, 1),
             obspy.UTCDateTime(2017, 1, 2)])
        self.assertEqual(p.call_args[1],
                         {"latitude": 1.0, "longitude": 2.0,
                          "maximumradius": 1.0, "level": "network"})

    def test_get_stations_bulk(self):
        # Some mock routing response.
        content = """
DATACENTER=GEOFON,http://geofon.gfz-potsdam.de
DATASELECTSERVICE=http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/
STATIONSERVICE=http://geofon.gfz-potsdam.de/fdsnws/station/1/
AF CER -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00
AF CVNA -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00

DATACENTER=IRISDMC,http://ds.iris.edu
DATASELECTSERVICE=http://service.iris.edu/fdsnws/dataselect/1/
STATIONSERVICE=http://service.iris.edu/fdsnws/station/1/
EVENTSERVICE=http://service.iris.edu/fdsnws/event/1/
SACPZSERVICE=http://service.iris.edu/irisws/sacpz/1/
RESPSERVICE=http://service.iris.edu/irisws/resp/1/
AF CNG -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00
AK CAPN -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00
        """
        if hasattr(content, "encode"):
            content = content.encode()

        with mock.patch(self._cls + "._download") as p1, \
                mock.patch(self._cls + "._download_stations") as p2:
            p1.return_value = _DummyResponse(content=content)
            p2.return_value = "1234"

            st = self.client.get_stations_bulk(
                [["A*", "C*", "", "LHZ", obspy.UTCDateTime(2017, 1, 1),
                  obspy.UTCDateTime(2017, 1, 2)]],
                level="network")
        self.assertEqual(st, "1234")

        self.assertEqual(p1.call_count, 1)
        self.assertEqual(p1.call_args[0][0],
                         "http://service.iris.edu/irisws/fedcatalog/1/query")
        self.assertEqual(p1.call_args[1]["data"], (
            b"level=network\n"
            b"format=request\n"
            b"A* C* -- LHZ 2017-01-01T00:00:00.000000 "
            b"2017-01-02T00:00:00.000000"))

        self.assertEqual(p2.call_args[0][0], {
            "http://geofon.gfz-potsdam.de": (
                "AF CER -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00\n"
                "AF CVNA -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00"),
            "http://service.iris.edu": (
                "AF CNG -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00\n"
                "AK CAPN -- LHZ 2017-01-01T00:00:00 2017-01-02T00:00:00")})
        self.assertEqual(p2.call_args[1],
                         {"level": "network"})

    def test_get_stations_error_handling(self):
        # Some parameters should not be passed explicitly.
        with self.assertRaises(ValueError) as e:
            self.client.get_stations_bulk([[
                "AA", "BB", "", "LHZ", obspy.UTCDateTime(2016, 1, 1),
                obspy.UTCDateTime(2016, 1, 2)]], network="BB")
        self.assertEqual(
            e.exception.args[0],
            "`network` must not be part of the optional parameters in a bulk "
            "request.")

    def test_get_waveforms_integration_test(self):
        """
        Integration test that does not mock anything but actually downloads
        things.
        """
        st = self.client.get_waveforms(
            starttime=obspy.UTCDateTime(2017, 1, 1),
            endtime=obspy.UTCDateTime(2017, 1, 1, 0, 1),
            latitude=35.0, longitude=-110, maxradius=0.3,
            channel="LHZ")
        # This yields 1 channel at the time of writing this test - I assume
        # it is unlikely to every yield less. So this test should be fairly
        # stable.
        self.assertGreaterEqual(len(st), 1)

        # Same with the bulk request.
        st2 = self.client.get_waveforms_bulk(
            [["*", "*", "*", "LHZ", obspy.UTCDateTime(2017, 1, 1),
              obspy.UTCDateTime(2017, 1, 1, 0, 1)]],
            latitude=35.0, longitude=-110, maxradius=0.3)
        self.assertGreaterEqual(len(st2), 1)

        self.assertEqual(st, st2)

    def test_get_stations_integration_test(self):
        """
        Integration test that does not mock anything but actually downloads
        things.
        """
        inv = self.client.get_stations(
            starttime=obspy.UTCDateTime(2017, 1, 1),
            endtime=obspy.UTCDateTime(2017, 1, 1, 0, 1),
            latitude=35.0, longitude=-110, maxradius=0.3, network="TA",
            channel="LHZ", level="station")
        # This yields 1 network at the time of writing this test - I assume
        # it is unlikely to every yield less. So this test should be fairly
        # stable.
        self.assertGreaterEqual(len(inv), 1)

        # Again repeat with the bulk request.
        inv2 = self.client.get_stations_bulk(
            [["*", "*", "*", "LHZ", obspy.UTCDateTime(2017, 1, 1),
              obspy.UTCDateTime(2017, 1, 1, 0, 1)]],
            latitude=35.0, longitude=-110, maxradius=0.3,
            level="station")
        self.assertGreaterEqual(len(inv2), 1)
        inv2 = inv2.select(network="TA")

        # The results should be basically identical - they will still differ
        # because times stamps and also order might change slightly.
        # But the get_contents() method should be safe enough.
        self.assertEqual(inv.get_contents(), inv2.get_contents())
