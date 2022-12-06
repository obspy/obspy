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
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn.routing.eidaws_routing_client import \
    EIDAWSRoutingClient


_DummyResponse = collections.namedtuple("_DummyResponse", ["content"])
pytestmark = pytest.mark.network


class EIDAWSRoutingClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = EIDAWSRoutingClient(timeout=20)
        self._cls = ("obspy.clients.fdsn.routing.eidaws_routing_client."
                     "EIDAWSRoutingClient")

    def test_get_service_version(self):
        # At the time of test writing the version is 1.1.1. Here we just
        # make sure it is larger.
        self.assertGreaterEqual(
            parse_version(self.client.get_service_version()),
            parse_version("1.1.1"))

    def test_response_splitting(self):
        data = """
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
NU * * * 2017-01-01T00:00:00 2017-01-01T00:10:00

http://www.orfeus-eu.org/fdsnws/station/1/query
NS * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
NR * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
NO * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
NL * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
NA * * * 2017-01-01T00:00:00 2017-01-01T00:10:00

http://webservices.ingv.it/fdsnws/station/1/query
NI * * * 2017-01-01T00:00:00 2017-01-01T00:10:00

https://ws.resif.fr/fdsnws/station/1/query
ND * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
        """.strip()
        # This should return a dictionary that contains the root URL of each
        # fdsn implementation and the POST payload ready to be submitted.
        self.assertEqual(
            EIDAWSRoutingClient._split_routing_response(data),
            {
                "http://geofon.gfz-potsdam.de": (
                    "NU * * * 2017-01-01T00:00:00 2017-01-01T00:10:00"),
                "http://www.orfeus-eu.org": (
                    "NS * * * 2017-01-01T00:00:00 2017-01-01T00:10:00\n"
                    "NR * * * 2017-01-01T00:00:00 2017-01-01T00:10:00\n"
                    "NO * * * 2017-01-01T00:00:00 2017-01-01T00:10:00\n"
                    "NL * * * 2017-01-01T00:00:00 2017-01-01T00:10:00\n"
                    "NA * * * 2017-01-01T00:00:00 2017-01-01T00:10:00"),
                "http://webservices.ingv.it": (
                    "NI * * * 2017-01-01T00:00:00 2017-01-01T00:10:00"),
                "https://ws.resif.fr": (
                    "ND * * * 2017-01-01T00:00:00 2017-01-01T00:10:00")})

        data = """
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
NU * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
        """.strip()
        # This should return a dictionary that contains the root URL of each
        # fdsn implementation and the POST payload ready to be submitted.
        self.assertEqual(
            EIDAWSRoutingClient._split_routing_response(data),
            {
                "http://geofon.gfz-potsdam.de": (
                    "NU * * * 2017-01-01T00:00:00 2017-01-01T00:10:00")})

    def test_response_splitting_fdsnws_subdomain(self):
        data = """
http://eida.gein.noa.gr/fdsnws/station/1/
HP LTHK * * 2017-10-20T00:00:00 2599-12-31T23:59:59

http://fdsnws.raspberryshakedata.com/fdsnws/station/1/
AM RA14E * * 2017-10-20T00:00:00 2599-12-31T23:59:59
        """
        self.assertEqual(
            EIDAWSRoutingClient._split_routing_response(data),
            {"http://eida.gein.noa.gr":
                "HP LTHK * * 2017-10-20T00:00:00 2599-12-31T23:59:59",
             "http://fdsnws.raspberryshakedata.com":
                "AM RA14E * * 2017-10-20T00:00:00 2599-12-31T23:59:59"})

    def test_non_allowed_parameters(self):
        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms(
                network="BW", station="ALTM", location="", channel="LHZ",
                starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 1, 0, 1),
                filename="out.mseed")
        self.assertEqual(e.exception.args[0],
                         'The `filename` argument is not supported')

        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms(
                network="BW", station="ALTM", location="", channel="LHZ",
                starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 1, 0, 1),
                attach_response=True)
        self.assertEqual(e.exception.args[0],
                         'The `attach_response` argument is not supported')

        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms_bulk([], filename="out.mseed")
        self.assertEqual(e.exception.args[0],
                         'The `filename` argument is not supported')

        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms_bulk([], attach_response=True)
        self.assertEqual(e.exception.args[0],
                         'The `attach_response` argument is not supported')

        with self.assertRaises(ValueError) as e:
            self.client.get_stations(filename="out.xml")
        self.assertEqual(e.exception.args[0],
                         'The `filename` argument is not supported')

        with self.assertRaises(ValueError) as e:
            self.client.get_stations_bulk([], filename="out.xml")
        self.assertEqual(e.exception.args[0],
                         'The `filename` argument is not supported')

    def test_error_handling(self):
        with self.assertRaises(FDSNNoDataException) as e:
            self.client.get_waveforms(
                network="XX", station="XXXXX", location="XX", channel="XXX",
                starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 2))
        self.assertTrue(e.exception.args[0].startswith(
            "No data available for request."))

    def test_get_waveforms(self):
        """
        This just dispatches to the get_waveforms_bulk() method - so no need
        to also test it explicitly.
        """
        with mock.patch(self._cls + ".get_waveforms_bulk") as p:
            p.return_value = "1234"
            st = self.client.get_waveforms(
                network="XX", station="XXXXX", location="XX", channel="XXX",
                starttime=obspy.UTCDateTime(2017, 1, 1),
                endtime=obspy.UTCDateTime(2017, 1, 2),
                longestonly=True, minimumlength=2)
        self.assertEqual(st, "1234")
        self.assertEqual(p.call_count, 1)
        self.assertEqual(
            p.call_args[0][0][0],
            ["XX", "XXXXX", "XX", "XXX", obspy.UTCDateTime(2017, 1, 1),
             obspy.UTCDateTime(2017, 1, 2)])
        self.assertEqual(p.call_args[1],
                         {"longestonly": True, "minimumlength": 2})

    def test_get_waveforms_bulk(self):
        # Some mock routing response.
        content = """
http://example1.com/fdsnws/station/1/query
AA B1 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00

http://example2.com/fdsnws/station/1/query
AA B2 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00
"""
        if hasattr(content, "encode"):
            content = content.encode()

        with mock.patch(self._cls + "._download") as p1, \
                mock.patch(self._cls + "._download_waveforms") as p2, \
                mock.patch(self._cls + ".get_stations_bulk") as p3:
            p1.return_value = _DummyResponse(content=content)
            p2.return_value = "1234"

            # For the underlying get_stations_bulk() call.
            _dummy_inv = mock.MagicMock()
            _dummy_inv.get_contents.return_value = {
                "channels": ["AA.BB.CC.DD", "AA.BB.CC.DD"]}
            p3.return_value = _dummy_inv

            st = self.client.get_waveforms_bulk(
                [["AA", "B*", "", "DD", obspy.UTCDateTime(2017, 1, 1),
                  obspy.UTCDateTime(2017, 1, 2)]],
                longestonly=True, minimumlength=2)
        self.assertEqual(st, "1234")

        self.assertEqual(p1.call_count, 1)
        self.assertEqual(p1.call_args[0][0],
                         "http://www.orfeus-eu.org/eidaws/routing/1/query")
        # This has been modified by our mocked call to get_stations_bulk().
        self.assertEqual(p1.call_args[1]["data"], (
            b"service=dataselect\nformat=post\n"
            b"AA BB CC DD 2017-01-01T00:00:00.000000 "
            b"2017-01-02T00:00:00.000000"))

        # This is the final call to _download_waveforms() which is again
        # dependent on the dummy response to the _download() function.
        self.assertEqual(p2.call_args[0][0], {
            "http://example1.com":
            "AA B1 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00",
            "http://example2.com":
            "AA B2 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00"})
        self.assertEqual(p2.call_args[1],
                         {"longestonly": True, "minimumlength": 2})

        # Call to this only dependent on the original bulk request.
        self.assertEqual(p3.call_count, 1)
        self.assertEqual(p3.call_args[0][0][0],
                         ["AA", "B*", "--", "DD",
                          str(obspy.UTCDateTime(2017, 1, 1))[:-1],
                          str(obspy.UTCDateTime(2017, 1, 2))[:-1]])
        # Everything should be passed on.
        self.assertEqual(p3.call_args[1], {
            "level": "channel", "longestonly": True, "minimumlength": 2})

    def test_get_stations(self):
        # Some mock routing response.
        content = """
http://example1.com/fdsnws/station/1/query
AA B1 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00

http://example2.com/fdsnws/station/1/query
AA B2 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00
"""
        if hasattr(content, "encode"):
            content = content.encode()

        with mock.patch(self._cls + "._download") as p1, \
                mock.patch(self._cls + "._download_stations") as p2:
            p1.return_value = _DummyResponse(content=content)
            p2.return_value = "1234"

            inv = self.client.get_stations(
                network="AA", channel="DD",
                starttime=obspy.UTCDateTime(2017, 1, 1), latitude=0.0,
                longitude=1.0)
        self.assertEqual(inv, "1234")

        self.assertEqual(p1.call_count, 1)
        self.assertEqual(p1.call_args[0][0],
                         "http://www.orfeus-eu.org/eidaws/routing/1/query")
        # Only a few arguments should be part of the URL.
        self.assertEqual(p1.call_args[1], {
            'content_type': 'text/plain',
            'data': b'service=station\nformat=post\nalternative=false\n'
                    b'AA * * DD 2017-01-01T00:00:00.000000 *'})

        self.assertEqual(p2.call_args[0][0], {
            "http://example1.com":
                "AA B1 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00",
            "http://example2.com":
                "AA B2 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00"})
        # SNCL and times should be filtered out.
        self.assertEqual(p2.call_args[1], {
            "longitude": 1.0, "latitude": 0.0})

    def test_get_stations_bulk(self):
        # Some mock routing response.
        content = """
http://example1.com/fdsnws/station/1/query
AA B1 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00

http://example2.com/fdsnws/station/1/query
AA B2 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00
"""
        if hasattr(content, "encode"):
            content = content.encode()

        with mock.patch(self._cls + "._download") as p1, \
                mock.patch(self._cls + "._download_stations") as p2:
            p1.return_value = _DummyResponse(content=content)
            p2.return_value = "1234"

            inv = self.client.get_stations_bulk(
                [["AA", "B*", "", "DD", obspy.UTCDateTime(2017, 1, 1),
                  obspy.UTCDateTime(2017, 1, 2)]],
                latitude=0.0, longitude=1.0,
                starttime=obspy.UTCDateTime(2017, 1, 1))
        self.assertEqual(inv, "1234")

        self.assertEqual(p1.call_count, 1)
        self.assertEqual(p1.call_args[0][0],
                         "http://www.orfeus-eu.org/eidaws/routing/1/query")
        self.assertEqual(p1.call_args[1]["data"], (
            b"service=station\nformat=post\nalternative=false\n"
            b"AA B* -- DD 2017-01-01T00:00:00.000000 "
            b"2017-01-02T00:00:00.000000"))

        self.assertEqual(p2.call_args[0][0], {
            "http://example1.com":
                "AA B1 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00",
            "http://example2.com":
                "AA B2 -- DD 2017-01-01T00:00:00 2017-01-02T00:10:00"})
        # Only select parameters are passed on.
        self.assertEqual(p2.call_args[1], {
            "longitude": 1.0, "latitude": 0.0,
            "starttime": obspy.UTCDateTime(2017, 1, 1)})

    def test_get_waveforms_integration_test(self):
        """
        Integration test that does not mock anything but actually downloads
        things.
        """
        st = self.client.get_waveforms(
            network="B*", station="*", location="*", channel="HHZ",
            starttime=obspy.UTCDateTime(2017, 1, 1),
            endtime=obspy.UTCDateTime(2017, 1, 1, 0, 0, 0, 1))
        # This yields 4 channels at the time of writing this test - I assume
        # it is unlikely to every yield less than 1.
        # So this test should be fairly stable.
        self.assertGreaterEqual(len(st), 1)

        # Same with the bulk download.
        st2 = self.client.get_waveforms_bulk(
            [["B*", "*", "*", "HHZ", obspy.UTCDateTime(2017, 1, 1),
              obspy.UTCDateTime(2017, 1, 1, 0, 0, 0, 1)]])
        self.assertGreaterEqual(len(st2), 1)

        # They should be identical.
        self.assertEqual(st, st2)

    def test_get_stations_integration_test(self):
        """
        Integration test that does not mock anything but actually downloads
        things.
        """
        inv = self.client.get_stations(
            network="B*", station="*", location="*", channel="LHZ",
            starttime=obspy.UTCDateTime(2017, 1, 1),
            endtime=obspy.UTCDateTime(2017, 1, 1, 0, 1),
            level="network")
        # This yields 1 network at the time of writing this test - I assume
        # it is unlikely to every yield less. So this test should be fairly
        # stable.
        self.assertGreaterEqual(len(inv), 1)

        # Can also be formulated as a bulk query.
        inv2 = self.client.get_stations_bulk(
            [["B*", "*", "*", "LHZ", obspy.UTCDateTime(2017, 1, 1),
              obspy.UTCDateTime(2017, 1, 1, 0, 1)]],
            level="network")
        self.assertGreaterEqual(len(inv2), 1)

        # The results should be basically identical - they will still differ
        # because times stamps and also order might change slightly.
        # But the get_contents() method should be safe enough.
        self.assertEqual(inv.get_contents(), inv2.get_contents())

    def test_proper_no_data_exception_on_out_of_epoch_dates(self):
        """
        Test for #2611 (EIDA/userfeedback#56) which was leading to bulk request
        of *all* EIDA data when querying a legit station but an out-of-epoch
        time window for which no data exists.
        """
        # this time window is before the requested station was installed
        t1 = obspy.UTCDateTime('2012-01-01')
        t2 = t1 + 2
        with self.assertRaises(FDSNNoDataException) as e:
            self.client.get_waveforms(
                network='OE', station='UNNA', channel='HHZ', location='*',
                starttime=t1, endtime=t2)
        self.assertIn('No data', e.exception.args[0])
