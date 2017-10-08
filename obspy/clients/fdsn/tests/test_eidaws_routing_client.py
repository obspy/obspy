#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from distutils.version import LooseVersion
import unittest

import obspy
from obspy.core.compatibility import mock
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn.routing.eidaws_routing_client import \
    EIDAWSRoutingClient


class EIDAWSRoutingClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = EIDAWSRoutingClient()

    def test_get_service_version(self):
        # At the time of test writing the version is 1.1.1. Here we just
        # make sure it is larger.
        self.assertGreaterEqual(
            LooseVersion(self.client.get_service_version()),
            LooseVersion("1.1.1"))

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

http://ws.resif.fr/fdsnws/station/1/query
ND * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
        """.strip()
        # This should return a dictionary that contains the root URL of each
        # fdsn implementation and the POST payload ready to be submitted.
        self.assertEqual(
            EIDAWSRoutingClient.split_routing_response(data),
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
                "http://ws.resif.fr": (
                    "ND * * * 2017-01-01T00:00:00 2017-01-01T00:10:00")})

        data = """
http://geofon.gfz-potsdam.de/fdsnws/station/1/query
NU * * * 2017-01-01T00:00:00 2017-01-01T00:10:00
        """.strip()
        # This should return a dictionary that contains the root URL of each
        # fdsn implementation and the POST payload ready to be submitted.
        self.assertEqual(
            EIDAWSRoutingClient.split_routing_response(data),
            {
                "http://geofon.gfz-potsdam.de": (
                    "NU * * * 2017-01-01T00:00:00 2017-01-01T00:10:00")})

    def test_non_allowed_parameters(self):
        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms(
                "BW", "ALTM", "", "LHZ",
                obspy.UTCDateTime(2017, 1, 1),
                obspy.UTCDateTime(2017, 1, 1, 0, 1),
                filename="out.mseed")
        self.assertEqual(e.exception.args[0],
                         'The `filename` argument is not supported')

        with self.assertRaises(ValueError) as e:
            self.client.get_waveforms(
                "BW", "ALTM", "", "LHZ",
                obspy.UTCDateTime(2017, 1, 1),
                obspy.UTCDateTime(2017, 1, 1, 0, 1),
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
                "XX", "XXXXX", "XX", "XXX",
                obspy.UTCDateTime(2017, 1, 1), obspy.UTCDateTime(2017, 1, 2))
        self.assertEqual(e.exception.args[0],
                         "No data available for request.")


def suite():
    return unittest.makeSuite(EIDAWSRoutingClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
