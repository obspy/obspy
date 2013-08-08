#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.fdsn.client test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.fdsn import Client
from obspy.fdsn.client import build_url
import os
import unittest


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.fdsn.client.Client.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)

    def test_url_building(self):
        """
        Tests the build_url() functions.
        """
        # Application WADL
        self.assertEqual(
            build_url("http://service.iris.edu", 1, "dataselect",
                      "application.wadl"),
            "http://service.iris.edu/fdsnws/dataselect/1/application.wadl")
        self.assertEqual(
            build_url("http://service.iris.edu", 1, "event",
                      "application.wadl"),
            "http://service.iris.edu/fdsnws/event/1/application.wadl")
        self.assertEqual(
            build_url("http://service.iris.edu", 1, "station",
                      "application.wadl"),
            "http://service.iris.edu/fdsnws/station/1/application.wadl")

        # Some parameters. Only one is tested because the order is random if
        # more than one is given.
        self.assertEqual(
            build_url("http://service.iris.edu", 1, "dataselect",
                      "query", {"network": "BW"}),
            "http://service.iris.edu/fdsnws/dataselect/1/query?network=BW")

        # A wrong resource_type raises a ValueError
        self.assertRaises(ValueError, build_url, "http://service.iris.edu", 1,
                          "obspy", "query")

    def test_service_discovery_iris(self):
        """
        Tests the automatic discovery of services with the IRIS endpoint.
        """
        client = Client(base_url="IRIS")
        from obspy import UTCDateTime
        st = client.get_waveform(UTCDateTime(2012, 1, 1), UTCDateTime(2012, 1, 1, 0, 10), "IU", "ANMO", "10", "BH1")
        print st


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
