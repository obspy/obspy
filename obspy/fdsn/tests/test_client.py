# -*- coding: utf-8 -*-
"""
The obspy.fdsn.client test suite.
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



def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
