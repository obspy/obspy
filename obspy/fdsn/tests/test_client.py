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
from obspy import readEvents, UTCDateTime
from obspy.fdsn import Client
from obspy.fdsn.client import build_url
from obspy.fdsn.header import DEFAULT_USER_AGENT
import os
import unittest
from difflib import Differ


USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())


def failmsg(got, expected, ignore_lines=[]):
    """
    Create message on difference between objects.

    If both are strings create a line-by-line diff, otherwise create info on
    both using str().
    For diffs, lines that contain any string given in ignore_lines will be
    excluded from the comparison.
    """
    if isinstance(got, str) and isinstance(expected, str):
        got = [l for l in got.splitlines(True)
               if all([x not in l for x in ignore_lines])]
        expected = [l for l in expected.splitlines(True)
                    if all([x not in l for x in ignore_lines])]
        diff = Differ().compare(got, expected)
        diff = "".join([l for l in diff if l[0] in "-+?"])
        if diff:
            return "\nDiff:\n%s" % diff
        else:
            return ""
    else:
        return "\nGot:\n%s\nExpected:\n%s" % (str(got), str(expected))


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.fdsn.client.Client.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.datapath = os.path.join(self.path, "data")
        self.client = Client(base_url="IRIS", user_agent=USER_AGENT)

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
        pass

    def test_IRIS_example_queries(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.
        """
        client = self.client

        # event example queries
        queries = [
            dict(eventid=609301),
            dict(starttime=UTCDateTime("2011-01-07T01:00:00"),
                 endtime=UTCDateTime("2011-01-07T02:00:00"),
                 catalog="NEIC PDE"),
            dict(starttime=UTCDateTime("2011-01-07T14:00:00"),
                 endtime=UTCDateTime("2011-01-08T00:00:00"), minlatitude=15,
                 maxlatitude=40, minlongitude=-170, maxlongitude=170,
                 includeallmagnitudes=True, minmagnitude=4,
                 orderby="magnitude"),
            ]
        result_files = ["events_by_eventid.xml",
                        "events_by_time.xml",
                        "events_by_misc.xml",
                        ]
        for query, filename in zip(queries, result_files):
            got = client.get_events(**query)
            file_ = os.path.join(self.datapath, filename)
            #got.write(file_, "QUAKEML")
            expected = readEvents(file_)
            self.assertEqual(got, expected, failmsg(got, expected))

        # station example queries
        queries = [
            dict(latitude=-56.1, longitude=-26.7, maxradius=15),
            dict(startafter=UTCDateTime("2003-01-07"),
                 endbefore=UTCDateTime("2011-02-07"), minlatitude=15,
                 maxlatitude=55, minlongitude=170, maxlongitude=-170),
            dict(starttime=UTCDateTime("2013-01-01"), network="IU",
                 sta="ANMO", level="channel"),
            dict(starttime=UTCDateTime("2013-01-01"), network="IU", sta="A*",
                 location="00", level="channel", format="text"),
            ]
        result_files = ["stations_by_latlon.xml",
                        "stations_by_misc.xml",
                        "stations_by_station.xml",
                        "stations_by_station_wildcard.xml",
                        ]
        for query, filename in zip(queries, result_files):
            got = client.get_stations(**query)
            file_ = os.path.join(self.datapath, filename)
            #with open(file_, "wt") as fh:
            #    fh.write(got)
            with open(file_) as fh:
                expected = fh.read()
            msg = failmsg(got, expected, ignore_lines=['<Created>'])
            self.assertEqual(msg, "", msg)


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
