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
from obspy import readEvents, UTCDateTime, read
from obspy.fdsn import Client
from obspy.fdsn.client import build_url, parse_simple_xml
from obspy.fdsn.header import DEFAULT_USER_AGENT, FDSNException
import os
from StringIO import StringIO
import sys
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
        Tests the automatic discovery of services with the IRIS endpoint. The
        test parameters are taken from IRIS' website.

        This will have to be adjusted once IRIS changes their implementation.
        """
        client = self.client
        self.assertEqual(set(client.services.keys()),
                         set(("dataselect", "event", "station",
                              "available_event_contributors",
                              "available_event_catalogs")))

        # The test sets are copied from the IRIS webpage.
        self.assertEqual(
            set(client.services["dataselect"].keys()),
            set(("starttime", "endtime", "network", "station", "location",
                 "channel", "quality", "minimumlength", "longestonly")))
        self.assertEqual(
            set(client.services["station"].keys()),
            set(("starttime", "endtime", "startbefore", "startafter",
                 "endbefore", "endafter", "network", "station", "location",
                 "channel", "minlatitude", "maxlatitude", "minlongitude",
                 "maxlongitude", "latitude", "longitude", "minradius",
                 "maxradius", "level", "includerestricted",
                 "includeavailability", "updatedafter", "matchtimeseries")))
        self.assertEqual(
            set(client.services["event"].keys()),
            set(("starttime", "endtime", "minlatitude", "maxlatitude",
                 "minlongitude", "maxlongitude", "latitude", "longitude",
                 "maxradius", "minradius", "mindepth", "maxdepth",
                 "minmagnitude", "maxmagnitude",
                 "magtype",  # XXX: Change once fixed.
                 "catalog", "contributor", "limit", "offset", "orderby",
                 "updatedafter", "includeallorigins", "includeallmagnitudes",
                 "includearrivals", "eventid",
                 "originid"  # XXX: This is currently just specified in the
                             #      WADL.
                 )))

        # Also check an exemplary value in more detail.
        minradius = client.services["event"]["minradius"]
        self.assertEqual(minradius["default_value"], 0.0)
        self.assertEqual(minradius["required"], False)
        self.assertEqual(minradius["doc"], "")
        self.assertEqual(minradius["doc_title"], "Specify minimum distance "
                         "from the geographic point defined by latitude and "
                         "longitude")
        self.assertEqual(minradius["type"], float)
        self.assertEqual(minradius["options"], [])

    def test_IRIS_event_catalog_availability(self):
        """
        Tests the parsing of the available event catalogs.
        """
        self.assertEqual(set(self.client.services["available_event_catalogs"]),
                         set(("ANF", "GCMT", "TEST", "ISC", "UofW",
                              "NEIC PDE")))

    def test_IRIS_event_contributors_availability(self):
        """
        Tests the parsing of the available event contributors.
        """
        self.assertEqual(set(
                         self.client.services["available_event_contributors"]),
                         set(("University of Washington", "ANF", "GCMT",
                              "GCMT-Q", "ISC", "NEIC ALERT", "NEIC PDE-W",
                              "UNKNOWN", "NEIC PDE-M", "NEIC PDE-Q")))

    def test_simple_XML_parser(self):
        """
        Tests the simple XML parsing helper function.
        """
        catalogs = parse_simple_xml("""
            <?xml version="1.0"?>
            <Catalogs>
              <total>6</total>
              <Catalog>ANF</Catalog>
              <Catalog>GCMT</Catalog>
              <Catalog>TEST</Catalog>
              <Catalog>ISC</Catalog>
              <Catalog>UofW</Catalog>
              <Catalog>NEIC PDE</Catalog>
            </Catalogs>""")
        self.assertEqual(catalogs, {"catalogs": set(("ANF", "GCMT", "TEST",
                                                     "ISC", "UofW",
                                                     "NEIC PDE"))})

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

        # dataselect example queries
        queries = [
            ("IU", "ANMO", "00", "BHZ",
             UTCDateTime("2010-02-27T06:30:00.000"),
             UTCDateTime("2010-02-27T06:40:00.000")),
            ("IU", "A*", "*", "BHZ",
             UTCDateTime("2010-02-27T06:30:00.000"),
             UTCDateTime("2010-02-27T06:31:00.000")),
            ("IU", "A??", "*0", "BHZ",
             UTCDateTime("2010-02-27T06:30:00.000"),
             UTCDateTime("2010-02-27T06:31:00.000")),
            ]
        result_files = ["dataselect_example.mseed",
                        "dataselect_example_wildcards.mseed",
                        "dataselect_example_mixed_wildcards.mseed",
                        ]
        for query, filename in zip(queries, result_files):
            got = client.get_waveform(*query)
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
            self.assertEqual(got, expected, failmsg(got, expected))

    def test_conflicting_params(self):
        """
        """
        self.assertRaises(FDSNException, self.client.get_stations,
                          network="IU", net="IU")

    def test_help_function_with_IRIS(self):
        """
        Tests the help function with the IRIS example.

        This will have to be adopted any time IRIS changes their
        implementation.
        """
        client = self.client

        # Capture output
        sys.stdout = StringIO()

        client.help("event")
        got = sys.stdout.getvalue()
        expected = (
            "Parameter description for the 'event' service of "
            "'http://service.iris.edu':\n"
            "The service offers the following non-standard parameters:\n"
            "    magtype (str)\n"
            "        type of Magnitude used to test minimum and maximum limits"
            " (case\n"
            "        insensitive)\n"
            "    originid (int)\n"
            "        Retrieve an event based on the unique origin ID numbers "
            "assigned by\n"
            "        the IRIS DMC\n"
            "WARNING: The service does not offer the following standard "
            "parameters: magnitudetype\n"
            "Available catalogs: ANF, UofW, NEIC PDE, ISC, TEST, GCMT\n"
            "Available catalogs: NEIC PDE-W, ANF, University of Washington, "
            "GCMT-Q, NEIC PDE-Q, UNKNOWN, NEIC ALERT, ISC, NEIC PDE-M, GCMT\n")
        self.assertEqual(got, expected, failmsg(got, expected))

        # Reset. Creating a new one is faster then clearing the old one.
        sys.stdout.close()
        sys.stdout = StringIO()

        client.help("station")
        got = sys.stdout.getvalue()
        expected = (
            "Parameter description for the 'station' service of "
            "'http://service.iris.edu':\n"
            "The service offers the following non-standard parameters:\n"
            "    matchtimeseries (bool)\n"
            "        Specify that the availabilities line up with available "
            "data. This is\n"
            "        an IRIS extension to the FDSN specification\n")
        self.assertEqual(got, expected, failmsg(got, expected))

        # Reset.
        sys.stdout.close()
        sys.stdout = StringIO()

        client.help("dataselect")
        got = sys.stdout.getvalue()
        expected = (
            "Parameter description for the 'dataselect' service of "
            "'http://service.iris.edu':\n"
            "No derivations from standard detected\n")
        self.assertEqual(got, expected, failmsg(got, expected))

        sys.stdout.close()
        sys.stdout = sys.__stdout__


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
