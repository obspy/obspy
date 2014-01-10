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
from obspy import readEvents, UTCDateTime, read, read_inventory
from obspy.fdsn import Client
from obspy.fdsn.client import build_url, parse_simple_xml
from obspy.fdsn.header import DEFAULT_USER_AGENT, FDSNException
from obspy.core.util.base import NamedTemporaryFile
from obspy.station import Response
import os
from StringIO import StringIO
import sys
import unittest
from difflib import Differ
import re


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


def normalize_version_number(string):
    """
    Returns imput string with version numbers normalized for testing purposes.
    """
    return re.sub('[0-9]\.[0-9]\.[0-9]', "vX.X.X", string)


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.fdsn.client.Client.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.datapath = os.path.join(self.path, "data")
        self.client = Client(base_url="IRIS", user_agent=USER_AGENT)
        self.client_auth = \
            Client(base_url="IRIS", user_agent=USER_AGENT,
                   user="nobody@iris.edu", password="anonymous")

    def test_url_building(self):
        """
        Tests the build_url() functions.
        """
        # Application WADL
        self.assertEqual(
            build_url("http://service.iris.edu", "dataselect", 1,
                      "application.wadl"),
            "http://service.iris.edu/fdsnws/dataselect/1/application.wadl")
        self.assertEqual(
            build_url("http://service.iris.edu", "event", 1,
                      "application.wadl"),
            "http://service.iris.edu/fdsnws/event/1/application.wadl")
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "application.wadl"),
            "http://service.iris.edu/fdsnws/station/1/application.wadl")

        # Test one parameter.
        self.assertEqual(
            build_url("http://service.iris.edu", "dataselect", 1,
                      "query", {"network": "BW"}),
            "http://service.iris.edu/fdsnws/dataselect/1/query?network=BW")
        self.assertEqual(
            build_url("http://service.iris.edu", "dataselect", 1,
                      "queryauth", {"network": "BW"}),
            "http://service.iris.edu/fdsnws/dataselect/1/queryauth?network=BW")
        # Test two parameters. Note random order, two possible results.
        self.assertTrue(
            build_url("http://service.iris.edu", "dataselect", 1,
                      "query", {"net": "A", "sta": "BC"}) in
            ("http://service.iris.edu/fdsnws/dataselect/1/query?net=A&sta=BC",
             "http://service.iris.edu/fdsnws/dataselect/1/query?sta=BC&net=A"))

        # A wrong service raises a ValueError
        self.assertRaises(ValueError, build_url, "http://service.iris.edu",
                          "obspy", 1, "query")

    def test_location_parameters(self):
        """
        Tests how the variety of location values are handled.

        Why location? Mostly because it is one tricky parameter.  It is not
        uncommon to assume that a non-existant location is "--", but in
        reality "--" is "<space><space>". This substitution exists because
        mostly because various applications have trouble digesting spaces
        (spaces in the URL, for example).
        The confusion begins when location is treated as empty instead, which
        would imply "I want all locations" instead of "I only want locations
        of <space><space>"
        """
        # requests with no specified location should be treated as a wildcard
        self.assertFalse(
            "--" in build_url("http://service.iris.edu", "station", 1,
                              "query", {"network": "IU", "station": "ANMO",
                                        "starttime": "2013-01-01"}))
        # location of "  " is the same as "--"
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "  "}),
            "http://service.iris.edu/fdsnws/station/1/query?location=--")
        # wildcard locations are valid. Will be encoded.
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "*"}),
            "http://service.iris.edu/fdsnws/station/1/query?location=%2A")
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "A?"}),
            "http://service.iris.edu/fdsnws/station/1/query?location=A%3F")

        # lists are valid, including <space><space> lists. Again encoded
        # result.
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "  ,1?,?0"}),
            "http://service.iris.edu/fdsnws/station/1/query?"
            "location=--%2C1%3F%2C%3F0")
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "1?,--,?0"}),
            "http://service.iris.edu/fdsnws/station/1/query?"
            "location=1%3F%2C--%2C%3F0")

        # Test all three special cases with empty parameters into lists.
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "  ,AA,BB"}),
            "http://service.iris.edu/fdsnws/station/1/query?"
            "location=--%2CAA%2CBB")
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "AA,  ,BB"}),
            "http://service.iris.edu/fdsnws/station/1/query?"
            "location=AA%2C--%2CBB")
        self.assertEqual(
            build_url("http://service.iris.edu", "station", 1,
                      "query", {"location": "AA,BB,  "}),
            "http://service.iris.edu/fdsnws/station/1/query?"
            "location=AA%2CBB%2C--")

    def test_url_building_with_auth(self):
        """
        Tests the Client._build_url() method with authentication.

        Necessary on top of test_url_building test case because clients with
        authentication have to build different URLs for dataselect.
        """
        # no authentication
        got = self.client._build_url("dataselect", "query", {'net': "BW"})
        expected = "http://service.iris.edu/fdsnws/dataselect/1/query?net=BW"
        self.assertEqual(got, expected)
        # with authentication
        got = self.client_auth._build_url("dataselect", "query", {'net': "BW"})
        expected = ("http://service.iris.edu/fdsnws/dataselect/1/"
                    "queryauth?net=BW")
        self.assertEqual(got, expected)

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

    def test_IRIS_example_queries_event(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.
        """
        client = self.client

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
            # got.write(file_, "QUAKEML")
            expected = readEvents(file_)
            self.assertEqual(got, expected, failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_events(filename=tf.name, **query)
                with open(tf.name) as fh:
                    got = fh.read()
                with open(file_) as fh:
                    expected = fh.read()
            self.assertEqual(got, expected, failmsg(got, expected))

    def test_IRIS_example_queries_station(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.
        """
        client = self.client

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
            # with open(file_, "wt") as fh:
            #    fh.write(got)
            expected = read_inventory(file_, format="STATIONXML")
            # delete both creating times and modules before comparing objects.
            got.created = None
            expected.created = None
            got.module = None
            expected.module = None

            self.assertEqual(got, expected, failmsg(got, expected))

            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_stations(filename=tf.name, **query)
                with open(tf.name) as fh:
                    got = fh.read()
                with open(file_) as fh:
                    expected = fh.read()
            ignore_lines = ['<Created>', '<TotalNumberStations>', '<Module>']
            msg = failmsg(got, expected, ignore_lines=ignore_lines)
            self.assertEqual(msg, "", msg)

    def test_IRIS_example_queries_dataselect(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.
        """
        client = self.client

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
            # test output to stream
            got = client.get_waveforms(*query)
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
            self.assertEqual(got, expected, failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms(*query, filename=tf.name)
                with open(tf.name) as fh:
                    got = fh.read()
                with open(file_) as fh:
                    expected = fh.read()
            self.assertEqual(got, expected, failmsg(got, expected))

    def test_authentication(self):
        """
        Test dataselect with authentication.
        """
        client = self.client_auth
        # dataselect example queries
        query = ("IU", "ANMO", "00", "BHZ",
                 UTCDateTime("2010-02-27T06:30:00.000"),
                 UTCDateTime("2010-02-27T06:40:00.000"))
        filename = "dataselect_example.mseed"
        got = client.get_waveforms(*query)
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
        try:
            client = self.client
            sys.stdout = StringIO()
            client.help()
            sys.stdout.close()

            # Capture output
            sys.stdout = StringIO()

            client.help("event")
            got = sys.stdout.getvalue()
            expected = (
                "Parameter description for the 'event' service (v1.0.6) of "
                "'http://service.iris.edu':\n"
                "The service offers the following non-standard parameters:\n"
                "    magtype (str)\n"
                "        type of Magnitude used to test minimum and maximum "
                "limits (case\n        insensitive)\n"
                "    originid (int)\n"
                "        Retrieve an event based on the unique origin ID "
                "numbers assigned by\n"
                "        the IRIS DMC\n"
                "WARNING: The service does not offer the following standard "
                "parameters: magnitudetype\n"
                "Available catalogs: ANF, UofW, NEIC PDE, ISC, TEST, GCMT\n"
                "Available contributors: NEIC PDE-W, ANF, University of "
                "Washington, GCMT-Q, NEIC PDE-Q, UNKNOWN, NEIC ALERT, ISC, "
                "NEIC PDE-M, GCMT\n")
            # allow for changes in version number..
            self.assertEqual(normalize_version_number(got),
                             normalize_version_number(expected),
                             failmsg(got, expected))

            # Reset. Creating a new one is faster then clearing the old one.
            sys.stdout.close()
            sys.stdout = StringIO()

            client.help("station")
            got = sys.stdout.getvalue()
            expected = (
                "Parameter description for the 'station' service (v1.0.7) of "
                "'http://service.iris.edu':\n"
                "The service offers the following non-standard parameters:\n"
                "    matchtimeseries (bool)\n"
                "        Specify that the availabilities line up with "
                "available data. This is\n"
                "        an IRIS extension to the FDSN specification\n")
            self.assertEqual(normalize_version_number(got),
                             normalize_version_number(expected),
                             failmsg(got, expected))

            # Reset.
            sys.stdout.close()
            sys.stdout = StringIO()

            client.help("dataselect")
            got = sys.stdout.getvalue()
            expected = (
                "Parameter description for the 'dataselect' service (v1.0.0) "
                "of 'http://service.iris.edu':\n"
                "No derivations from standard detected\n")
            self.assertEqual(normalize_version_number(got),
                             normalize_version_number(expected),
                             failmsg(got, expected))

            sys.stdout.close()
        finally:
            sys.stdout = sys.__stdout__

    def test_str_method(self):
        got = str(self.client)
        expected = (
            "FDSN Webservice Client (base url: http://service.iris.edu)\n"
            "Available Services: 'dataselect' (v1.0.0), 'event' (v1.0.6), "
            "'station' (v1.0.7), 'available_event_contributors', "
            "'available_event_catalogs'\n\n"
            "Use e.g. client.help('dataselect') for the\n"
            "parameter description of the individual services\n"
            "or client.help() for parameter description of\n"
            "all webservices.")
        self.assertEqual(normalize_version_number(got),
                         normalize_version_number(expected),
                         failmsg(got, expected))

    def test_bulk(self):
        """
        Test bulk requests, POSTing data to server. Also tests authenticated
        bulk request.
        """
        clients = [self.client, self.client_auth]
        file1 = os.path.join(self.datapath, "bulk1.mseed")
        file2 = os.path.join(self.datapath, "bulk2.mseed")
        expected1 = read(file1)
        expected2 = read(file2)
        # test cases for providing lists of lists
        bulk1 = (("TA", "A25A", "", "BHZ",
                  UTCDateTime("2010-03-25T00:00:00"),
                  UTCDateTime("2010-03-25T00:00:04")),
                 ("IU", "ANMO", "*", "BH?",
                  UTCDateTime("2010-03-25"),
                  UTCDateTime("2010-03-25T00:00:08")),
                 ("IU", "ANMO", "10", "HHZ",
                  UTCDateTime("2010-05-25T00:00:00"),
                  UTCDateTime("2010-05-25T00:00:04")),
                 ("II", "KURK", "00", "BHN",
                  UTCDateTime("2010-03-25T00:00:00"),
                  UTCDateTime("2010-03-25T00:00:04")))
        bulk2 = (("TA", "A25A", "", "BHZ",
                  UTCDateTime("2010-03-25T00:00:00"),
                  UTCDateTime("2010-03-25T00:00:04")),
                 ("TA", "A25A", "", "BHE",
                  UTCDateTime("2010-03-25T00:00:00"),
                  UTCDateTime("2010-03-25T00:00:06")),
                 ("IU", "ANMO", "*", "HHZ",
                  UTCDateTime("2010-03-25T00:00:00"),
                  UTCDateTime("2010-03-25T00:00:08")))
        params2 = dict(quality="B", longestonly=False, minimumlength=5)
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk1)
            self.assertEqual(got, expected1, failmsg(got, expected1))
            got = client.get_waveforms_bulk(bulk2, **params2)
            self.assertEqual(got, expected2, failmsg(got, expected2))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk1, filename=tf.name)
                got = read(tf.name)
            self.assertEqual(got, expected1, failmsg(got, expected1))
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk2, filename=tf.name, **params2)
                got = read(tf.name)
            self.assertEqual(got, expected2, failmsg(got, expected2))
        # test cases for providing a request string
        bulk1 = ("TA A25A -- BHZ 2010-03-25T00:00:00 2010-03-25T00:00:04\n"
                 "IU ANMO * BH? 2010-03-25 2010-03-25T00:00:08\n"
                 "IU ANMO 10 HHZ 2010-05-25T00:00:00 2010-05-25T00:00:04\n"
                 "II KURK 00 BHN 2010-03-25T00:00:00 2010-03-25T00:00:04\n")
        bulk2 = ("quality=B\n"
                 "longestonly=false\n"
                 "minimumlength=5\n"
                 "TA A25A -- BHZ 2010-03-25T00:00:00 2010-03-25T00:00:04\n"
                 "TA A25A -- BHE 2010-03-25T00:00:00 2010-03-25T00:00:06\n"
                 "IU ANMO * HHZ 2010-03-25T00:00:00 2010-03-25T00:00:08\n")
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk1)
            self.assertEqual(got, expected1, failmsg(got, expected1))
            got = client.get_waveforms_bulk(bulk2)
            self.assertEqual(got, expected2, failmsg(got, expected2))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk1, filename=tf.name)
                got = read(tf.name)
            self.assertEqual(got, expected1, failmsg(got, expected1))
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk2, filename=tf.name)
                got = read(tf.name)
            self.assertEqual(got, expected2, failmsg(got, expected2))
        # test cases for providing a filename
        for client in clients:
            with NamedTemporaryFile() as tf:
                with open(tf.name, "wb") as fh:
                    fh.write(bulk1)
                got = client.get_waveforms_bulk(bulk1)
            self.assertEqual(got, expected1, failmsg(got, expected1))
            with NamedTemporaryFile() as tf:
                with open(tf.name, "wb") as fh:
                    fh.write(bulk2)
                got = client.get_waveforms_bulk(bulk2)
            self.assertEqual(got, expected2, failmsg(got, expected2))
        # test cases for providing a file-like object
        for client in clients:
            got = client.get_waveforms_bulk(StringIO(bulk1))
            self.assertEqual(got, expected1, failmsg(got, expected1))
            got = client.get_waveforms_bulk(StringIO(bulk2))
            self.assertEqual(got, expected2, failmsg(got, expected2))

    def test_get_waveform_attach_response(self):
        """
        minimal test for automatic attaching of metadata
        """
        client = self.client

        bulk = ("TA A25A -- BHZ 2010-03-25T00:00:00 2010-03-25T00:00:04\n"
                "IU ANMO * BH? 2010-03-25 2010-03-25T00:00:08\n"
                "IU ANMO 10 HHZ 2010-05-25T00:00:00 2010-05-25T00:00:04\n"
                "II KURK 00 BHN 2010-03-25T00:00:00 2010-03-25T00:00:04\n")
        st = client.get_waveforms_bulk(bulk, attach_response=True)
        for tr in st:
            self.assertTrue(isinstance(tr.stats.get("response"), Response))

        st = client.get_waveforms("IU", "ANMO", "00", "BHZ",
                                  UTCDateTime("2010-02-27T06:30:00.000"),
                                  UTCDateTime("2010-02-27T06:40:00.000"),
                                  attach_response=True)
        for tr in st:
            self.assertTrue(isinstance(tr.stats.get("response"), Response))


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
