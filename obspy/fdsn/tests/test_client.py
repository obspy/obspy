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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import readEvents, UTCDateTime, read, read_inventory
from obspy.fdsn import Client
from obspy.fdsn.client import build_url, parse_simple_xml
from obspy.fdsn.header import DEFAULT_USER_AGENT, FDSNException
from obspy.core.util.base import NamedTemporaryFile
from obspy.core.compatibility import mock
from obspy.station import Response

from difflib import Differ
import io
import os
import re
import sys
import unittest
import warnings


USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())


def failmsg(got, expected, ignore_lines=[]):
    """
    Create message on difference between objects.

    If both are strings create a line-by-line diff, otherwise create info on
    both using str().
    For diffs, lines that contain any string given in ignore_lines will be
    excluded from the comparison.
    """
    if isinstance(got, bytes) and isinstance(expected, bytes):
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

    Due to Py3k arbitrary dictionary ordering it also sorts word wise the
    input string, independent of commas and newlines.
    """
    repl = re.sub('v[0-9]+\.[0-9]+\.[0-9]+', "vX.X.X", string).replace(",", "")
    return " ".join(
        sorted(s.strip() for l in repl.splitlines() for s in l.split(" ")))


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.fdsn.client.Client.
    """
    def __init__(self, *args, **kwargs):
        """
        setupClass() would be better suited for the task at hand but is not
        supported by Python 2.6.
        """
        super(ClientTestCase, self).__init__(*args, **kwargs)

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

        Why location? Mostly because it is one tricky parameter. It is not
        uncommon to assume that a non-existent location is "--", but in reality
        "--" is "<space><space>". This substitution exists because mostly
        because various applications have trouble digesting spaces (spaces in
        the URL, for example).
        The confusion begins when location is treated as empty instead, which
        would imply "I want all locations" instead of "I only want locations of
        <space><space>"
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
                 "magnitudetype",
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
                              "UNKNOWN", "NEIC PDE-M", "NEIC COMCAT",
                              "NEIC PDE-Q")))

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
        Tests the (sometimes modified) example queries given on the IRIS
        web page.
        """
        client = self.client

        queries = [
            dict(eventid=609301),
            dict(starttime=UTCDateTime("2001-01-07T01:00:00"),
                 endtime=UTCDateTime("2001-01-07T01:05:00"),
                 catalog="ISC"),
            dict(starttime=UTCDateTime("2001-01-07T14:00:00"),
                 endtime=UTCDateTime("2001-01-08T00:00:00"), minlatitude=15,
                 maxlatitude=40, minlongitude=-170, maxlongitude=170,
                 includeallmagnitudes=True, minmagnitude=4,
                 orderby="magnitude"),
        ]
        result_files = ["events_by_eventid.xml",
                        "events_by_time.xml",
                        "events_by_misc.xml",
                        ]
        for query, filename in zip(queries, result_files):
            file_ = os.path.join(self.datapath, filename)
            # query["filename"] = file_
            got = client.get_events(**query)
            expected = readEvents(file_)
            self.assertEqual(got, expected, failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_events(filename=tf.name, **query)
                with open(tf.name, 'rb') as fh:
                    got = fh.read()
                with open(file_, 'rb') as fh:
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
            dict(starttime=UTCDateTime("2000-01-01"),
                 endtime=UTCDateTime("2001-01-01"), net="IU",
                 sta="ANMO"),
            dict(starttime=UTCDateTime("2000-01-01"),
                 endtime=UTCDateTime("2002-01-01"), network="IU", sta="A*",
                 location="00"),
        ]
        result_files = ["stations_by_latlon.xml",
                        "stations_by_misc.xml",
                        "stations_by_station.xml",
                        "stations_by_station_wildcard.xml",
                        ]
        for query, filename in zip(queries, result_files):
            file_ = os.path.join(self.datapath, filename)
            # query["filename"] = file_
            got = client.get_stations(**query)
            expected = read_inventory(file_, format="STATIONXML")
            # delete both creating times and modules before comparing objects.
            got.created = None
            expected.created = None
            got.module = None
            expected.module = None

            # XXX Py3k: the objects differ in direct comparison, however,
            # the strings of them are equal
            self.assertEqual(str(got), str(expected), failmsg(got, expected))

            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_stations(filename=tf.name, **query)
                with open(tf.name, 'rb') as fh:
                    got = fh.read()
                with open(file_, 'rb') as fh:
                    expected = fh.read()
            ignore_lines = [b'<Created>', b'<TotalNumberStations>',
                            b'<Module>', b'<ModuleURI>']
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
                with open(tf.name, 'rb') as fh:
                    got = fh.read()
                with open(file_, 'rb') as fh:
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

            # Capture output
            tmp = io.StringIO()
            sys.stdout = tmp

            client.help("event")
            got = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            tmp.close()
            expected = (
                "Parameter description for the 'event' service (v1.1.0) of "
                "'http://service.iris.edu':\n"
                "The service offers the following non-standard parameters:\n"
                "    originid (str)\n"
                "        Retrieve an event based on the unique origin ID "
                "numbers assigned by\n"
                "        the IRIS DMC\n"
                "Available catalogs: ANF, UofW, NEIC PDE, ISC, TEST, GCMT\n"
                "Available contributors: NEIC PDE-W, ANF, University of "
                "Washington, GCMT-Q, NEIC PDE-Q, UNKNOWN, NEIC ALERT, ISC, "
                "NEIC PDE-M, NEIC COMCAT, GCMT\n")
            # allow for changes in version number..
            self.assertEqual(normalize_version_number(got),
                             normalize_version_number(expected),
                             failmsg(normalize_version_number(got),
                                     normalize_version_number(expected)))

            # Reset. Creating a new one is faster then clearing the old one.
            tmp = io.StringIO()
            sys.stdout = tmp

            client.help("station")
            got = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            tmp.close()
            expected = (
                "Parameter description for the 'station' service (v1.1.4) "
                "of 'http://service.iris.edu':\n"
                "No derivations from standard detected")
            self.assertEqual(normalize_version_number(got),
                             normalize_version_number(expected),
                             failmsg(normalize_version_number(got),
                                     normalize_version_number(expected)))

            # Reset.
            tmp = io.StringIO()
            sys.stdout = tmp

            client.help("dataselect")
            got = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            tmp.close()
            expected = (
                "Parameter description for the 'dataselect' service (v1.0.0) "
                "of 'http://service.iris.edu':\n"
                "No derivations from standard detected\n")
            self.assertEqual(normalize_version_number(got),
                             normalize_version_number(expected),
                             failmsg(normalize_version_number(got),
                                     normalize_version_number(expected)))

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
                         failmsg(normalize_version_number(got),
                                 normalize_version_number(expected)))

    def test_dataselect_bulk(self):
        """
        Test bulk dataselect requests, POSTing data to server. Also tests
        authenticated bulk request.
        """
        clients = [self.client, self.client_auth]
        file = os.path.join(self.datapath, "bulk.mseed")
        expected = read(file)
        # test cases for providing lists of lists
        bulk = (("TA", "A25A", "", "BHZ",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:04")),
                ("TA", "A25A", "", "BHE",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:06")),
                ("IU", "ANMO", "*", "HHZ",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:08")))
        params = dict(quality="B", longestonly=False, minimumlength=5)
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk, **params)
            self.assertEqual(got, expected, failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk, filename=tf.name, **params)
                got = read(tf.name)
            self.assertEqual(got, expected, failmsg(got, expected))
        # test cases for providing a request string
        bulk = ("quality=B\n"
                "longestonly=false\n"
                "minimumlength=5\n"
                "TA A25A -- BHZ 2010-03-25T00:00:00 2010-03-25T00:00:04\n"
                "TA A25A -- BHE 2010-03-25T00:00:00 2010-03-25T00:00:06\n"
                "IU ANMO * HHZ 2010-03-25T00:00:00 2010-03-25T00:00:08\n")
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk)
            self.assertEqual(got, expected, failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk, filename=tf.name)
                got = read(tf.name)
            self.assertEqual(got, expected, failmsg(got, expected))
        # test cases for providing a file name
        for client in clients:
            with NamedTemporaryFile() as tf:
                with open(tf.name, "wt") as fh:
                    fh.write(bulk)
                got = client.get_waveforms_bulk(bulk)
            self.assertEqual(got, expected, failmsg(got, expected))
        # test cases for providing a file-like object
        for client in clients:
            got = client.get_waveforms_bulk(io.StringIO(bulk))
            self.assertEqual(got, expected, failmsg(got, expected))

    def test_station_bulk(self):
        """
        Test bulk station requests, POSTing data to server. Also tests
        authenticated bulk request.

        Does currently only test reading from a list of list. The other
        input types are tested with the waveform bulk downloader and thus
        should work just fine.
        """
        clients = [self.client, self.client_auth]
        # test cases for providing lists of lists
        starttime = UTCDateTime(1990, 1, 1)
        endtime = UTCDateTime(1990, 1, 1) + 10
        bulk = [
            ["IU", "ANMO", "", "BHE", starttime, endtime],
            ["IU", "CCM", "", "BHZ", starttime, endtime],
            ["IU", "COR", "", "UHZ", starttime, endtime],
            ["IU", "HRV", "", "LHN", starttime, endtime],
        ]
        for client in clients:
            # Test with station level.
            inv = client.get_stations_bulk(bulk, level="station")
            # Test with output to file.
            with NamedTemporaryFile() as tf:
                client.get_stations_bulk(
                    bulk, filename=tf.name, level="station")
                inv2 = read_inventory(tf.name, format="stationxml")

            self.assertEqual(inv.networks, inv2.networks)
            self.assertEqual(len(inv.networks), 1)
            self.assertEqual(inv[0].code, "IU")
            self.assertEqual(len(inv.networks[0].stations), 4)
            self.assertEqual(
                sorted([_i.code for _i in inv.networks[0].stations]),
                sorted(["ANMO", "CCM", "COR", "HRV"]))

            # Test with channel level.
            inv = client.get_stations_bulk(bulk, level="channel")
            # Test with output to file.
            with NamedTemporaryFile() as tf:
                client.get_stations_bulk(
                    bulk, filename=tf.name, level="channel")
                inv2 = read_inventory(tf.name, format="stationxml")

            self.assertEqual(inv.networks, inv2.networks)
            self.assertEqual(len(inv.networks), 1)
            self.assertEqual(inv[0].code, "IU")
            self.assertEqual(len(inv.networks[0].stations), 4)
            self.assertEqual(
                sorted([_i.code for _i in inv.networks[0].stations]),
                sorted(["ANMO", "CCM", "COR", "HRV"]))
            channels = []
            for station in inv[0]:
                for channel in station:
                    channels.append("IU.%s.%s.%s" % (
                        station.code, channel.location_code,
                        channel.code))
            self.assertEqual(
                sorted(channels),
                sorted(["IU.ANMO..BHE", "IU.CCM..BHZ", "IU.COR..UHZ",
                        "IU.HRV..LHN"]))
        return

    def test_get_waveform_attach_response(self):
        """
        minimal test for automatic attaching of metadata
        """
        client = self.client

        bulk = ("IU ANMO 00 BHZ 2000-03-25T00:00:00 2000-03-25T00:00:04\n")
        st = client.get_waveforms_bulk(bulk, attach_response=True)
        for tr in st:
            self.assertTrue(isinstance(tr.stats.get("response"), Response))

        st = client.get_waveforms("IU", "ANMO", "00", "BHZ",
                                  UTCDateTime("2000-02-27T06:00:00.000"),
                                  UTCDateTime("2000-02-27T06:00:05.000"),
                                  attach_response=True)
        for tr in st:
            self.assertTrue(isinstance(tr.stats.get("response"), Response))

    @mock.patch("obspy.fdsn.client.download_url")
    def test_default_requested_urls(self, download_url_mock):
        """
        Five request should be sent upon initializing a client. Test these.
        """
        download_url_mock.return_value = (404, None)
        base_url = "http://example.com"

        # An exception will be raised if not actual WADLs are returned.
        try:
            Client(base_url=base_url)
        except FDSNException:
            pass

        expected_urls = sorted([
            "%s/fdsnws/event/1/contributors" % base_url,
            "%s/fdsnws/event/1/catalogs" % base_url,
            "%s/fdsnws/event/1/application.wadl" % base_url,
            "%s/fdsnws/station/1/application.wadl" % base_url,
            "%s/fdsnws/dataselect/1/application.wadl" % base_url,
        ])
        got_urls = sorted([_i[0][0] for _i in
                           download_url_mock.call_args_list])

        self.assertEqual(expected_urls, got_urls)

    @mock.patch("obspy.fdsn.client.download_url")
    def test_setting_service_major_version(self, download_url_mock):
        """
        Test the setting of custom major versions.
        """
        download_url_mock.return_value = (404, None)
        base_url = "http://example.com"

        # Passing an empty dictionary results in the default urls.
        major_versions = {}
        # An exception will be raised if not actual WADLs are returned.
        try:
            Client(base_url=base_url, major_versions=major_versions)
        except FDSNException:
            pass
        expected_urls = sorted([
            "%s/fdsnws/event/1/contributors" % base_url,
            "%s/fdsnws/event/1/catalogs" % base_url,
            "%s/fdsnws/event/1/application.wadl" % base_url,
            "%s/fdsnws/station/1/application.wadl" % base_url,
            "%s/fdsnws/dataselect/1/application.wadl" % base_url,
        ])
        got_urls = sorted([_i[0][0] for _i in
                           download_url_mock.call_args_list])
        self.assertEqual(expected_urls, got_urls)

        # Replace all
        download_url_mock.reset_mock()
        download_url_mock.return_value = (404, None)
        major_versions = {"event": 7, "station": 8, "dataselect": 9}
        # An exception will be raised if not actual WADLs are returned.
        try:
            Client(base_url=base_url, major_versions=major_versions)
        except FDSNException:
            pass
        expected_urls = sorted([
            "%s/fdsnws/event/7/contributors" % base_url,
            "%s/fdsnws/event/7/catalogs" % base_url,
            "%s/fdsnws/event/7/application.wadl" % base_url,
            "%s/fdsnws/station/8/application.wadl" % base_url,
            "%s/fdsnws/dataselect/9/application.wadl" % base_url,
        ])
        got_urls = sorted([_i[0][0] for _i in
                           download_url_mock.call_args_list])
        self.assertEqual(expected_urls, got_urls)

        # Replace only some
        download_url_mock.reset_mock()
        download_url_mock.return_value = (404, None)
        major_versions = {"event": 7, "station": 8}
        # An exception will be raised if not actual WADLs are returned.
        try:
            Client(base_url=base_url, major_versions=major_versions)
        except FDSNException:
            pass
        expected_urls = sorted([
            "%s/fdsnws/event/7/contributors" % base_url,
            "%s/fdsnws/event/7/catalogs" % base_url,
            "%s/fdsnws/event/7/application.wadl" % base_url,
            "%s/fdsnws/station/8/application.wadl" % base_url,
            "%s/fdsnws/dataselect/1/application.wadl" % base_url,
        ])
        got_urls = sorted([_i[0][0] for _i in
                           download_url_mock.call_args_list])
        self.assertEqual(expected_urls, got_urls)

    @mock.patch("obspy.fdsn.client.download_url")
    def test_setting_service_provider_mappings(self, download_url_mock):
        """
        Tests the setting of per service endpoints
        """
        base_url = "http://example.com"

        # Replace all.
        download_url_mock.return_value = (404, None)
        # Some custom urls
        base_url_event = "http://other_url.com/beta/event_service/11"
        base_url_station = "http://some_url.com/beta2/stat_serv/7"
        base_url_ds = "http://new.com/beta3/waveforms/8"
        # An exception will be raised if not actual WADLs are returned.
        try:
            Client(base_url=base_url, service_mappings={
                "event": base_url_event,
                "station": base_url_station,
                "dataselect": base_url_ds,
            })
        except FDSNException:
            pass
        expected_urls = sorted([
            "%s/contributors" % base_url_event,
            "%s/catalogs" % base_url_event,
            "%s/application.wadl" % base_url_event,
            "%s/application.wadl" % base_url_station,
            "%s/application.wadl" % base_url_ds,
        ])
        got_urls = sorted([_i[0][0] for _i in
                           download_url_mock.call_args_list])
        self.assertEqual(expected_urls, got_urls)

        # Replace only two. The others keep the default mapping.
        download_url_mock.reset_mock()
        download_url_mock.return_value = (404, None)
        # Some custom urls
        base_url_station = "http://some_url.com/beta2/stat_serv/7"
        base_url_ds = "http://new.com/beta3/waveforms/8"
        # An exception will be raised if not actual WADLs are returned.
        try:
            Client(base_url=base_url, service_mappings={
                "station": base_url_station,
                "dataselect": base_url_ds,
            })
        except FDSNException:
            pass
        expected_urls = sorted([
            "%s/fdsnws/event/1/contributors" % base_url,
            "%s/fdsnws/event/1/catalogs" % base_url,
            "%s/fdsnws/event/1/application.wadl" % base_url,
            "%s/application.wadl" % base_url_station,
            "%s/application.wadl" % base_url_ds,
        ])
        got_urls = sorted([_i[0][0] for _i in
                           download_url_mock.call_args_list])
        self.assertEqual(expected_urls, got_urls)

    def test_manually_deactivate_single_service(self):
        """
        Test manually deactivating a single service.
        """
        client = Client(base_url="IRIS", user_agent=USER_AGENT,
                        service_mappings={"event": None})
        self.assertEqual(sorted(client.services.keys()),
                         ['dataselect', 'station'])

    @mock.patch("obspy.fdsn.client.download_url")
    def test_download_urls_for_custom_mapping(self, download_url_mock):
        """
        Tests the downloading of data with custom mappings.
        """
        base_url = "http://example.com"

        # More extensive mock setup simulation service discovery.
        def custom_side_effects(*args, **kwargs):
            if "version" in args[0]:
                return 200, "1.0.200"
            elif "event" in args[0]:
                with open(os.path.join(
                        self.datapath, "2014-01-07_iris_event.wadl"),
                        "rb") as fh:
                    return 200, fh.read()
            elif "station" in args[0]:
                with open(os.path.join(
                        self.datapath,
                        "2014-01-07_iris_station.wadl"), "rb") as fh:
                    return 200, fh.read()
            elif "dataselect" in args[0]:
                with open(os.path.join(
                        self.datapath,
                        "2014-01-07_iris_dataselect.wadl"), "rb") as fh:
                    return 200, fh.read()
            return 404, None

        download_url_mock.side_effect = custom_side_effects

        # Some custom urls
        base_url_event = "http://other_url.com/beta/event_service/11"
        base_url_station = "http://some_url.com/beta2/station/7"
        base_url_ds = "http://new.com/beta3/dataselect/8"

        # An exception will be raised if not actual WADLs are returned.
        # Catch warnings to avoid them being raised for the tests.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = Client(base_url=base_url, service_mappings={
                "event": base_url_event,
                "station": base_url_station,
                "dataselect": base_url_ds,
            })
        for warning in w:
            self.assertTrue("Could not parse" in str(warning) or
                            "cannot deal with" in str(warning))

        # Test the dataselect downloading.
        download_url_mock.reset_mock()
        download_url_mock.side_effect = None
        download_url_mock.return_value = 404, None
        try:
            c.get_waveforms("A", "B", "C", "D", UTCDateTime() - 100,
                            UTCDateTime())
        except:
            pass
        self.assertTrue(
            base_url_ds in download_url_mock.call_args_list[0][0][0])

        # Test the station downloading.
        download_url_mock.reset_mock()
        download_url_mock.side_effect = None
        download_url_mock.return_value = 404, None
        try:
            c.get_stations()
        except:
            pass
        self.assertTrue(
            base_url_station in download_url_mock.call_args_list[0][0][0])

        # Test the event downloading.
        download_url_mock.reset_mock()
        download_url_mock.side_effect = None
        download_url_mock.return_value = 404, None
        try:
            c.get_events()
        except:
            pass
        self.assertTrue(
            base_url_event in download_url_mock.call_args_list[0][0][0])


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
