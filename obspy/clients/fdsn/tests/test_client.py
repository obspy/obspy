#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.fdsn.client test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
import os
import re
import sys
import unittest
import warnings
from difflib import Differ
from unittest import mock

import urllib.request as urllib_request

import lxml
import numpy as np
import requests

from obspy import UTCDateTime, read, read_inventory, Stream, Trace
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients.fdsn import Client, RoutingClient
from obspy.clients.fdsn.client import (build_url, parse_simple_xml,
                                       get_bulk_string)
from obspy.clients.fdsn.header import (DEFAULT_USER_AGENT, URL_MAPPINGS,
                                       FDSNException, FDSNRedirectException,
                                       FDSNNoDataException,
                                       FDSNRequestTooLargeException,
                                       FDSNBadRequestException,
                                       FDSNNoAuthenticationServiceException,
                                       FDSNTimeoutException,
                                       FDSNNoServiceException,
                                       FDSNInternalServerException,
                                       FDSNTooManyRequestsException,
                                       FDSNServiceUnavailableException,
                                       FDSNUnauthorizedException,
                                       FDSNForbiddenException,
                                       FDSNDoubleAuthenticationException,
                                       FDSNInvalidRequestException,
                                       DEFAULT_SERVICES)
from obspy.core.inventory import Response
from obspy.geodetics import locations2degrees


USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())


def _normalize_stats(obj):
    if isinstance(obj, Stream):
        for tr in obj:
            _normalize_stats(tr)
    else:
        if "processing" in obj.stats:
            del obj.stats["processing"]
        if "_fdsnws_dataselect_url" in obj.stats:
            del obj.stats._fdsnws_dataselect_url


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

    Due to Py3k arbitrary dictionary ordering it also sorts word wise the
    input string, independent of commas and newlines.
    """
    match = r'v[0-9]+\.[0-9]+\.[0-9]+'
    repl = re.sub(match, "vX.X.X", string).replace(",", "")
    return [l.strip() for l in repl.splitlines()]


class ClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.fdsn.client.Client.
    """
    @classmethod
    def setUpClass(cls):
        # directory where the test files are located
        cls.path = os.path.dirname(__file__)
        cls.datapath = os.path.join(cls.path, "data")
        cls.client = Client(base_url="IRIS", user_agent=USER_AGENT)
        cls.client_auth = \
            Client(base_url="IRIS", user_agent=USER_AGENT,
                   user="nobody@iris.edu", password="anonymous")

    def test_empty_bulk_string(self):
        """
        Makes sure an exception is raised if an empty bulk string would be
        produced (e.g. empty list as input for `get_bulk_string()`)
        """
        msg = ("Empty 'bulk' parameter potentially leading to a FDSN request "
               "of all available data")
        for bad_input in [[], '', None]:
            with self.assertRaises(FDSNInvalidRequestException) as e:
                get_bulk_string(bulk=bad_input, arguments={})
            self.assertEqual(e.exception.args[0], msg)

    def test_validate_base_url(self):
        """
        Tests the _validate_base_url() method.
        """

        test_urls_valid = list(URL_MAPPINGS.values())
        test_urls_valid += [
            "http://arclink.ethz.ch",
            "http://example.org",
            "https://webservices.rm.ingv.it",
            "http://localhost:8080/test/",
            "http://93.63.40.85/",
            "http://[::1]:80/test/",
            "http://[2001:db8:85a3:8d3:1319:8a2e:370:7348]",
            "http://[2001:db8::ff00:42:8329]",
            "http://[::ffff:192.168.89.9]",
            "http://jane",
            "http://localhost"]

        test_urls_fails = [
            "http://",
            "http://127.0.1",
            "http://127.=.0.1",
            "http://127.0.0.0.1"]
        test_urls_fails += [
            "http://[]",
            "http://[1]",
            "http://[1:2]",
            "http://[1::2::3]",
            "http://[1::2:3::4]",
            "http://[1:2:2:4:5:6:7]"]

        for url in test_urls_valid:
            self.assertEqual(
                self.client._validate_base_url(url),
                True)

        for url in test_urls_fails:
            self.assertEqual(
                self.client._validate_base_url(url),
                False)

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

        # The location parameter is also passed through the
        # _create_url_from_parameters() method and thus has to survive it!
        # This guards against a regression where all empty location codes
        # where removed by this function!
        for service in ["station", "dataselect"]:
            for loc in ["", " ", "  ", "--", b"", b" ", b"  ", b"--",
                        u"", u" ", u"  ", u"--"]:
                self.assertIn(
                    "location=--",
                    self.client._create_url_from_parameters(
                        service, [],
                        {"location": loc, "starttime": 0, "endtime": 1}))

        # Also check the full call with a mock test.
        for loc in ["", " ", "  ", "--", b"", b" ", b"  ", b"--",
                    u"", u" ", u"  ", u"--"]:
            with mock.patch("obspy.clients.fdsn.Client._download") as p:
                self.client.get_stations(0, 0, location=loc,
                                         filename=mock.Mock())
            self.assertEqual(p.call_count, 1)
            self.assertIn("location=--", p.call_args[0][0])
            with mock.patch("obspy.clients.fdsn.Client._download") as p:
                self.client.get_waveforms(1, 2, loc, 4, 0, 0,
                                          filename=mock.Mock())
            self.assertEqual(p.call_count, 1)
            self.assertIn("location=--", p.call_args[0][0])

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

    def test_set_credentials(self):
        """
        Test for issue #2146

        When setting credentials not during `__init__` but using
        `set_credentials`, waveform queries should still properly go to
        "queryauth" endpoint.
        """
        client = Client(base_url="IRIS", user_agent=USER_AGENT)
        user = "nobody@iris.edu"
        password = "anonymous"
        client.set_credentials(user=user, password=password)
        got = client._build_url("dataselect", "query", {'net': "BW"})
        expected = ("http://service.iris.edu/fdsnws/dataselect/1/"
                    "queryauth?net=BW")
        self.assertEqual(got, expected)
        # more basic test: check that set_credentials has set Client.user
        # (which is tested when checking which endpoint to use, query or
        # queryauth)
        self.assertEqual(client.user, user)

    def test_trim_stream_after_get_waveform(self):
        """
        Tests that stream is properly trimmed to user requested times after
        fetching from datacenter, see #1887
        """
        c = Client(
            service_mappings={'dataselect':
                              'http://ws.ipgp.fr/fdsnws/dataselect/1'})
        starttime = UTCDateTime('2016-11-01T00:00:00')
        endtime = UTCDateTime('2016-11-01T00:00:10')
        stream = c.get_waveforms('G', 'PEL', '*', 'LHZ', starttime, endtime)
        self.assertEqual(starttime, stream[0].stats.starttime)
        self.assertEqual(endtime, stream[0].stats.endtime)

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
                 "maxradius", "level", "includerestricted", "format",
                 "includeavailability", "updatedafter", "matchtimeseries")))
        self.assertEqual(
            set(client.services["event"].keys()),
            set(("starttime", "endtime", "minlatitude", "maxlatitude",
                 "minlongitude", "maxlongitude", "latitude", "longitude",
                 "maxradius", "minradius", "mindepth", "maxdepth",
                 "minmagnitude", "maxmagnitude",
                 "magnitudetype", "format",
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

    def test_iris_event_catalog_availability(self):
        """
        Tests the parsing of the available event catalogs.
        """
        self.assertEqual(set(self.client.services["available_event_catalogs"]),
                         set(("GCMT", "ISC", "NEIC PDE")))

    def test_iris_event_contributors_availability(self):
        """
        Tests the parsing of the available event contributors.
        """
        response = requests.get(
            'http://service.iris.edu/fdsnws/event/1/contributors')
        xml = lxml.etree.fromstring(response.content)
        expected = {
            elem.text for elem in xml.xpath('/Contributors/Contributor')}
        # check that we have some values in there
        self.assertTrue(len(expected) > 5)
        self.assertEqual(
            set(self.client.services["available_event_contributors"]),
            expected)

    def test_discover_services_defaults(self):
        """
        A Client initialized with _discover_services=False shouldn't perform
        any services/WADL queries on the endpoint, and should show only the
        default service parameters.
        """
        client = Client(base_url="IRIS", user_agent=USER_AGENT,
                        _discover_services=False)
        self.assertEqual(client.services, DEFAULT_SERVICES)

    def test_simple_xml_parser(self):
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

    def test_iris_example_queries_event(self):
        """
        Tests the (sometimes modified) example queries given on the IRIS
        web page.

        Used to be tested against files but that was not maintainable. It
        now tests if the queries return what was asked for.
        """
        client = self.client

        # Event id query.
        cat = client.get_events(eventid=609301)
        self.assertEqual(len(cat), 1)
        self.assertIn("609301", cat[0].resource_id.id)

        # Temporal query.
        cat = client.get_events(
            starttime=UTCDateTime("2001-01-07T01:00:00"),
            endtime=UTCDateTime("2001-01-07T01:05:00"), catalog="ISC")
        self.assertGreater(len(cat), 0)
        for event in cat:
            self.assertEqual(event.origins[0].extra.catalog.value, "ISC")
            self.assertGreater(event.origins[0].time,
                               UTCDateTime("2001-01-07T01:00:00"))
            self.assertGreater(UTCDateTime("2001-01-07T01:05:00"),
                               event.origins[0].time)

        # Misc query.
        cat = client.get_events(
            starttime=UTCDateTime("2001-01-07T14:00:00"),
            endtime=UTCDateTime("2001-01-08T00:00:00"), minlatitude=15,
            maxlatitude=40, minlongitude=-170, maxlongitude=170,
            includeallmagnitudes=True, minmagnitude=4, orderby="magnitude")
        self.assertGreater(len(cat), 0)
        for event in cat:
            self.assertGreater(event.origins[0].time,
                               UTCDateTime("2001-01-07T14:00:00"))
            self.assertGreater(UTCDateTime("2001-01-08T00:00:00"),
                               event.origins[0].time)
            self.assertGreater(event.origins[0].latitude, 14.9)
            self.assertGreater(40.1, event.origins[0].latitude)
            self.assertGreater(event.origins[0].latitude, -170.1)
            self.assertGreater(170.1, event.origins[0].latitude)
            # events returned by FDSNWS can contain many magnitudes with a wide
            # range, and currently (at least for IRIS) the magnitude threshold
            # sent to the server checks if at least one magnitude matches, it
            # does not only check the preferred magnitude..
            self.assertTrue(any(m.mag >= 3.999 for m in event.magnitudes))

    def test_irisph5_event(self):
        """
        Tests the IRISPH5 URL mapping, which is special due to its custom
        subpath.
        """
        client = Client('IRISPH5')

        # Event id query.
        cat = client.get_events(catalog='8A')
        self.assertEqual(len(cat), 19)
        self.assertEqual(cat[0].event_type, 'controlled explosion')

    def test_iris_example_queries_station(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.

        This test used to download files but that is almost impossible to
        keep up to date - thus it is now a bit smarter and tests the
        returned inventory in different ways.
        """
        client = self.client

        # Radial query.
        inv = client.get_stations(latitude=-56.1, longitude=-26.7,
                                  maxradius=15)
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                dist = locations2degrees(sta.latitude, sta.longitude,
                                         -56.1, -26.7)
                # small tolerance for WGS84.
                self.assertGreater(15.1, dist, "%s.%s" % (net.code,
                                                          sta.code))

        # Misc query.
        inv = client.get_stations(
            startafter=UTCDateTime("2003-01-07"),
            endbefore=UTCDateTime("2011-02-07"), minlatitude=15,
            maxlatitude=55, minlongitude=170, maxlongitude=-170, network="IM")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                msg = "%s.%s" % (net.code, sta.code)
                self.assertGreater(sta.start_date, UTCDateTime("2003-01-07"),
                                   msg)
                if sta.end_date is not None:
                    self.assertGreater(UTCDateTime("2011-02-07"), sta.end_date,
                                       msg)
                self.assertGreater(sta.latitude, 14.9, msg)
                self.assertGreater(55.1, sta.latitude, msg)
                self.assertFalse(-170.1 <= sta.longitude <= 170.1, msg)
                self.assertEqual(net.code, "IM", msg)

        # Simple query
        inv = client.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2001-01-01"), net="IU", sta="ANMO")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                self.assertGreater(UTCDateTime("2001-01-01"), sta.start_date)
                if sta.end_date is not None:
                    self.assertGreater(sta.end_date, UTCDateTime("2000-01-01"))
                self.assertEqual(net.code, "IU")
                self.assertEqual(sta.code, "ANMO")

        # Station wildcard query.
        inv = client.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2002-01-01"), network="IU", sta="A*",
            location="00")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                self.assertGreater(UTCDateTime("2002-01-01"), sta.start_date)
                if sta.end_date is not None:
                    self.assertGreater(sta.end_date, UTCDateTime("2000-01-01"))
                self.assertEqual(net.code, "IU")
                self.assertTrue(sta.code.startswith("A"))

    def test_iris_example_queries_dataselect(self):
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
            # Assert that the meta-information about the provider is stored.
            for tr in got:
                self.assertEqual(
                    tr.stats._fdsnws_dataselect_url,
                    client.base_url + "/fdsnws/dataselect/1/query")
            # Remove fdsnws URL as it is not in the data from the disc.
            for tr in got:
                del tr.stats._fdsnws_dataselect_url
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
            # The client trims by default.
            _normalize_stats(got)
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms(*query, filename=tf.name)
                with open(tf.name, 'rb') as fh:
                    got = fh.read()
                with open(file_, 'rb') as fh:
                    expected = fh.read()
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))

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
        _normalize_stats(got)
        self.assertEqual(got, expected, failmsg(got, expected))

    def test_iris_example_queries_event_discover_services_false(self):
        """
        Tests the (sometimes modified) example queries given on the IRIS
        web page, without service discovery.

        Used to be tested against files but that was not maintainable. It
        now tests if the queries return what was asked for.
        """
        client = Client(base_url="IRIS", user_agent=USER_AGENT,
                        _discover_services=False)

        # Event id query.
        cat = client.get_events(eventid=609301)
        self.assertEqual(len(cat), 1)
        self.assertIn("609301", cat[0].resource_id.id)

        # Temporal query.
        cat = client.get_events(
            starttime=UTCDateTime("2001-01-07T01:00:00"),
            endtime=UTCDateTime("2001-01-07T01:05:00"), catalog="ISC")
        self.assertGreater(len(cat), 0)
        for event in cat:
            self.assertEqual(event.origins[0].extra.catalog.value, "ISC")
            self.assertGreater(event.origins[0].time,
                               UTCDateTime("2001-01-07T01:00:00"))
            self.assertGreater(UTCDateTime("2001-01-07T01:05:00"),
                               event.origins[0].time)

        # Misc query.
        cat = client.get_events(
            starttime=UTCDateTime("2001-01-07T14:00:00"),
            endtime=UTCDateTime("2001-01-08T00:00:00"), minlatitude=15,
            maxlatitude=40, minlongitude=-170, maxlongitude=170,
            includeallmagnitudes=True, minmagnitude=4, orderby="magnitude")
        self.assertGreater(len(cat), 0)
        for event in cat:
            self.assertGreater(event.origins[0].time,
                               UTCDateTime("2001-01-07T14:00:00"))
            self.assertGreater(UTCDateTime("2001-01-08T00:00:00"),
                               event.origins[0].time)
            self.assertGreater(event.origins[0].latitude, 14.9)
            self.assertGreater(40.1, event.origins[0].latitude)
            self.assertGreater(event.origins[0].latitude, -170.1)
            self.assertGreater(170.1, event.origins[0].latitude)
            # events returned by FDSNWS can contain many magnitudes with a wide
            # range, and currently (at least for IRIS) the magnitude threshold
            # sent to the server checks if at least one magnitude matches, it
            # does not only check the preferred magnitude..
            self.assertTrue(any(m.mag >= 3.999 for m in event.magnitudes))

    def test_iris_example_queries_station_discover_services_false(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage,
        without service discovery.

        This test used to download files but that is almost impossible to
        keep up to date - thus it is now a bit smarter and tests the
        returned inventory in different ways.
        """
        client = Client(base_url="IRIS", user_agent=USER_AGENT,
                        _discover_services=False)

        # Radial query.
        inv = client.get_stations(latitude=-56.1, longitude=-26.7,
                                  maxradius=15)
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                dist = locations2degrees(sta.latitude, sta.longitude,
                                         -56.1, -26.7)
                # small tolerance for WGS84.
                self.assertGreater(15.1, dist, "%s.%s" % (net.code,
                                                          sta.code))

        # Misc query.
        inv = client.get_stations(
            startafter=UTCDateTime("2003-01-07"),
            endbefore=UTCDateTime("2011-02-07"), minlatitude=15,
            maxlatitude=55, minlongitude=170, maxlongitude=-170, network="IM")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                msg = "%s.%s" % (net.code, sta.code)
                self.assertGreater(sta.start_date, UTCDateTime("2003-01-07"),
                                   msg)
                if sta.end_date is not None:
                    self.assertGreater(UTCDateTime("2011-02-07"), sta.end_date,
                                       msg)
                self.assertGreater(sta.latitude, 14.9, msg)
                self.assertGreater(55.1, sta.latitude, msg)
                self.assertFalse(-170.1 <= sta.longitude <= 170.1, msg)
                self.assertEqual(net.code, "IM", msg)

        # Simple query
        inv = client.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2001-01-01"), net="IU", sta="ANMO")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                self.assertGreater(UTCDateTime("2001-01-01"), sta.start_date)
                if sta.end_date is not None:
                    self.assertGreater(sta.end_date, UTCDateTime("2000-01-01"))
                self.assertEqual(net.code, "IU")
                self.assertEqual(sta.code, "ANMO")

        # Station wildcard query.
        inv = client.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2002-01-01"), network="IU", sta="A*",
            location="00")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                self.assertGreater(UTCDateTime("2002-01-01"), sta.start_date)
                if sta.end_date is not None:
                    self.assertGreater(sta.end_date, UTCDateTime("2000-01-01"))
                self.assertEqual(net.code, "IU")
                self.assertTrue(sta.code.startswith("A"))

    def test_iris_example_queries_dataselect_discover_services_false(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage,
        without discovering services first.
        """
        client = Client(base_url="IRIS", user_agent=USER_AGENT,
                        _discover_services=False)

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
            # Assert that the meta-information about the provider is stored.
            for tr in got:
                self.assertEqual(
                    tr.stats._fdsnws_dataselect_url,
                    client.base_url + "/fdsnws/dataselect/1/query")
            # Remove fdsnws URL as it is not in the data from the disc.
            for tr in got:
                del tr.stats._fdsnws_dataselect_url
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
            _normalize_stats(got)
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms(*query, filename=tf.name)
                with open(tf.name, 'rb') as fh:
                    got = fh.read()
                with open(file_, 'rb') as fh:
                    expected = fh.read()
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))

    def test_conflicting_params(self):
        """
        """
        self.assertRaises(FDSNInvalidRequestException,
                          self.client.get_stations,
                          network="IU", net="IU")

    def test_help_function_with_iris(self):
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
            filename = "event_helpstring.txt"
            with open(os.path.join(self.datapath, filename)) as fh:
                expected = fh.read()
            # allow for changes in version number..
            got = normalize_version_number(got)
            expected = normalize_version_number(expected)
            # catalogs/contributors are checked in separate tests
            self.assertTrue(got[-2].startswith('Available catalogs:'))
            self.assertTrue(got[-1].startswith('Available contributors:'))
            got = got[:-2]
            expected = expected[:-2]
            for line_got, line_expected in zip(got, expected):
                self.assertEqual(line_got, line_expected)

            # Reset. Creating a new one is faster then clearing the old one.
            tmp = io.StringIO()
            sys.stdout = tmp

            client.help("station")
            got = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            tmp.close()

            filename = "station_helpstring.txt"
            with open(os.path.join(self.datapath, filename)) as fh:
                expected = fh.read()
            got = normalize_version_number(got)
            expected = normalize_version_number(expected)
            self.assertEqual(got, expected, failmsg(got, expected))

            # Reset.
            tmp = io.StringIO()
            sys.stdout = tmp

            client.help("dataselect")
            got = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            tmp.close()

            filename = "dataselect_helpstring.txt"
            with open(os.path.join(self.datapath, filename)) as fh:
                expected = fh.read()
            got = normalize_version_number(got)
            expected = normalize_version_number(expected)
            self.assertEqual(got, expected, failmsg(got, expected))

        finally:
            sys.stdout = sys.__stdout__

    def test_str_method(self):
        got = str(self.client)
        expected = (
            "FDSN Webservice Client (base url: http://service.iris.edu)\n"
            "Available Services: 'dataselect' (v1.0.0), 'event' (v1.0.6), "
            "'station' (v1.0.7), 'available_event_catalogs', "
            "'available_event_contributors'\n\n"
            "Use e.g. client.help('dataselect') for the\n"
            "parameter description of the individual services\n"
            "or client.help() for parameter description of\n"
            "all webservices.")
        got = normalize_version_number(got)
        expected = normalize_version_number(expected)
        self.assertEqual(got, expected, failmsg(got, expected))

    def test_dataselect_bulk(self):
        """
        Test bulk dataselect requests, POSTing data to server. Also tests
        authenticated bulk request.
        """
        clients = [self.client, self.client_auth]
        file = os.path.join(self.datapath, "bulk.mseed")
        expected = read(file)
        # test cases for providing lists of lists
        # Deliberately requesting data that overlap the end-time of a channel.
        # TA.A25A..BHZ ends at 2011-07-22T14:50:25.5
        bulk = (("TA", "A25A", "", "BHZ",
                 UTCDateTime("2011-07-22T14:50:23"),
                 UTCDateTime("2011-07-22T14:50:29")),
                ("TA", "A25A", "", "BHE",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:06")),
                ("IU", "ANMO", "*", "HHZ",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:08")))
        # As of 03 December 2018, it looks like IRIS is ignoring minimumlength?
        params = dict(quality="B", longestonly=False, minimumlength=5)
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk, **params)
            # Remove fdsnws URL as it is not in the data from the disc.
            for tr in got:
                del tr.stats._fdsnws_dataselect_url
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
            # Assert that the meta-information about the provider is stored.
            for tr in got:
                if client.user:
                    self.assertEqual(
                        tr.stats._fdsnws_dataselect_url,
                        client.base_url + "/fdsnws/dataselect/1/queryauth")
                else:
                    self.assertEqual(
                        tr.stats._fdsnws_dataselect_url,
                        client.base_url + "/fdsnws/dataselect/1/query")
            # Remove fdsnws URL as it is not in the data from the disc.
            for tr in got:
                del tr.stats._fdsnws_dataselect_url
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
            # Remove fdsnws URL as it is not in the data from the disc.
            for tr in got:
                del tr.stats._fdsnws_dataselect_url
            self.assertEqual(got, expected, failmsg(got, expected))
        # test cases for providing a file-like object
        for client in clients:
            got = client.get_waveforms_bulk(io.StringIO(bulk))
            # Remove fdsnws URL as it is not in the data from the disc.
            for tr in got:
                del tr.stats._fdsnws_dataselect_url
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

    @mock.patch("obspy.clients.fdsn.client.download_url")
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

    @mock.patch("obspy.clients.fdsn.client.download_url")
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

    @mock.patch("obspy.clients.fdsn.client.download_url")
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

    @mock.patch("obspy.clients.fdsn.client.download_url")
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
        base_url_event = "http://example.com/beta/event_service/11"
        base_url_station = "http://example.org/beta2/station/7"
        base_url_ds = "http://example.edu/beta3/dataselect/8"

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
        except Exception:
            pass
        self.assertTrue(
            base_url_ds in download_url_mock.call_args_list[0][0][0])

        # Test the station downloading.
        download_url_mock.reset_mock()
        download_url_mock.side_effect = None
        download_url_mock.return_value = 404, None
        try:
            c.get_stations()
        except Exception:
            pass
        self.assertTrue(
            base_url_station in download_url_mock.call_args_list[0][0][0])

        # Test the event downloading.
        download_url_mock.reset_mock()
        download_url_mock.side_effect = None
        download_url_mock.return_value = 404, None
        try:
            c.get_events()
        except Exception:
            pass
        self.assertTrue(
            base_url_event in download_url_mock.call_args_list[0][0][0])

    def test_redirection(self):
        """
        Tests the redirection of GET and POST requests. We redirect
        everything if not authentication is used.

        IRIS runs three services to test it:
            http://ds.iris.edu/files/redirect/307/station/1
            http://ds.iris.edu/files/redirect/307/dataselect/1
            http://ds.iris.edu/files/redirect/307/event/1
        """
        c = Client("IRIS", service_mappings={
            "station":
                "http://ds.iris.edu/files/redirect/307/station/1",
            "dataselect":
                "http://ds.iris.edu/files/redirect/307/dataselect/1",
            "event":
                "http://ds.iris.edu/files/redirect/307/event/1"},
            user_agent=USER_AGENT)

        st = c.get_waveforms(
            network="IU", station="ANMO", location="00", channel="BHZ",
            starttime=UTCDateTime("2010-02-27T06:30:00.000"),
            endtime=UTCDateTime("2010-02-27T06:30:01.000"))
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(st)))

        inv = c.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2001-01-01"),
            network="IU", station="ANMO", level="network")
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(inv.networks)))

        cat = c.get_events(starttime=UTCDateTime("2001-01-07T01:00:00"),
                           endtime=UTCDateTime("2001-01-07T01:05:00"),
                           catalog="ISC")
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(cat)))

        # Also test the bulk requests which are done using POST requests.
        bulk = (("TA", "A25A", "", "BHZ",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:01")),
                ("TA", "A25A", "", "BHE",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:01")))
        st = c.get_waveforms_bulk(bulk, quality="B", longestonly=False)
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(st)))

        starttime = UTCDateTime(1990, 1, 1)
        endtime = UTCDateTime(1990, 1, 1) + 10
        bulk = [
            ["IU", "ANMO", "", "BHE", starttime, endtime],
            ["IU", "CCM", "", "BHZ", starttime, endtime],
        ]
        inv = c.get_stations_bulk(bulk, level="network")
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(inv.networks)))

    def test_redirection_auth(self):
        """
        Tests the redirection of GET and POST requests using authentication.

        By default these should not redirect and an exception is raised.
        """
        # Clear the cache.
        Client._Client__service_discovery_cache.clear()

        # The error will already be raised during the initialization in most
        # cases.
        self.assertRaises(
            FDSNRedirectException,
            Client, "IRIS", service_mappings={
                "station": "http://ds.iris.edu/files/redirect/307/station/1",
                "dataselect":
                    "http://ds.iris.edu/files/redirect/307/dataselect/1",
                "event": "http://ds.iris.edu/files/redirect/307/event/1"},
            user="nobody@iris.edu", password="anonymous",
            user_agent=USER_AGENT)

        # The force_redirect flag overwrites that behaviour.
        c_auth = Client("IRIS", service_mappings={
            "station":
                "http://ds.iris.edu/files/redirect/307/station/1",
            "dataselect":
                "http://ds.iris.edu/files/redirect/307/dataselect/1",
            "event":
                "http://ds.iris.edu/files/redirect/307/event/1"},
            user="nobody@iris.edu", password="anonymous",
            user_agent=USER_AGENT, force_redirect=True)

        st = c_auth.get_waveforms(
            network="IU", station="ANMO", location="00", channel="BHZ",
            starttime=UTCDateTime("2010-02-27T06:30:00.000"),
            endtime=UTCDateTime("2010-02-27T06:30:01.000"))
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(st)))

        inv = c_auth.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2001-01-01"),
            network="IU", station="ANMO", level="network")
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(inv.networks)))

        cat = c_auth.get_events(starttime=UTCDateTime("2001-01-07T01:00:00"),
                                endtime=UTCDateTime("2001-01-07T01:05:00"),
                                catalog="ISC")
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(cat)))

        # Also test the bulk requests which are done using POST requests.
        bulk = (("TA", "A25A", "", "BHZ",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:01")),
                ("TA", "A25A", "", "BHE",
                 UTCDateTime("2010-03-25T00:00:00"),
                 UTCDateTime("2010-03-25T00:00:01")))
        st = c_auth.get_waveforms_bulk(bulk, quality="B", longestonly=False)
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(st)))

        starttime = UTCDateTime(1990, 1, 1)
        endtime = UTCDateTime(1990, 1, 1) + 10
        bulk = [
            ["IU", "ANMO", "", "BHE", starttime, endtime],
            ["IU", "CCM", "", "BHZ", starttime, endtime],
        ]
        inv = c_auth.get_stations_bulk(bulk, level="network")
        # Just make sure something is being downloaded.
        self.assertTrue(bool(len(inv.networks)))

    def test_get_waveforms_empty_seed_codes(self):
        """
        Make sure that network, station, and channel codes specified as empty
        strings are not omitted in `get_waveforms(...)` when building the url
        (which results in default values '*' (wildcards) at the server,
        see #1578).
        """
        t = UTCDateTime(2000, 1, 1)
        url_base = "http://service.iris.edu/fdsnws/dataselect/1/query?"
        kwargs = dict(network='IU', station='ANMO', location='00',
                      channel='HHZ', starttime=t, endtime=t)

        for key in ('network', 'station', 'channel'):
            kwargs_ = kwargs.copy()
            # set empty SEED code for given key
            kwargs_.update(((key, ''),))

            # use a mock object and check what URL would have been downloaded
            with mock.patch.object(
                    self.client, '_download') as m:
                try:
                    self.client.get_waveforms(**kwargs_)
                except Exception:
                    # Mocking returns something different.
                    continue
                # URL downloading comes before the error and can be checked now
                url = m.call_args[0][0]
            url_parts = url.replace(url_base, '').split("&")
            self.assertIn('{}='.format(key), url_parts)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_no_data_exception(self, download_url_mock):
        """
        Verify that a request returning no data raises an identifiable
        exception
        """
        download_url_mock.return_value = (204, None)
        self.assertRaises(FDSNNoDataException, self.client.get_stations)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_request_too_large_exception(self, download_url_mock):
        """
        Verify that a request returning too much data raises an identifiable
        exception
        """
        download_url_mock.return_value = (413, None)
        self.assertRaises(FDSNRequestTooLargeException,
                          self.client.get_stations)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_authentication_exceptions(self, download_url_mock):
        """
        Verify that a request with missing authentication raises an
        identifiable exception
        """
        with mock.patch("obspy.clients.fdsn.client.Client._has_eida_auth",
                        new_callable=mock.PropertyMock,
                        return_value=False):
            self.assertRaises(FDSNNoAuthenticationServiceException, Client,
                              eida_token="TEST")

        self.assertRaises(FDSNDoubleAuthenticationException, Client, "IRIS",
                          eida_token="TEST", user="TEST")

        download_url_mock.return_value = (401, None)
        self.assertRaises(FDSNUnauthorizedException,
                          self.client.get_stations)

        download_url_mock.return_value = (403, None)
        self.assertRaises(FDSNForbiddenException,
                          self.client.get_stations)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_timeout_exception(self, download_url_mock):
        """
        Verify that a request timing out raises an identifiable exception
        """
        download_url_mock.return_value = (None, "timeout")
        self.assertRaises(FDSNTimeoutException, self.client.get_stations)

    def test_no_service_exception(self):
        """
        Verify that opening a client to a provider without FDSN service raises
        an identifiable exception
        """
        self.assertRaises(FDSNNoServiceException, Client,
                          "http://nofdsnservice.org")

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_service_unavailable_exception(self, download_url_mock):
        """
        Verify that opening a client to a service temporarily unavailable
        raises an identifiable exception
        """
        download_url_mock.return_value = (503, None)
        self.assertRaises(FDSNServiceUnavailableException,
                          self.client.get_stations)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_bad_request_exception(self, download_url_mock):
        """
        Verify that a bad request raises an identifiable exception
        """
        download_url_mock.return_value = (400, io.BytesIO(b""))
        self.assertRaises(FDSNBadRequestException, self.client.get_stations)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_server_exception(self, download_url_mock):
        """
        Verify that a server error raises an identifiable exception
        """
        download_url_mock.return_value = (500, None)
        self.assertRaises(FDSNInternalServerException,
                          self.client.get_stations)

    @mock.patch("obspy.clients.fdsn.client.download_url")
    def test_too_many_requests_exception(self, download_url_mock):
        """
        Verify that too many requests raise an identifiable exception
        """
        download_url_mock.return_value = (429, None)
        self.assertRaises(FDSNTooManyRequestsException,
                          self.client.get_stations)

    def test_eida_token_resolution(self):
        """
        Tests that EIDA tokens are resolved correctly and new credentials get
        installed with the opener of the Client.
        """
        token = os.path.join(self.datapath, 'eida_token.txt')
        with open(token, 'rb') as fh:
            token_data = fh.read().decode()

        def _assert_eida_user_and_password(user, password):
            # user/pass is not static for the static test token
            for value in user, password:
                # seems safe to assume both user and password are at least 10
                # chars long
                # example user/password:
                # wWGgJnH4GvdVY7gDMH21xEpb wDnzlpljqdaCXlP2
                re.match('^[a-zA-Z0-9]{10,}$', value)

        def _get_http_digest_auth_handler(client):
            handlers = [h for h in client._url_opener.handlers
                        if isinstance(h, urllib_request.HTTPDigestAuthHandler)]
            self.assertLessEqual(len(handlers), 1)
            return handlers and handlers[0] or None

        def _assert_credentials(client, user, password):
            handler = _get_http_digest_auth_handler(client)
            self.assertIsInstance(handler,
                                  urllib_request.HTTPDigestAuthHandler)
            for user_, password_ in handler.passwd.passwd[None].values():
                self.assertEqual(user, user_)
                self.assertEqual(password, password_)

        client = Client('GFZ')
        # this is a plain client, so it should not have http digest auth
        self.assertEqual(_get_http_digest_auth_handler(client), None)
        # now, if we set new user/password, we should get a http digest auth
        # handler
        user, password = ("spam", "eggs")
        client._set_opener(user=user, password=password)
        _assert_credentials(client, user, password)
        # now, if we resolve the EIDA token, the http digest auth handler
        # should change
        user, password = client._resolve_eida_token(token=token)
        _assert_eida_user_and_password(user, password)
        client._set_opener(user=user, password=password)
        _assert_credentials(client, user, password)
        # do it again, now providing the token data directly as a string (first
        # change the authentication again to dummy user/password
        client._set_opener(user="foo", password="bar")
        _assert_credentials(client, "foo", "bar")
        user, password = client._resolve_eida_token(token=token_data)
        _assert_eida_user_and_password(user, password)
        client.set_eida_token(token_data)
        _assert_credentials(client, user, password)

        # Raise if token and user/pw are given.
        with self.assertRaises(FDSNException) as err:
            Client('GFZ', eida_token=token, user="foo", password="bar")
        self.assertEqual(
            err.exception.args[0],
            "EIDA authentication token provided, but user and password are "
            "also given.")

        # now lets test the RoutingClient with credentials..
        credentials_ = {'geofon.gfz-potsdam.de': {'eida_token': token}}
        credentials_mapping_ = {'GFZ': {'eida_token': token}}
        global_eida_credentials_ = {'EIDA_TOKEN': token}
        for credentials, should_have_credentials in zip(
                (None, credentials_, credentials_mapping_,
                 global_eida_credentials_), (False, True, True, True)):
            def side_effect(self_, *args, **kwargs):
                """
                This mocks out Client.get_waveforms_bulk which gets called by
                the routing client, checks authentication handlers and returns
                a dummy stream.
                """
                # check that we're at the expected FDSN WS server
                self.assertEqual('http://geofon.gfz-potsdam.de',
                                 self_.base_url)
                # check if credentials were used
                # eida auth availability should be positive in all cases
                self.assertTrue(self_._has_eida_auth)
                # depending on whether we specified credentials, the
                # underlying FDSN client should have EIDA authentication
                # flag and should also have a HTTP digest handler with
                # appropriate user/password
                handler = _get_http_digest_auth_handler(self_)
                if should_have_credentials:
                    for user, password in handler.passwd.passwd[None].values():
                        _assert_eida_user_and_password(user, password)
                else:
                    self.assertEqual(handler, None)
                # just always return some dummy stream, we're not
                # interested in checking the data downloading which
                # succeeds regardless if auth is used or not as it's public
                # data
                return Stream([Trace(data=np.ones(2))])

            with mock.patch(
                    'obspy.clients.fdsn.client.Client.get_waveforms_bulk',
                    autospec=True) as p:

                p.side_effect = side_effect

                routing_client = RoutingClient('eida-routing',
                                               credentials=credentials)
                # do a waveform request on the routing client which internally
                # connects to the GFZ FDSNWS. this should be done using the
                # above supplied credentials, i.e. should use the given EIDA
                # token to resolve user/password for the normal FDSN queryauth
                # mechanism
                routing_client.get_waveforms(
                    network="GE", station="KMBO", location="00", channel="BHZ",
                    starttime=UTCDateTime("2010-02-27T06:30:00.000"),
                    endtime=UTCDateTime("2010-02-27T06:40:00.000"))

        # test invalid token/token file
        with self.assertRaisesRegex(
                ValueError,
                'EIDA token does not seem to be a valid PGP message'):
            client = Client('GFZ', eida_token="spam")
        with self.assertRaisesRegex(
                ValueError,
                "Read EIDA token from file '[^']*event_helpstring.txt' but it "
                "does not seem to contain a valid PGP message."):
            client = Client(
                'GFZ', eida_token=os.path.join(self.datapath,
                                               'event_helpstring.txt'))


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
