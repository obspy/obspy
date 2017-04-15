#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.fdsn.client test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
#TODO clean this module up. it's merely swiped from the Client test suite. 

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
#import re
#import sys
import unittest
#import warnings

import requests

from obspy import UTCDateTime, read, read_inventory
#from obspy.core.compatibility import mock
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients.fdsn.routers import Federatedclient
from obspy.clients.fdsn.header import (DEFAULT_USER_AGENT,
                                       FDSNException, FDSNRedirectException,
                                       FDSNNoDataException)
from obspy.core.inventory import Response
from obspy.geodetics import locations2degrees


USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())
class FederatedclientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.fdsn.client.routers.Federatedclient.
    """

    @classmethod
    def setUpClass(cls):
        # directory where the test files are located
        cls.path = os.path.dirname(__file__)
        cls.datapath = os.path.join(cls.path, "data")
        cls.client = Federatedclient(user_agent=USER_AGENT)
        cls.client_auth = \
            Federatedclient(base_url="IRIS", user_agent=USER_AGENT,
                   user="nobody@iris.edu", password="anonymous")

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
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
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
        self.assertEqual(got, expected, failmsg(got, expected))

    def test_conflicting_params(self):
        """
        """
        self.assertRaises(FDSNException, self.client.get_stations,
                          network="IU", net="IU")

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


def suite():
    return unittest.makeSuite(FederatedClientTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')