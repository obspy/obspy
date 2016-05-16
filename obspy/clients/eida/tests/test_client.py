#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.eida test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Helmholtz-Zentrum Potsdam - Deutsches GeoForschungsZentrum GFZ
    (geofon@gfz-potsdam.de)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2

import io
import os
import re
import unittest
from difflib import Differ

if PY2:
    from urllib import urlopen

else:
    from urllib.request import urlopen

from obspy import UTCDateTime, read, read_inventory
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients.eida import Client
from obspy.clients.fdsn.header import (URL_MAPPINGS, FDSNException)
from obspy.core.inventory import Response
from obspy.geodetics import locations2degrees


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
    Test cases for obspy.clients.eida.Client.
    """
    @classmethod
    def credentials(cls):
        url = re.sub("^http:", "https:", URL_MAPPINGS[cls.base_url]) + \
                "/fdsnws/dataselect/1/auth"

        fd = urlopen(url, cls.token)

        try:
            userpass = fd.read()

            if isinstance(userpass, bytes):
                userpass = userpass.decode('utf-8')

            return userpass.split(':')

        finally:
            fd.close()

    @classmethod
    def setUpClass(cls):
        # directory where the test files are located
        cls.path = os.path.dirname(__file__)
        cls.datapath = os.path.join(cls.path, "data")
        cls.base_url = "GFZ"
        cls.token = b"""
-----BEGIN PGP MESSAGE-----
Version: GnuPG v2.0.9 (GNU/Linux)

owGbwMvMwMR4USg04vQMEQbG0+5JDOEWH+OrlXITM3OUrBSUSlKLSxzyk4oLKvXy
i9KVdBSUyhJzMlPiS/NKIAqMDIwMdA0MgSjEwMAKjKKUajsZZVgYGJkY2FiZQOYx
cHEKwCx5Lcv+T0W79KhOxauJLEnP7t0768ZTvzk1Z7XbnSSmqvB7/rfsRfOvSrMt
39GjosTgveqB7gHRYz0+jPP36dj073r2s8h6Xvu0+sAtLbfO6C3xNolRDPgQ4tud
m6J8tm573KGFFTG/+uI/hSir56Uav33xN1DBwKPs1pt8H6vFirf81J5Y3eR8wbmm
c9eXV5VeBy/9bYrd+TXsnFkiayjrxViWdm8uUXnj4hzDE5vvTeVZGH/nclVE1v6z
DO3n3STZvjMKsRUxVPKcnTmHcU64qiTX10bRqi1yHzuMs35Gzz0sMpXbTencLduG
hMrp91PMe04c0otcNytOKDFe/uDva3sd9jZKcxSXb3Gs2Rxf2QgA
=VVHX
-----END PGP MESSAGE-----
"""
        cls.cred = {
            URL_MAPPINGS[cls.base_url] + "/fdsnws/dataselect/1/queryauth":
                cls.credentials()}

        cls.client = Client(base_url=cls.base_url)

        cls.client_cred = Client(base_url=cls.base_url,
                                 credentials=cls.cred)

        cls.client_token = Client(base_url=cls.base_url,
                                  authdata=cls.token)

    def test_iris_example_queries_station(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.

        This test used to download files but that is almost impossible to
        keep up to date - thus it is now a bit smarter and tests the
        returned inventory in different ways.
        """
        client = self.client

        # Simple query
        inv = client.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2001-01-01"), net="GE", sta="KMBO")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                self.assertGreater(UTCDateTime("2001-01-01"), sta.start_date)
                if sta.end_date is not None:
                    self.assertGreater(sta.end_date, UTCDateTime("2000-01-01"))
                self.assertEqual(net.code, "GE")
                self.assertEqual(sta.code, "KMBO")

        # Station wildcard query.
        inv = client.get_stations(
            starttime=UTCDateTime("2000-01-01"),
            endtime=UTCDateTime("2002-01-01"), network="GE", sta="K*",
            location="00")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                self.assertGreater(UTCDateTime("2002-01-01"), sta.start_date)
                if sta.end_date is not None:
                    self.assertGreater(sta.end_date, UTCDateTime("2000-01-01"))
                self.assertEqual(net.code, "GE")
                self.assertTrue(sta.code.startswith("K"))

        # Radial query.
        inv = client.get_stations(latitude=20, longitude=-150,
                                  maxradius=15)
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                dist = locations2degrees(sta.latitude, sta.longitude,
                                         20, -150)
                # small tolerance for WGS84.
                self.assertGreater(15.1, dist, "%s.%s" % (net.code,
                                                          sta.code))

        # Misc query.
        inv = client.get_stations(
            startafter=UTCDateTime("2000-01-07"),
            endbefore=UTCDateTime("2011-02-07"), minlatitude=15,
            maxlatitude=55, minlongitude=150, maxlongitude=-150, network="GE")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                msg = "%s.%s" % (net.code, sta.code)
                self.assertGreater(sta.start_date, UTCDateTime("2000-01-07"),
                                   msg)
                if sta.end_date is not None:
                    self.assertGreater(UTCDateTime("2011-02-07"), sta.end_date,
                                       msg)
                self.assertGreater(sta.latitude, 14.9, msg)
                self.assertGreater(55.1, sta.latitude, msg)
                self.assertFalse(-150.1 <= sta.longitude <= 150.1, msg)
                self.assertEqual(net.code, "GE", msg)

    def test_iris_example_queries_dataselect(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.
        """
        client = self.client

        queries = [
            ("GE", "KMBO", "00", "BHZ",
             UTCDateTime("2010-02-27T06:30:00.000"),
             UTCDateTime("2010-02-27T06:40:00.000")),
            ("GE", "K*", "*", "BHZ",
             UTCDateTime("2010-02-27T06:30:00.000"),
             UTCDateTime("2010-02-27T06:31:00.000")),
            ("GE", "K??", "*0", "BHZ",
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

    def test_cred_auth(self):
        """
        Test dataselect with authentication.
        """
        client = self.client_cred
        # dataselect example queries
        query = ("GE", "KMBO", "00", "BHZ",
                 UTCDateTime("2010-02-27T06:30:00.000"),
                 UTCDateTime("2010-02-27T06:40:00.000"))
        filename = "dataselect_example.mseed"
        got = client.get_waveforms(*query)
        file_ = os.path.join(self.datapath, filename)
        expected = read(file_)
        self.assertEqual(got, expected, failmsg(got, expected))

    def test_token_auth(self):
        """
        Test dataselect with authentication.
        """
        client = self.client_token
        # dataselect example queries
        query = ("GE", "KMBO", "00", "BHZ",
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
                          network="GE", net="GE")

    def test_dataselect_bulk(self):
        """
        Test bulk dataselect requests, POSTing data to server. Also tests
        authenticated bulk request.
        """
        clients = [self.client, self.client_cred, self.client_token]
        file = os.path.join(self.datapath, "bulk.mseed")
        expected = read(file)
        # test cases for providing lists of lists
        bulk = (("GE", "KMBO", "00", "BH1",
                 UTCDateTime("2016-03-25T00:00:00"),
                 UTCDateTime("2016-03-25T00:00:04")),
                ("G", "IVI", "*", "BHZ",
                 UTCDateTime("2016-03-25T00:00:00"),
                 UTCDateTime("2016-03-25T00:00:06")),
                ("NO", "TROLL", "", "BHE",
                 UTCDateTime("2016-03-25T00:00:00"),
                 UTCDateTime("2016-03-25T00:00:06")),
                ("CH", "PANIX", "*", "LHN",
                 UTCDateTime("2016-03-25T00:00:00"),
                 UTCDateTime("2016-03-25T00:00:08")))
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk)
            self.assertEqual(''.join(sorted(str(got).splitlines())),
                             ''.join(sorted(str(expected).splitlines())),
                             failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk, filename=tf.name)
                got = read(tf.name)
            self.assertEqual(''.join(sorted(str(got).splitlines())),
                             ''.join(sorted(str(expected).splitlines())),
                             failmsg(got, expected))
        # test cases for providing a request string
        bulk = ("GE KMBO 00 BH1 2016-03-25T00:00:00 2016-03-25T00:00:04\n"
                "G IVI * BHZ 2016-03-25T00:00:00 2016-03-25T00:00:06\n"
                "NO TROLL -- BHE 2016-03-25T00:00:00 2016-03-25T00:00:06\n"
                "CH PANIX * LHN 2016-03-25T00:00:00 2016-03-25T00:00:08\n")
        for client in clients:
            # test output to stream
            got = client.get_waveforms_bulk(bulk)
            self.assertEqual(''.join(sorted(str(got).splitlines())),
                             ''.join(sorted(str(expected).splitlines())),
                             failmsg(got, expected))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms_bulk(bulk, filename=tf.name)
                got = read(tf.name)
            self.assertEqual(''.join(sorted(str(got).splitlines())),
                             ''.join(sorted(str(expected).splitlines())),
                             failmsg(got, expected))
        # test cases for providing a file name
        for client in clients:
            with NamedTemporaryFile() as tf:
                with open(tf.name, "wt") as fh:
                    fh.write(bulk)
                got = client.get_waveforms_bulk(bulk)
            self.assertEqual(''.join(sorted(str(got).splitlines())),
                             ''.join(sorted(str(expected).splitlines())),
                             failmsg(got, expected))
        # test cases for providing a file-like object
        for client in clients:
            got = client.get_waveforms_bulk(io.StringIO(bulk))
            self.assertEqual(''.join(sorted(str(got).splitlines())),
                             ''.join(sorted(str(expected).splitlines())),
                             failmsg(got, expected))

    def test_station_bulk(self):
        """
        Test bulk station requests, POSTing data to server. Also tests
        authenticated bulk request.

        Does currently only test reading from a list of list. The other
        input types are tested with the waveform bulk downloader and thus
        should work just fine.
        """
        clients = [self.client, self.client_cred, self.client_token]
        # test cases for providing lists of lists
        starttime = UTCDateTime(2016, 1, 1)
        endtime = UTCDateTime(2016, 1, 1) + 10
        bulk = [
            ["GE", "KMBO", "00", "BH1", starttime, endtime],
            ["G", "IVI", "00", "BHZ", starttime, endtime],
            ["NO", "TROLL", "", "BHE", starttime, endtime],
            ["CH", "PANIX", "", "LHN", starttime, endtime],
        ]
        for client in clients:
            # Test with station level.
            inv = client.get_stations_bulk(bulk, level="station")
            # Test with output to file.
            with NamedTemporaryFile() as tf:
                client.get_stations_bulk(
                    bulk, filename=tf.name, level="station")
                inv2 = read_inventory(tf.name, format="stationxml")

            self.assertEqual(sorted(inv.networks, key=lambda _i: _i.code),
                             sorted(inv2.networks, key=lambda _i: _i.code))
            self.assertEqual(len(inv.networks), 4)
            self.assertEqual(sorted([net.code for net in inv.networks]),
                             sorted(["GE", "G", "NO", "CH"]))
            self.assertEqual(sum([len(net.stations) for net in inv.networks]),
                             4)
            self.assertEqual(
                sorted([_i.code for _j in inv.networks for _i in _j.stations]),
                sorted(["KMBO", "IVI", "TROLL", "PANIX"]))

            # Test with channel level.
            inv = client.get_stations_bulk(bulk, level="channel")
            # Test with output to file.
            with NamedTemporaryFile() as tf:
                client.get_stations_bulk(
                    bulk, filename=tf.name, level="channel")
                inv2 = read_inventory(tf.name, format="stationxml")

            self.assertEqual(sorted(inv.networks, key=lambda _i: _i.code),
                             sorted(inv2.networks, key=lambda _i: _i.code))
            self.assertEqual(len(inv.networks), 4)
            self.assertEqual(sorted([net.code for net in inv.networks]),
                             sorted(["GE", "G", "NO", "CH"]))
            self.assertEqual(sum([len(net.stations) for net in inv.networks]),
                             4)
            self.assertEqual(
                sorted([_i.code for _j in inv.networks for _i in _j.stations]),
                sorted(["KMBO", "IVI", "TROLL", "PANIX"]))
            channels = []
            for network in inv.networks:
                for station in network:
                    for channel in station:
                        channels.append("%s.%s.%s.%s" % (network.code,
                                                         station.code,
                                                         channel.location_code,
                                                         channel.code))
            self.assertEqual(
                sorted(channels),
                sorted(["GE.KMBO.00.BH1", "G.IVI.00.BHZ", "NO.TROLL..BHE",
                        "CH.PANIX..LHN"]))
        return

    def test_get_waveform_attach_response(self):
        """
        minimal test for automatic attaching of metadata
        """
        client = self.client

        bulk = ("GE KMBO 00 BHZ 2016-03-25T00:00:00 2016-03-25T00:00:04\n")
        st = client.get_waveforms_bulk(bulk, attach_response=True)
        for tr in st:
            self.assertTrue(isinstance(tr.stats.get("response"), Response))

        st = client.get_waveforms("GE", "KMBO", "00", "BHZ",
                                  UTCDateTime("2000-02-27T06:00:00.000"),
                                  UTCDateTime("2000-02-27T06:00:05.000"),
                                  attach_response=True)
        for tr in st:
            self.assertTrue(isinstance(tr.stats.get("response"), Response))


def suite():
    return unittest.makeSuite(ClientTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
