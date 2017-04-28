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
from obspy.clients.fdsn import (FederatedClient, Client)
from obspy.clients.fdsn.header import (DEFAULT_USER_AGENT,
                                       FDSNException, FDSNRedirectException,
                                       FDSNNoDataException)
from obspy.core.inventory import Response
from obspy.geodetics import locations2degrees
from obspy.clients.fdsn.tests.test_client import failmsg
from obspy.clients.fdsn.routers.fedcatalog_client import (data_to_request, get_bulk_string)
from obspy.clients.fdsn.routers.fedcatalog_parser import (FDSNBulkRequestItem,
                                                          FDSNBulkRequests)


USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())

def hasLevel(inv, level):
    """
    """
    if not inv or not inv.networks:
        return level is None
    if not inv.networks[0].stations:
        return level is "network"
    if not inv.networks[0].stations[0].channels:
        return level is "station"
    if not inv.networks[0].stations[0].channels[0].response.response_stages:
        return level is "channel"
    return level is "response"

class BulkConversionTestCase(unittest.TestCase):
    def test_simple(self):
        bulk = "AB STA 00 ABC 2015-12-23-01:00:00 2015-12-23-02:00:00"
        ans = get_bulk_string(bulk, None)
        self.assertTrue(isinstance(ans, str))
        self.assertEqual(len(ans.splitlines()), 1)

        bulk = "AB STA 00 ABC 2015-12-23-01:00:00.000 2015-12-23-02:00:00.000"
        ans = get_bulk_string(bulk, None)
        self.assertTrue(isinstance(ans, str))
        self.assertEqual(len(ans.splitlines()), 1)

    def test_multiline(self):
        bulk1 = "AB STA 00 ABC 2015-12-23-01:00:00.000 2015-12-23-02:00:00.000"
        bulk2 = "AB STA 10 BBB 2015-12-23-01:00:00.000 2015-12-23-02:00:00.000"
        bulk = "\n".join((bulk1, bulk2))
        ans = get_bulk_string(bulk, None)
        self.assertTrue(isinstance(ans, str))
        self.assertEqual(len(ans.splitlines()), 2)

    def test_with_args(self):
        bulk1 = "AB STA 00 ABC 2015-12-23-01:00:00.000 2015-12-23-02:00:00.000"
        bulk2 = "AB STA 10 BBB 2015-12-23-01:00:00.000 2015-12-23-02:00:00.000"
        args = {"first":1, "second":'two', "third":False}
        bulk = "\n".join((bulk1, bulk2))
        ans = get_bulk_string(bulk, args)
        self.assertTrue(isinstance(ans, str))
        self.assertEqual(len(ans.splitlines()), 5)


class FDSNBulkRequestItemClientTestCase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass

    def test_inits(self):        
        text = "IU ANMO -- BHZ 2012-04-25T00:00:00 2012-06-12T10:10:10"
        l2 = '* * * * * *'
        self.assertEqual(str(FDSNBulkRequestItem(line=text)),
                          "IU ANMO -- BHZ 2012-04-25T00:00:00.000 2012-06-12T10:10:10.000")
        self.assertEqual(str(FDSNBulkRequestItem(line=l2)), "* * * * * *")
        self.assertEqual(str(FDSNBulkRequestItem()), "* * * * * *")

        byparams = FDSNBulkRequestItem(network='IU', station='ANMO', location='  ',
                                       channel='BHZ', starttime='2012-04-25',
                                       endtime='2012-06-12T10:10:10')
        self.assertEqual(str(byparams),
                          "IU ANMO -- BHZ 2012-04-25T00:00:00.000 2012-06-12T10:10:10.000")

        byparams = FDSNBulkRequestItem(station='ANMO', location='  ', starttime='2012-04-25')
        self.assertEqual(str(byparams), "* ANMO -- * 2012-04-25T00:00:00.000 *")

        byparams = FDSNBulkRequestItem(network='IU', channel='BHZ', endtime='2012-06-12T10:10:10')
        self.assertEqual(str(byparams), "IU * * BHZ * 2012-06-12T10:10:10.000")

    def test_comparisons(self):
        l1 = 'AB CDE 01 BHZ 2015-04-25T02:45:32 2015-04-25T02:47:00'
        l2 = '* * * * * *'
        l3 = 'AB CDE 01 BHZ 2015-04-25T00:00:00 2015-04-25T02:47:00'
        l4 = 'AB CDE 01 BHZ 2015-04-25T02:45:32 2015-04-25T03:00:00'
        A = FDSNBulkRequestItem(line=l3) #   [-------]
        B = FDSNBulkRequestItem(line=l4) #        [-------]
        C = FDSNBulkRequestItem(line=l1) #        [--]
        D = FDSNBulkRequestItem(line=l2) # <---------------->
        self.assertTrue(A.contains(l1) and B.contains(C))
        self.assertTrue(C.contains(C) and D.contains(C))
        self.assertFalse(C.contains(A) or C.contains(B) or C.contains(D))
        self.assertTrue(A == A)
        self.assertTrue(A == FDSNBulkRequestItem(line=l3))

class FederatedClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.fdsn.client.routers.Federatedclient.
    """

    @classmethod
    def setUpClass(cls):
        # directory where the test files are located
        cls.path = os.path.dirname(__file__)
        cls.datapath = os.path.join(cls.path,"..","..","tests", "data")
        cls.fed_client = FederatedClient(user_agent=USER_AGENT)
        cls.iris_client = Client("IRIS", user_agent=USER_AGENT)
        cls.client = FederatedClient(user_agent=USER_AGENT, include_provider='IRIS')
        cls.client_auth = \
            FederatedClient(base_url="IRIS", user_agent=USER_AGENT,
                   user="nobody@iris.edu", password="anonymous", include_provider='IRIS')

    def test_fedstations_bulk_simple(self):
        bulktext = "IU ANMO 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"
        inv = self.fed_client.get_stations_bulk(bulktext)
        #default level of station
        self.assertTrue(hasLevel(inv, "station") and not hasLevel(inv, "channel"))
        self.assertEqual(len(inv.networks[0].stations), 1)
        self.assertEqual(inv.networks[0].stations[0].code, 'ANMO')

    def test_fedstations_bulk_stringio(self):
        # ensure service works at all
        bulktext = "level=channel\nIU ANMO 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"
        file_thing = io.StringIO(bulktext)

        # send bulk as a file-type-object
        inv = self.fed_client.get_stations_bulk(file_thing)
        self.assertEqual(len(inv.networks[0].stations), 1)
        self.assertEqual(inv.networks[0].stations[0].code, 'ANMO')
        self.assertEqual(len(inv.networks[0].stations[0].channels), 1)
        self.assertEqual(inv.networks[0].stations[0].channels[0].code, 'BHZ')

    def test_fedstations_bulk_with_separate_param(self):
        # ensure service works at all
        bulktext = "IU ANMO 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"

        # send bulk with a specified level as param
        inv = self.fed_client.get_stations_bulk(bulktext, level="channel")
        self.assertEqual(len(inv.networks[0].stations), 1)
        self.assertEqual(inv.networks[0].stations[0].code, 'ANMO')
        self.assertEqual(len(inv.networks[0].stations[0].channels), 1)
        self.assertEqual(inv.networks[0].stations[0].channels[0].code, 'BHZ')

    def test_fedstations_bulk_with_embeded_param(self):
        # ensure service works at all
        bulktext = "IU ANMO 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"
        #  send bulk with a specified level included in text
        inv = self.fed_client.get_stations_bulk("level=channel\n" + bulktext)
        self.assertEqual(len(inv.networks[0].stations), 1)
        self.assertEqual(inv.networks[0].stations[0].code, 'ANMO')
        self.assertEqual(len(inv.networks[0].stations[0].channels), 1)
        self.assertEqual(inv.networks[0].stations[0].channels[0].code, 'BHZ')

    def test_fedstations_param(self):
        endt = UTCDateTime(2010, 2, 27, 6, 45, 0)
        params = {"network":"IU", "station":"ANMO", "location":"00",
                  "channel":"BHZ", "starttime":"2010-02-27T06:30:00",
                  "endtime": endt,
                  "level":"channel"}
        inv = self.fed_client.get_stations(**params)
        self.assertTrue(hasLevel(inv, "channel") and not hasLevel(inv, "response"))

    def test_fedwaveforms_bulk(self):
        bulktext = "IU ANMO 00 BHZ 2010-02-27T06:30:00 2010-02-27T06:45:00"
        data = self.fed_client.get_waveforms_bulk(bulktext)
        self.assertEqual(len(data), 1, msg="expected one waveform in stream")
        self.assertEqual(data[0].id, 'IU.ANMO.00.BHZ',
                         msg="retrieved incorrect waveform {0}".format(data[0].id))

    def test_fedwaveforms(self):
        endt = UTCDateTime(2010, 2, 27, 6, 45, 0)
        params = {"network":"IU", "station":"ANMO", "location":"00",
                  "channel":"BHZ", "starttime":"2010-02-27T06:30:00",
                  "endtime": endt,
                  "level":"channel"}
        data = self.fed_client.get_waveforms(**params)
        self.assertEqual(len(data), 1, msg="expected one waveform in stream")
        self.assertEqual(data[0].id, 'IU.ANMO.00.BHZ',
                         msg="retrieved incorrect waveform {0}".format(data[0].id))
        # don't forget to test the attach_response


    def test_fedstations_retry(self):
        pass
    def test_fedwaveforms_retry(self):
        pass

    def test__example_queries_station(self):
        """
        Tests the (sometimes modified) example queries given on IRIS webpage.

        This test used to download files but that is almost impossible to
        keep up to date - thus it is now a bit smarter and tests the
        returned inventory in different ways.
        """
        client = self.client

        # Radial query.
        inv = client.get_stations(latitude=-56.1, longitude=-26.7,
                                maxradius=15, station="*")
        self.assertGreater(len(inv.networks), 0)  # at least one network
        for net in inv:
            self.assertGreater(len(net.stations), 0)  # at least one station
            for sta in net:
                dist = locations2degrees(sta.latitude, sta.longitude,
                                        -56.1, -26.7)
                # small tolerance for WGS84.
                self.assertGreater(15.1, dist, "%s.%s" % (net.code,
                                                        sta.code))

        '''XXX Commented out because service itself cannot support maxlon<minlon
                should be fixed soon by IRIS
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
        '''
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
            got = client.get_waveforms(*query, include_provider="IRIS")
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))
            # test output to file
            with NamedTemporaryFile() as tf:
                client.get_waveforms(*query, filename=tf.name, include_provider="IRIS")

                
                base_name = os.path.basename(tf.name)
                path_name = os.path.dirname(tf.name)
                base_name = '-'.join(('IRIS', base_name))
                tf.name = os.path.join(path_name, base_name)


                with open(tf.name, 'rb') as fh:
                    got = fh.read()
                with open(file_, 'rb') as fh:
                    expected = fh.read()
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))

    def xtest_authentication(self):
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

    def xtest_dataselect_bulk(self):
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

    def xtest_station_bulk(self):
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

    def xtest_get_waveform_attach_response(self):
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

def test_suite():
    from unittest import (TestSuite, makeSuite)
    suite = TestSuite()
    suite.addTest(makeSuite(BulkConversionTestCase, 'test'))
    suite.addTest(makeSuite(FederatedClientTestCase, 'test'))
    suite.addTest(makeSuite(FDSNBulkRequestItemClientTestCase, 'test'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite', verbosity=100)