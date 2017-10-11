#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.fdsn.client test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
import sys
import unittest

from obspy import UTCDateTime, read
# from obspy.core.compatibility import mock
from obspy.core.util.base import NamedTemporaryFile
from obspy.clients.fdsn import (FederatedClient, Client)
from obspy.clients.fdsn.header import (DEFAULT_USER_AGENT, FDSNNoDataException)
from obspy.core.inventory import Response
from obspy.geodetics import locations2degrees
from obspy.clients.fdsn.tests.test_client import failmsg
from obspy.clients.fdsn.routers.fedcatalog_client import (
    get_bulk_string, FederatedRoutingManager)
from obspy.clients.fdsn.routers.fedcatalog_parser import FDSNBulkRequestItem


USER_AGENT = "ObsPy (test suite) " + " ".join(DEFAULT_USER_AGENT.split())


def has_level(inv, level):
    """
    Check to see if the inventory structure has details down to specific level

    :type level: str
    :param level: 'network', 'station', 'location', 'channel'
    :rtype: bool
    :returns: true if an Inventory structure has any fields of that level

    :note: only checks the first item of each category
    """
    if not inv or not inv.networks:
        return level is None
    if not inv.networks[0].stations:
        return level == "network"
    if not inv.networks[0].stations[0].channels:
        return level == "station"
    if not inv.networks[0].stations[0].channels[0].response.response_stages:
        return level == "channel"
    return level is "response"


class BulkConversionTestCase(unittest.TestCase):
    """
    Test bulk conversion
    """
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
        args = {"first": 1, "second": 'two', "third": False}
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
        self.assertEqual(
            str(FDSNBulkRequestItem(line=text)),
            "IU ANMO -- BHZ 2012-04-25T00:00:00.000 2012-06-12T10:10:10.000")
        self.assertEqual(str(FDSNBulkRequestItem(line=l2)), "* * * * * *")
        self.assertEqual(str(FDSNBulkRequestItem()), "* * * * * *")

        byparams = FDSNBulkRequestItem(
            network='IU', station='ANMO', location='  ', channel='BHZ',
            starttime='2012-04-25', endtime='2012-06-12T10:10:10')
        self.assertEqual(
            str(byparams),
            "IU ANMO -- BHZ 2012-04-25T00:00:00.000 2012-06-12T10:10:10.000")

        byparams = FDSNBulkRequestItem(station='ANMO', location='  ',
                                       starttime='2012-04-25')
        self.assertEqual(str(byparams),
                         "* ANMO -- * 2012-04-25T00:00:00.000 *")

        byparams = FDSNBulkRequestItem(network='IU', channel='BHZ',
                                       endtime='2012-06-12T10:10:10')
        self.assertEqual(str(byparams),
                         "IU * * BHZ * 2012-06-12T10:10:10.000")

    def test_comparisons(self):
        l1 = 'AB CDE 01 BHZ 2015-04-25T02:45:32 2015-04-25T02:47:00'
        l2 = '* * * * * *'
        l3 = 'AB CDE 01 BHZ 2015-04-25T00:00:00 2015-04-25T02:47:00'
        l4 = 'AB CDE 01 BHZ 2015-04-25T02:45:32 2015-04-25T03:00:00'
        a = FDSNBulkRequestItem(line=l3)  # .   [-------]
        b = FDSNBulkRequestItem(line=l4)  # .        [-------]
        c = FDSNBulkRequestItem(line=l1)  # .        [--]
        d = FDSNBulkRequestItem(line=l2)  # .<---------------->
        self.assertTrue(a.contains(l1) and b.contains(c))
        self.assertTrue(c.contains(c) and d.contains(c))
        self.assertFalse(c.contains(a) or c.contains(b) or c.contains(d))
        self.assertTrue(a == a)
        self.assertTrue(a == FDSNBulkRequestItem(line=l3))


class FederatedClientTestCase(unittest.TestCase):
    """
    Test cases for obspy.clients.fdsn.client.routers.Federatedclient.
    """

    @classmethod
    def setUpClass(cls):
        # directory where the test files are located
        cls.path = os.path.dirname(__file__)
        cls.datapath = os.path.join(cls.path, "data")
        cls.fed_client = FederatedClient(user_agent=USER_AGENT)
        cls.iris_client = Client("IRIS", user_agent=USER_AGENT)
        cls.client = FederatedClient(user_agent=USER_AGENT,
                                     include_provider='IRIS')
        cls.client_auth = \
            FederatedClient(user_agent=USER_AGENT, user="nobody@iris.edu",
                            password="anonymous", include_provider='IRIS')

    def test_fedstations_bulk_simple(self):
        bulktext = "IU ANMO 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"
        inv = self.fed_client.get_stations_bulk(bulktext)
        # default level of station
        self.assertTrue(has_level(inv, "station"))
        self.assertFalse(has_level(inv, "channel"))
        self.assertEqual(len(inv.networks[0].stations), 1)
        self.assertEqual(inv.networks[0].stations[0].code, 'ANMO')

    def test_fedstations_parallel(self):
        fed_client = FederatedClient(use_parallel=True)
        bulktext = "IU ANTO * BHZ 2015-01-01T00:00:00 2015-02-01T00:00:00"
        inv = fed_client.get_stations_bulk(bulktext, includeoverlaps=True)
        print(inv)

        # test something against the SED/ETH
        bulktext = "C? B* -- BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"
        inv = fed_client.get_stations_bulk(bulktext)
        # default level of station
        self.assertTrue(has_level(inv, "station"))
        self.assertFalse(has_level(inv, "channel"))
        self.assertGreater(len(inv.networks[0].stations), 4)
        endt = UTCDateTime(2015, 5, 31, 0, 0, 0)
        params = {"network": "I?", "station": "B*", "channel": "BHZ",
                  "starttime": "2015-01-01T00:00:00",
                  "endtime": endt,
                  "level": "channel"}

        inv = fed_client.get_stations(**params)
        self.assertTrue(has_level(inv, "channel"))
        self.assertFalse(has_level(inv, "response"))

    def test_fedstations_bulk_simple_many(self):
        bulktext = "* A* 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00"
        inv = self.fed_client.get_stations_bulk(bulktext)
        # default level of station
        self.assertTrue(has_level(inv, "station"))
        self.assertFalse(has_level(inv, "channel"))
        self.assertGreater(len(inv.networks), 5)

    def test_fedstations_bulk_stringio(self):
        # ensure service works at all
        bulktext = '''level=channel
        IU ANMO 00 BHZ 2015-01-01T00:00:00 2015-05-31T00:00:00'''
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
        params = {"network": "IU", "station": "ANMO", "location": "00",
                  "channel": "BHZ", "starttime": "2010-02-27T06:30:00",
                  "endtime": endt,
                  "level": "channel"}
        inv = self.fed_client.get_stations(**params)
        self.assertTrue(has_level(inv, "channel"))
        self.assertFalse(has_level(inv, "response"))

    def test_fedwaveforms_bulk(self):
        bulktext = "IU ANMO 00 BHZ 2010-02-27T06:30:00 2010-02-27T06:45:00"
        data = self.fed_client.get_waveforms_bulk(bulktext)
        self.assertEqual(len(data), 1, msg="expected one waveform in stream")
        self.assertEqual(
            data[0].id, 'IU.ANMO.00.BHZ',
            msg="retrieved incorrect waveform {0}".format(data[0].id))

    def test_fedwaveforms(self):
        endt = UTCDateTime(2010, 2, 27, 6, 45, 0)
        params = {"network": "IU", "station": "ANMO", "location": "00",
                  "channel": "BHZ", "starttime": "2010-02-27T06:30:00",
                  "endtime": endt,
                  "level": "channel"}
        data = self.fed_client.get_waveforms(**params)
        self.assertEqual(len(data), 1, msg="expected one waveform in stream")
        self.assertEqual(
            data[0].id, 'IU.ANMO.00.BHZ',
            msg="retrieved incorrect waveform {0}".format(data[0].id))
        self.assertEqual(len(data), 1, msg="expected one waveform in stream")
        self.assertEqual(
            data[0].id, 'IU.ANMO.00.BHZ',
            msg="retrieved incorrect waveform {0}".format(data[0].id))
        # test including an empty location code
        endt = UTCDateTime(2017, 6, 1, 1, 0, 0)
        params = {"network": "TA", "station": "O14K", "location": "",
                  "channel": "LHZ", "starttime": "2017-06-01T00:00:00",
                  "endtime": endt}
        data = self.fed_client.get_waveforms(**params)
        self.assertEqual(len(data), 1, msg="expected one waveform in stream")
        self.assertEqual(
            data[0].id, 'TA.O14K..LHZ',
            msg="retrieved incorrect waveform {0}".format(data[0].id))
        # don't forget to test the attach_response

    def test_fedstations_use_existing(self):
        client = FederatedClient()
        frm = client.get_routing(station="ANTO",
                                 starttime=UTCDateTime(2015, 1, 1),
                                 includeoverlaps=True)
        self.assertEqual(len(frm), 2, msg=("Expected to retrieve 2 routes, "
                                           "got %s" % len(frm)))

        orf_frm = FederatedRoutingManager(frm.get_route('ORFEUS'))

        # if this worked, then data will be retrieved from ORFEUS, otherwise
        # it will come from IRIS
        inv = client.get_stations(existing_routes=orf_frm, level="station")
        self.assertTrue(has_level(inv, 'station'),
                        "Should contain station details, but doesn't")

    def test_fedstations_reroute(self):
        """
        test the retry capability of the FederatedClient.get_stations() by
        giving it a bad route (knowing that the data exists elsewhere)
        """
        # find out what the ETH (SED) has on hand
        client = FederatedClient(include_provider='ETH')
        frm = client.get_routing(station="BALST", network="CH",
                                 level="channel")
        self.assertEqual(frm.routes[0].provider_id, 'ETH',
                         msg="Expected provider to be ETH")
        eth_route = frm.get_route('ETH')
        # get some data from IRIS, too
        client = FederatedClient()
        frm = client.get_routing(station="ANTO",
                                 starttime=UTCDateTime(2015, 1, 1))
        iris_route = frm.get_route('IRIS')

        # now, confuse things.
        nodata_route = iris_route
        nodata_route.request_items = eth_route.request_items

        # should come up empty
        try:
            inv = client.get_stations(existing_routes=nodata_route)
            self.assertFalse(inv)
        except FDSNNoDataException:
            pass
        else:
            self.assertTrue(1 == 0, "Expected FDSNNoDataException")

        # should work!
        inv = client.get_stations(existing_routes=nodata_route,
                                  reroute=True)
        self.assertTrue(inv, msg="Rerouting for get_stations failed")

        inv = client.get_stations_bulk(None, existing_routes=nodata_route,
                                       reroute=True)
        self.assertTrue(inv, msg="Rerouting for get_stations_bulk failed")

    def test_fedwaveforms_reroute(self):

        # find out what the ETH (SED) has on hand
        client = FederatedClient(include_provider='ETH')
        frm = client.get_routing(station="BALST", network="CH",
                                 starttime=UTCDateTime(2015, 5, 1, 10, 0, 0),
                                 endtime=UTCDateTime(2015, 5, 1, 10, 2, 0))
        self.assertEqual(frm.routes[0].provider_id, 'ETH')
        eth_route = frm.get_route('ETH')
        # get some data from IRIS, too
        client = FederatedClient()
        frm = client.get_routing(station="ANTO",
                                 starttime=UTCDateTime(2015, 5, 1, 10, 0, 0),
                                 endtime=UTCDateTime(2015, 5, 1, 10, 2, 0))
        iris_route = frm.get_route('IRIS')

        # now, confuse things.
        nodata_route = iris_route
        nodata_route.request_items = eth_route.request_items

        # should come up empty
        try:
            data = client.get_waveforms(None, None, None, None, None, None,
                                        existing_routes=nodata_route)
        except FDSNNoDataException:
            pass
        else:
            self.assertTrue(1 == 0, "Expected FDSNNoDataException")

        # should work!
        data = client.get_waveforms(None, None, None, None, None, None,
                                    existing_routes=nodata_route, reroute=True)
        self.assertTrue(data, msg="Rerouting for get_waveforms failed")

        data = client.get_waveforms_bulk(None, existing_routes=nodata_route,
                                         reroute=True)
        self.assertTrue(data, msg="Rerouting for get_waveforms_bulk failed")

    def test_example_queries_station(self):
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

        '''XXX Because service itself cannot support maxlon<minlon
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
            # test output to stream, with provider already limited to IRIS
            got = client.get_waveforms(*query)
            file_ = os.path.join(self.datapath, filename)
            expected = read(file_)
            self.assertEqual(got, expected, "Dataselect failed for query %s" %
                             repr(query))
            # test output to file
            with NamedTemporaryFile(prefix="IRIS-") as tf:
                base_name = os.path.basename(tf.name)
                base_name = base_name[5:]
                path_name = os.path.dirname(tf.name)
                providerless_name = os.path.join(path_name, base_name)
                print("filename: " + tf.name + " using [" +
                      providerless_name + "]\n\n", file=sys.stderr)
                client.get_waveforms(*query, filename=providerless_name)

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
    from unittest import (TestSuite, makeSuite)
    suite = TestSuite()
    suite.addTest(makeSuite(BulkConversionTestCase, 'test'))
    suite.addTest(makeSuite(FederatedClientTestCase, 'test'))
    suite.addTest(makeSuite(FDSNBulkRequestItemClientTestCase, 'test'))
    return suite


if __name__ == '__main__':
    # unittest.main()
    unittest.TextTestRunner().run(suite())
