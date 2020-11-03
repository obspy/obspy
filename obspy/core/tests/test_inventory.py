#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the inventory class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import io
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib import rcParams

import obspy
from obspy import UTCDateTime, read_inventory, read_events
from obspy.core.util import (
    BASEMAP_VERSION, CARTOPY_VERSION, MATPLOTLIB_VERSION, PROJ4_VERSION)
from obspy.core.util.base import _get_entry_points
from obspy.core.util.testing import ImageComparison
from obspy.core.inventory import (Channel, Inventory, Network, Response,
                                  Station)
from obspy.core.inventory.util import _unified_content_strings


class InventoryTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.core.inventory.inventory.Inventory` class.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.path = path
        self.station_xml1 = os.path.join(path, 'IU_ANMO_00_BHZ.xml')
        self.station_xml2 = os.path.join(path, 'IU_ULN_00_LH1.xml')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_initialization(self):
        """
        Some simple sanity tests.
        """
        dt = UTCDateTime()
        inv = Inventory(source="TEST", networks=[])
        # If no time is given, the creation time should be set to the current
        # time. Use a large offset for potentially slow computers and test
        # runs.
        self.assertLessEqual(inv.created - dt, 10.0)

    def test_get_response(self):
        response_n1_s1 = Response('RESPN1S1')
        response_n1_s2 = Response('RESPN1S2')
        response_n2_s1 = Response('RESPN2S1')
        channels_n1_s1 = [Channel(code='BHZ',
                                  location_code='',
                                  latitude=0.0,
                                  longitude=0.0,
                                  elevation=0.0,
                                  depth=0.0,
                                  response=response_n1_s1)]
        channels_n1_s2 = [Channel(code='BHZ',
                                  location_code='',
                                  latitude=0.0,
                                  longitude=0.0,
                                  elevation=0.0,
                                  depth=0.0,
                                  response=response_n1_s2)]
        channels_n2_s1 = [Channel(code='BHZ',
                                  location_code='',
                                  latitude=0.0,
                                  longitude=0.0,
                                  elevation=0.0,
                                  depth=0.0,
                                  response=response_n2_s1)]
        stations_1 = [Station(code='N1S1',
                              latitude=0.0,
                              longitude=0.0,
                              elevation=0.0,
                              channels=channels_n1_s1),
                      Station(code='N1S2',
                              latitude=0.0,
                              longitude=0.0,
                              elevation=0.0,
                              channels=channels_n1_s2)]
        stations_2 = [Station(code='N2S1',
                              latitude=0.0,
                              longitude=0.0,
                              elevation=0.0,
                              channels=channels_n2_s1)]
        networks = [Network('N1', stations=stations_1),
                    Network('N2', stations=stations_2)]
        inv = Inventory(networks=networks, source='TEST')

        response = inv.get_response('N1.N1S1..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, response_n1_s1)
        response = inv.get_response('N1.N1S2..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, response_n1_s2)
        response = inv.get_response('N2.N2S1..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, response_n2_s1)

    def test_get_coordinates(self):
        """
        Test extracting coordinates
        """
        expected = {u'latitude': 47.737166999999999,
                    u'longitude': 12.795714,
                    u'elevation': 860.0,
                    u'local_depth': 0.0}
        channels = [Channel(code='EHZ',
                            location_code='',
                            start_date=UTCDateTime('2007-01-01'),
                            latitude=47.737166999999999,
                            longitude=12.795714,
                            elevation=860.0,
                            depth=0.0)]
        stations = [Station(code='RJOB',
                            latitude=0.0,
                            longitude=0.0,
                            elevation=0.0,
                            channels=channels)]
        networks = [Network('BW', stations=stations)]
        inv = Inventory(networks=networks, source='TEST')
        # 1
        coordinates = inv.get_coordinates('BW.RJOB..EHZ',
                                          UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(sorted(coordinates.items()), sorted(expected.items()))
        # 2 - without datetime
        coordinates = inv.get_coordinates('BW.RJOB..EHZ')
        self.assertEqual(sorted(coordinates.items()), sorted(expected.items()))
        # 3 - unknown SEED ID should raise exception
        self.assertRaises(Exception, inv.get_coordinates, 'BW.RJOB..XXX')

    def test_get_orientation(self):
        """
        Test extracting orientation
        """
        expected = {u'azimuth': 90.0,
                    u'dip': 0.0}
        channels = [Channel(code='EHZ',
                            location_code='',
                            start_date=UTCDateTime('2007-01-01'),
                            latitude=47.737166999999999,
                            longitude=12.795714,
                            elevation=860.0,
                            depth=0.0,
                            azimuth=90.0,
                            dip=0.0)]
        stations = [Station(code='RJOB',
                            latitude=0.0,
                            longitude=0.0,
                            elevation=0.0,
                            channels=channels)]
        networks = [Network('BW', stations=stations)]
        inv = Inventory(networks=networks, source='TEST')
        # 1
        orientation = inv.get_orientation('BW.RJOB..EHZ',
                                          UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(sorted(orientation.items()), sorted(expected.items()))
        # 2 - without datetime
        orientation = inv.get_orientation('BW.RJOB..EHZ')
        self.assertEqual(sorted(orientation.items()), sorted(expected.items()))
        # 3 - unknown SEED ID should raise exception
        self.assertRaises(Exception, inv.get_orientation, 'BW.RJOB..XXX')

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        inv = read_inventory()
        t = UTCDateTime(2008, 7, 1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir,
                                 "inventory_response.png") as ic:
                rcParams['savefig.dpi'] = 72
                inv.plot_response(0.01, output="ACC", channel="*N",
                                  station="[WR]*", time=t, outfile=ic.name)

    def test_response_plot_epoch_times_in_label(self):
        """
        Tests response plot with epoch times in labels switched on.
        """
        import matplotlib.pyplot as plt
        inv = read_inventory().select(station='RJOB', channel='EHZ')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            fig = inv.plot_response(0.01, label_epoch_dates=True, show=False)
        try:
            legend = fig.axes[0].get_legend()
            texts = legend.get_texts()
            expecteds = ['BW.RJOB..EHZ\n2001-05-15 -- 2006-12-12',
                         'BW.RJOB..EHZ\n2006-12-13 -- 2007-12-17',
                         'BW.RJOB..EHZ\n2007-12-17 -- open']
            self.assertEqual(len(texts), 3)
            for text, expected in zip(texts, expecteds):
                self.assertEqual(text.get_text(), expected)
        finally:
            plt.close(fig)

    def test_inventory_merging_metadata_update(self):
        """
        Tests the metadata update during merging of inventory objects.
        """
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1 += inv_2

        self.assertEqual(inv_1.source, inv_2.source)
        self.assertEqual(inv_1.sender, inv_2.sender)
        self.assertIn("ObsPy", inv_1.module)
        self.assertIn("obspy.org", inv_1.module_uri)
        self.assertTrue((UTCDateTime() - inv_1.created) < 5)

        # Now a more advanced case.
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1.source = "B"
        inv_2.source = "A"

        inv_1.sender = "Random"
        inv_2.sender = "String"

        inv_1 += inv_2

        self.assertEqual(inv_1.source, "A,B")
        self.assertEqual(inv_1.sender, "Random,String")
        self.assertIn("ObsPy", inv_1.module)
        self.assertIn("obspy.org", inv_1.module_uri)
        self.assertTrue((UTCDateTime() - inv_1.created) < 5)

        # One more. Containing a couple of Nones.
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1.source = None
        inv_2.source = "A"

        inv_1.sender = "Random"
        inv_2.sender = None

        inv_1 += inv_2

        self.assertEqual(inv_1.source, "A")
        self.assertEqual(inv_1.sender, "Random")
        self.assertIn("ObsPy", inv_1.module)
        self.assertIn("obspy.org", inv_1.module_uri)
        self.assertTrue((UTCDateTime() - inv_1.created) < 5)

    def test_len(self):
        """
        Tests the __len__ property.
        """
        inv = read_inventory()
        self.assertEqual(len(inv), len(inv.networks))
        self.assertEqual(len(inv), 2)

    def test_inventory_remove(self):
        """
        Test for the Inventory.remove() method.
        """
        inv = read_inventory()

        # Currently contains 30 channels.
        self.assertEqual(sum(len(sta) for net in inv for sta in net), 30)

        # No arguments, everything should be removed, as `None` values left in
        # network/station/location/channel are interpreted as wildcards.
        inv_ = inv.remove()
        self.assertEqual(len(inv_), 0)

        # remove one entire network code
        for network in ['GR', 'G?', 'G*', '?R']:
            inv_ = inv.remove(network=network)
            self.assertEqual(len(inv_), 1)
            self.assertEqual(inv_[0].code, 'BW')
            self.assertEqual(len(inv_[0]), 3)
            for sta in inv_[0]:
                self.assertEqual(len(sta), 3)

        # remove one specific network/station
        for network in ['GR', 'G?', 'G*', '?R']:
            for station in ['FUR', 'F*', 'F??', '*R']:
                inv_ = inv.remove(network=network, station=station)
                self.assertEqual(len(inv_), 2)
                self.assertEqual(inv_[0].code, 'GR')
                self.assertEqual(len(inv_[0]), 1)
                for sta in inv_[0]:
                    self.assertEqual(len(sta), 9)
                    self.assertEqual(sta.code, 'WET')
                self.assertEqual(inv_[1].code, 'BW')
                self.assertEqual(len(inv_[1]), 3)
                for sta in inv_[1]:
                    self.assertEqual(len(sta), 3)
                    self.assertEqual(sta.code, 'RJOB')

        # remove one specific channel
        inv_ = inv.remove(channel='*Z')
        self.assertEqual(len(inv_), 2)
        self.assertEqual(inv_[0].code, 'GR')
        self.assertEqual(len(inv_[0]), 2)
        self.assertEqual(len(inv_[0][0]), 8)
        self.assertEqual(len(inv_[0][1]), 6)
        self.assertEqual(inv_[0][0].code, 'FUR')
        self.assertEqual(inv_[0][1].code, 'WET')
        self.assertEqual(inv_[1].code, 'BW')
        self.assertEqual(len(inv_[1]), 3)
        for sta in inv_[1]:
            self.assertEqual(len(sta), 2)
            self.assertEqual(sta.code, 'RJOB')
        for net in inv_:
            for sta in net:
                for cha in sta:
                    self.assertTrue(cha.code[2] != 'Z')

        # check keep_empty kwarg
        inv_ = inv.remove(station='R*')
        self.assertEqual(len(inv_), 1)
        self.assertEqual(inv_[0].code, 'GR')
        inv_ = inv.remove(station='R*', keep_empty=True)
        self.assertEqual(len(inv_), 2)
        self.assertEqual(inv_[0].code, 'GR')
        self.assertEqual(inv_[1].code, 'BW')
        self.assertEqual(len(inv_[1]), 0)

        inv_ = inv.remove(channel='EH*')
        self.assertEqual(len(inv_), 1)
        self.assertEqual(inv_[0].code, 'GR')
        inv_ = inv.remove(channel='EH*', keep_empty=True)
        self.assertEqual(len(inv_), 2)
        self.assertEqual(inv_[0].code, 'GR')
        self.assertEqual(inv_[1].code, 'BW')
        self.assertEqual(len(inv_[1]), 3)
        for sta in inv_[1]:
            self.assertEqual(sta.code, 'RJOB')
            self.assertEqual(len(sta), 0)

        # some remove calls that don't match anything and should not do
        # anything
        for kwargs in [dict(network='AA'),
                       dict(network='AA', station='FUR'),
                       dict(network='GR', station='ABCD'),
                       dict(network='GR', channel='EHZ')]:
            inv_ = inv.remove(**kwargs)
            self.assertEqual(inv_, inv)

    def test_issue_2266(self):
        """
        Ensure the remove method works for more than just channel level
        inventories. See #2266.
        """
        # get inventory and remove all channel level info
        inv = obspy.read_inventory()
        for net in inv:
            for sta in net:
                sta.channels = []
        # filter by one of the networks
        inv_net = copy.deepcopy(inv).remove(network='BW')
        self.assertEqual(len(inv_net.networks), 1)
        # filter by the stations, this should also remove network BW
        inv_sta = copy.deepcopy(inv).remove(station='RJOB')
        self.assertEqual(len(inv_sta.networks), 1)
        self.assertEqual(len(inv_sta.networks[0].stations), 2)
        # but is keep empty is selected network BW should remain
        inv_sta = copy.deepcopy(inv).remove(station='RJOB', keep_empty=True)
        self.assertEqual(len(inv_sta.networks), 2)

    def test_inventory_select(self):
        """
        Test for the Inventory.select() method.
        """
        inv = read_inventory()

        # Currently contains 30 channels.
        self.assertEqual(sum(len(sta) for net in inv for sta in net), 30)

        # No arguments, everything should be selected.
        self.assertEqual(
            sum(len(sta) for net in inv.select() for sta in net),
            30)

        # All networks.
        self.assertEqual(
            sum(len(sta) for net in inv.select(network="*") for sta in net),
            30)

        # All stations.
        self.assertEqual(
            sum(len(sta) for net in inv.select(station="*") for sta in net),
            30)

        # All locations.
        self.assertEqual(
            sum(len(sta) for net in inv.select(location="*") for sta in net),
            30)

        # All channels.
        self.assertEqual(
            sum(len(sta) for net in inv.select(channel="*") for sta in net),
            30)

        # Only BW network.
        self.assertEqual(
            sum(len(sta) for net in inv.select(network="BW") for sta in net),
            9)
        self.assertEqual(
            sum(len(sta) for net in inv.select(network="B?") for sta in net),
            9)

        # Only RJOB Station.
        self.assertEqual(
            sum(len(sta) for net in inv.select(station="RJOB") for sta in net),
            9)
        self.assertEqual(
            sum(len(sta) for net in inv.select(station="R?O*") for sta in net),
            9)
        self.assertEqual(
            sum(len(sta) for net in inv.select(
                minlatitude=47.5, maxlatitude=47.9,
                minlongitude=11.9, maxlongitude=13.3) for sta in net),
            9)
        self.assertEqual(
            sum(len(sta) for net in inv.select(
                latitude=48, longitude=13,
                maxradius=0.5) for sta in net),
            9)

        # Only WET Station.
        self.assertEqual(
            sum(len(sta) for net in inv.select(
                latitude=48, longitude=13,
                minradius=0.5, maxradius=1.15) for sta in net),
            9)

        # Most parameters are just passed to the Network.select() method.
        select_kwargs = {
            "station": "BW",
            "location": "00",
            "channel": "EHE",
            "keep_empty": True,
            "time": UTCDateTime(2001, 1, 1),
            "sampling_rate": 123.0,
            "starttime": UTCDateTime(2002, 1, 1),
            "endtime": UTCDateTime(2003, 1, 1),
            "minlatitude": None,
            "maxlatitude": None,
            "minlongitude": None,
            "maxlongitude": None,
            "latitude": None,
            "longitude": None,
            "minradius": None,
            "maxradius": None}
        with mock.patch("obspy.core.inventory.network.Network.select") as p:
            p.return_value = obspy.core.inventory.network.Network("BW")
            inv.select(**select_kwargs)
        self.assertEqual(p.call_args[1], select_kwargs)

        # Artificially set start-and end dates for the first network.
        inv[0].start_date = UTCDateTime(2000, 1, 1)
        inv[0].end_date = UTCDateTime(2015, 1, 1)

        # Nothing will stick around if keep_empty it False.
        self.assertEqual(len(inv.select(time=UTCDateTime(2001, 1, 1))), 0)
        # If given, both will stick around.
        self.assertEqual(len(inv.select(time=UTCDateTime(2001, 1, 1),
                                        keep_empty=True)), 2)
        # Or only one.
        self.assertEqual(len(inv.select(time=UTCDateTime(1999, 1, 1),
                                        keep_empty=True)), 1)

        # Also test the starttime and endtime parameters.
        self.assertEqual(len(inv.select(starttime=UTCDateTime(1999, 1, 1),
                                        keep_empty=True)), 2)
        self.assertEqual(len(inv.select(starttime=UTCDateTime(2016, 1, 1),
                                        keep_empty=True)), 1)
        self.assertEqual(len(inv.select(endtime=UTCDateTime(1999, 1, 1),
                                        keep_empty=True)), 1)
        self.assertEqual(len(inv.select(endtime=UTCDateTime(2016, 1, 1),
                                        keep_empty=True)), 2)

    def test_inventory_select_with_empty_networks(self):
        """
        Tests the behaviour of the Inventory.select() method with empty
        Network objects.
        """
        inv = read_inventory()

        # Empty all networks.
        for net in inv:
            net.stations = []

        self.assertEqual(len(inv), 2)
        self.assertEqual(sum(len(net) for net in inv), 0)

        # No arguments, everything should be selected.
        self.assertEqual(len(inv), 2)
        # Same if everything is selected.
        self.assertEqual(len(inv.select(network="*")), 2)
        # Select only one.
        self.assertEqual(len(inv.select(network="BW")), 1)
        self.assertEqual(len(inv.select(network="G?")), 1)
        # Should only be empty if trying to select something that does not
        # exist.
        self.assertEqual(len(inv.select(network="RR")), 0)

    def test_util_unified_content_string(self):
        """
        Tests helper routine that compresses inventory content lists.
        """
        contents = (
            [u'IU.ULN (Ulaanbaatar, Mongolia)',
             u'IU.ULN (Ulaanbaatar, Mongolia)',
             u'IU.ULN (Ulaanbaatar, Mongolia)'],
            [u'IU.ULN.00.BH1', u'IU.ULN.00.BH2', u'IU.ULN.00.BHE',
             u'IU.ULN.00.BHE', u'IU.ULN.00.BHE', u'IU.ULN.00.BHE',
             u'IU.ULN.00.BHN', u'IU.ULN.00.BHN', u'IU.ULN.00.BHN',
             u'IU.ULN.00.BHN', u'IU.ULN.00.BHZ', u'IU.ULN.00.BHZ',
             u'IU.ULN.00.BHZ', u'IU.ULN.00.BHZ', u'IU.ULN.00.BHZ',
             u'IU.ULN.00.LH1', u'IU.ULN.00.LH2', u'IU.ULN.00.LHE',
             u'IU.ULN.00.LHE', u'IU.ULN.00.LHE', u'IU.ULN.00.LHE',
             u'IU.ULN.00.LHN', u'IU.ULN.00.LHN', u'IU.ULN.00.LHN',
             u'IU.ULN.00.LHN', u'IU.ULN.00.LHZ', u'IU.ULN.00.LHZ',
             u'IU.ULN.00.LHZ', u'IU.ULN.00.LHZ', u'IU.ULN.00.LHZ',
             u'IU.ULN.00.UHE', u'IU.ULN.00.UHE', u'IU.ULN.00.UHN',
             u'IU.ULN.00.UHN', u'IU.ULN.00.UHZ', u'IU.ULN.00.UHZ',
             u'IU.ULN.00.VE1', u'IU.ULN.00.VE1', u'IU.ULN.00.VH1',
             u'IU.ULN.00.VH2', u'IU.ULN.00.VHE', u'IU.ULN.00.VHE',
             u'IU.ULN.00.VHE', u'IU.ULN.00.VHE', u'IU.ULN.00.VHN',
             u'IU.ULN.00.VHN', u'IU.ULN.00.VHN', u'IU.ULN.00.VHN',
             u'IU.ULN.00.VHZ', u'IU.ULN.00.VHZ', u'IU.ULN.00.VHZ',
             u'IU.ULN.00.VHZ', u'IU.ULN.00.VHZ', u'IU.ULN.00.VK1',
             u'IU.ULN.00.VK1', u'IU.ULN.00.VM1', u'IU.ULN.00.VM2',
             u'IU.ULN.00.VME', u'IU.ULN.00.VME', u'IU.ULN.00.VMN',
             u'IU.ULN.00.VMN', u'IU.ULN.00.VMZ', u'IU.ULN.00.VMZ',
             u'IU.ULN.00.VMZ'])
        expected = (
            [u'IU.ULN (Ulaanbaatar, Mongolia) (3x)'],
            [u'IU.ULN.00.BHZ (5x)', u'IU.ULN.00.BHN (4x)',
             u'IU.ULN.00.BHE (4x)', u'IU.ULN.00.BH1', u'IU.ULN.00.BH2',
             u'IU.ULN.00.LHZ (5x)', u'IU.ULN.00.LHN (4x)',
             u'IU.ULN.00.LHE (4x)', u'IU.ULN.00.LH1', u'IU.ULN.00.LH2',
             u'IU.ULN.00.UHZ (2x)', u'IU.ULN.00.UHN (2x)',
             u'IU.ULN.00.UHE (2x)', u'IU.ULN.00.VE1 (2x)',
             u'IU.ULN.00.VHZ (5x)', u'IU.ULN.00.VHN (4x)',
             u'IU.ULN.00.VHE (4x)', u'IU.ULN.00.VH1', u'IU.ULN.00.VH2',
             u'IU.ULN.00.VK1 (2x)', u'IU.ULN.00.VMZ (3x)',
             u'IU.ULN.00.VMN (2x)', u'IU.ULN.00.VME (2x)', u'IU.ULN.00.VM1',
             u'IU.ULN.00.VM2'])
        for contents_, expected_ in zip(contents, expected):
            self.assertEqual(expected_, _unified_content_strings(contents_))

    def test_util_unified_content_string_with_dots_in_description(self):
        """
        The unified content string might have dots in the station description.

        Make sure it still works.
        """
        contents = (
            ['II.ABKT (Alibek, Turkmenistan)',
             'II.ALE (Alert, N.W.T., Canada)'],
            [u'IU.ULN (Ulaanbaatar, A.B.C., Mongolia)',
             u'IU.ULN (Ulaanbaatar, A.B.C., Mongolia)',
             u'IU.ULN (Ulaanbaatar, A.B.C., Mongolia)'],
        )
        expected = (
            ['II.ABKT (Alibek, Turkmenistan)',
             'II.ALE (Alert, N.W.T., Canada)'],
            [u'IU.ULN (Ulaanbaatar, A.B.C., Mongolia) (3x)'],
        )
        for contents_, expected_ in zip(contents, expected):
            self.assertEqual(expected_, _unified_content_strings(contents_))

    def test_read_invalid_filename(self):
        """
        Tests that we get a sane error message when calling read_inventory()
        with a filename that doesn't exist
        """
        doesnt_exist = 'dsfhjkfs'
        for i in range(10):
            if os.path.exists(doesnt_exist):
                doesnt_exist += doesnt_exist
                continue
            break
        else:
            self.fail('unable to get invalid file path')

        exception_msg = "[Errno 2] No such file or directory: '{}'"

        formats = _get_entry_points(
            'obspy.plugin.inventory', 'readFormat').keys()
        # try read_inventory() with invalid filename for all registered read
        # plugins and also for filetype autodiscovery
        formats = [None] + list(formats)
        for format in formats[:1]:
            with self.assertRaises(IOError) as e:
                read_inventory(doesnt_exist, format=format)
            self.assertEqual(
                str(e.exception), exception_msg.format(doesnt_exist))

    def test_inventory_can_be_initialized_with_no_arguments(self):
        """
        Source and networks need not be specified.
        """
        inv = Inventory()
        self.assertEqual(inv.networks, [])
        self.assertEqual(inv.source, "ObsPy %s" % obspy.__version__)

        # Should also be serializable.
        with io.BytesIO() as buf:
            # This actually would not be a valid StationXML file but there
            # might be uses for this.
            inv.write(buf, format="stationxml")
            buf.seek(0, 0)
            inv2 = read_inventory(buf)

        self.assertEqual(inv, inv2)

    def test_copy(self):
        """
        Test for copying inventory.
        """
        inv = read_inventory()
        inv2 = inv.copy()
        self.assertIsNot(inv, inv2)
        self.assertEqual(inv, inv2)
        # make sure changing inv2 doesnt affect inv
        original_latitude = inv2[0][0][0].latitude
        inv2[0][0][0].latitude = original_latitude + 1
        self.assertEqual(inv[0][0][0].latitude, original_latitude)
        self.assertEqual(inv2[0][0][0].latitude, original_latitude + 1)
        self.assertNotEqual(inv[0][0][0].latitude, inv2[0][0][0].latitude)

    def test_add(self):
        """
        Test shallow copies for inventory addition
        """
        inv1 = read_inventory()
        inv2 = read_inventory()

        # __add__ creates two shallow copies
        inv_sum = inv1 + inv2
        self.assertEqual({id(net) for net in inv_sum},
                         {id(net) for net in inv1} | {id(net) for net in inv2})

        # __iadd__ creates a shallow copy of other and keeps self
        ids1 = {id(net) for net in inv1}
        inv1 += inv2
        self.assertEqual({id(net) for net in inv1},
                         ids1 | {id(net) for net in inv2})

        # __add__ with a network appends the network to a shallow copy of
        # the inventory
        net1 = Network('N1')
        inv_sum = inv1 + net1
        self.assertEqual({id(net) for net in inv_sum},
                         {id(net) for net in inv1} | {id(net1)})

        # __iadd__ with a network appends the network to the inventory
        net1 = Network('N1')
        ids1 = {id(net) for net in inv1}
        inv1 += net1
        self.assertEqual({id(net) for net in inv1}, ids1 | {id(net1)})

    def test_extend_metadata(self):
        """
        Test that extend merges the metadata of the Inventories
        """
        inv1 = Inventory([], source='S1', sender='T1')
        inv2 = Inventory([], source='S2', sender='T2')

        inv1.extend(inv2)

        self.assertEqual(inv1.source, 'S1,S2')
        self.assertEqual(inv1.sender, 'T1,T2')

    def test_read_inventory_with_wildcard(self):
        """
        Tests the read_inventory() function with a filename wild card.
        """
        # without wildcard..
        expected = read_inventory(self.station_xml1)
        expected += read_inventory(self.station_xml2)
        # with wildcard
        got = read_inventory(os.path.join(self.path, "IU_*_00*.xml"))
        self.assertEqual(expected, got)

    def test_read_inventory_with_path(self):
        """
        Tests that pathlib.Path objects works for input to read_inventory().
        """
        path1 = Path(self.station_xml1)
        inv1 = read_inventory(path1)
        self.assertEqual(inv1, read_inventory(self.station_xml1))


@unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
@unittest.skipIf(
    BASEMAP_VERSION or [] >= [1, 1, 0] and MATPLOTLIB_VERSION == [3, 0, 1],
    'matplotlib 3.0.1 is not compatible with basemap')
class InventoryBasemapTestCase(unittest.TestCase):
    """
    Tests the for :meth:`~obspy.station.inventory.Inventory.plot` with Basemap.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    @unittest.skipIf(PROJ4_VERSION and PROJ4_VERSION[0] == 5,
                     'unsupported proj4 library')
    def test_location_plot_global(self):
        """
        Tests the inventory location preview plot, default parameters, using
        Basemap.
        """
        inv = read_inventory()
        reltol = 1.3
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        if BASEMAP_VERSION < [1, 0, 7]:
            reltol = 3.0
        with ImageComparison(self.image_dir, 'inventory_location-basemap1.png',
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(outfile=ic.name)

    def test_location_plot_ortho(self):
        """
        Tests the inventory location preview plot, ortho projection, some
        non-default parameters, using Basemap.
        """
        inv = read_inventory()
        with ImageComparison(self.image_dir,
                             'inventory_location-basemap2.png') as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(method='basemap', projection='ortho', resolution='c',
                     continent_fill_color='0.3', marker='d', label=False,
                     colormap='Set3', color_per_network=True, outfile=ic.name)

    def test_location_plot_local(self):
        """
        Tests the inventory location preview plot, local projection, some more
        non-default parameters, using Basemap.
        """
        inv = read_inventory()
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        reltol = 2.0
        # Basemap smaller 1.0.4 has a serious issue with plotting. Thus the
        # tolerance must be much higher.
        if BASEMAP_VERSION < [1, 0, 4]:
            reltol = 100.0
        with ImageComparison(self.image_dir, 'inventory_location-basemap3.png',
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(method='basemap', projection='local', resolution='l',
                     size=20**2, color_per_network={'GR': 'b', 'BW': 'green'},
                     outfile=ic.name)

    @unittest.skipIf(PROJ4_VERSION and PROJ4_VERSION[0] == 5,
                     'unsupported proj4 library')
    def test_combined_station_event_plot(self):
        """
        Tests the combined plotting of inventory/event data in one plot,
        reusing the basemap instance.
        """
        inv = read_inventory()
        cat = read_events()
        reltol = 1.1
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        if BASEMAP_VERSION < [1, 0, 7]:
            reltol = 3.0
        with ImageComparison(self.image_dir,
                             'basemap_combined_stations-events.png',
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            fig = inv.plot(show=False)
            cat.plot(outfile=ic.name, fig=fig)


@unittest.skipIf(not (CARTOPY_VERSION and CARTOPY_VERSION >= [0, 12, 0]),
                 'cartopy not installed')
class InventoryCartopyTestCase(unittest.TestCase):
    """
    Tests the for :meth:`~obspy.station.inventory.Inventory.plot` with Cartopy.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_location_plot_global(self):
        """
        Tests the inventory location preview plot, default parameters, using
        Cartopy.
        """
        inv = read_inventory()
        with ImageComparison(self.image_dir,
                             'inventory_location-cartopy1.png') as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(method='cartopy', outfile=ic.name)

    def test_location_plot_ortho(self):
        """
        Tests the inventory location preview plot, ortho projection, some
        non-default parameters, using Cartopy.
        """
        inv = read_inventory()
        with ImageComparison(self.image_dir,
                             'inventory_location-cartopy2.png') as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(method='cartopy', projection='ortho', resolution='c',
                     continent_fill_color='0.3', marker='d', label=False,
                     colormap='Set3', color_per_network=True, outfile=ic.name)

    def test_location_plot_local(self):
        """
        Tests the inventory location preview plot, local projection, some more
        non-default parameters, using Cartopy.
        """
        inv = read_inventory()
        with ImageComparison(self.image_dir,
                             'inventory_location-cartopy3.png') as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(method='cartopy', projection='local', resolution='50m',
                     size=20**2, color_per_network={'GR': 'b', 'BW': 'green'},
                     outfile=ic.name)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(InventoryTestCase, 'test'))
    suite.addTest(unittest.makeSuite(InventoryBasemapTestCase, 'test'))
    suite.addTest(unittest.makeSuite(InventoryCartopyTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
