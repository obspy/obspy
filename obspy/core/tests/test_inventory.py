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
import re
import warnings
from pathlib import Path
from unittest import mock

import pytest

import obspy
from obspy import UTCDateTime, read_inventory, read_events
from obspy.core.util import CARTOPY_VERSION
from obspy.core.util.base import _get_entry_points
from obspy.core.util.testing import WarningsCapture
from obspy.core.inventory import (Channel, Inventory, Network, Response,
                                  Station)
from obspy.core.inventory.util import _unified_content_strings


def sum_stations(inv):
    """
    Count the number of stations in inventory.
    """
    return sum(len(sta) for net in inv for sta in net)


@pytest.mark.usefixtures('ignore_numpy_errors')
class TestInventory:
    """
    Tests the for :class:`~obspy.core.inventory.inventory.Inventory` class.
    """
    # TODO put these into fixtures
    path = os.path.join(os.path.dirname(__file__), 'data')
    station_xml1 = os.path.join(path, 'IU_ANMO_00_BHZ.xml')
    station_xml2 = os.path.join(path, 'IU_ULN_00_LH1.xml')

    def test_initialization(self):
        """
        Some simple sanity tests.
        """
        dt = UTCDateTime()
        inv = Inventory(source="TEST", networks=[])
        # If no time is given, the creation time should be set to the current
        # time. Use a large offset for potentially slow computers and test
        # runs.
        assert inv.created - dt <= 10.0

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
        assert response == response_n1_s1
        response = inv.get_response('N1.N1S2..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        assert response == response_n1_s2
        response = inv.get_response('N2.N2S1..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        assert response == response_n2_s1

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
        assert sorted(coordinates.items()) == sorted(expected.items())
        # 2 - without datetime
        coordinates = inv.get_coordinates('BW.RJOB..EHZ')
        assert sorted(coordinates.items()) == sorted(expected.items())
        # 3 - unknown SEED ID should raise exception
        with pytest.raises(Exception):
            inv.get_coordinates('BW.RJOB..XXX')

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
        assert sorted(orientation.items()) == sorted(expected.items())
        # 2 - without datetime
        orientation = inv.get_orientation('BW.RJOB..EHZ')
        assert sorted(orientation.items()) == sorted(expected.items())
        # 3 - unknown SEED ID should raise exception
        with pytest.raises(Exception):
            inv.get_orientation('BW.RJOB..XXX')

    def test_response_plot(self, image_path):
        """
        Tests the response plot.
        """
        inv = read_inventory()
        t = UTCDateTime(2008, 7, 1)
        with WarningsCapture():
            inv.plot_response(0.01, output="ACC", channel="*N",
                              station="[WR]*", time=t, outfile=image_path)

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
            assert len(texts) == 3
            for text, expected in zip(texts, expecteds):
                assert text.get_text() == expected
        finally:
            plt.close(fig)

    def test_inventory_merging_metadata_update(self):
        """
        Tests the metadata update during merging of inventory objects.
        """
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1 += inv_2

        assert inv_1.source == inv_2.source
        assert inv_1.sender == inv_2.sender
        assert "ObsPy" in inv_1.module
        assert "obspy.org" in inv_1.module_uri
        assert (UTCDateTime() - inv_1.created) < 5

        # Now a more advanced case.
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1.source = "B"
        inv_2.source = "A"

        inv_1.sender = "Random"
        inv_2.sender = "String"

        inv_1 += inv_2

        assert inv_1.source == "A,B"
        assert inv_1.sender == "Random,String"
        assert "ObsPy" in inv_1.module
        assert "obspy.org" in inv_1.module_uri
        assert (UTCDateTime() - inv_1.created) < 5

        # One more. Containing a couple of Nones.
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1.source = None
        inv_2.source = "A"

        inv_1.sender = "Random"
        inv_2.sender = None

        inv_1 += inv_2

        assert inv_1.source == "A"
        assert inv_1.sender == "Random"
        assert "ObsPy" in inv_1.module
        assert "obspy.org" in inv_1.module_uri
        assert (UTCDateTime() - inv_1.created) < 5

    def test_len(self):
        """
        Tests the __len__ property.
        """
        inv = read_inventory()
        assert len(inv) == len(inv.networks)
        assert len(inv) == 2

    def test_inventory_remove(self):
        """
        Test for the Inventory.remove() method.
        """
        inv = read_inventory()

        # Currently contains 30 channels.
        assert sum(len(sta) for net in inv for sta in net) == 30

        # No arguments, everything should be removed, as `None` values left in
        # network/station/location/channel are interpreted as wildcards.
        inv_ = inv.remove()
        assert len(inv_) == 0

        # remove one entire network code
        for network in ['GR', 'G?', 'G*', '?R']:
            inv_ = inv.remove(network=network)
            assert len(inv_) == 1
            assert inv_[0].code == 'BW'
            assert len(inv_[0]) == 3
            for sta in inv_[0]:
                assert len(sta) == 3

        # remove one specific network/station
        for network in ['GR', 'G?', 'G*', '?R']:
            for station in ['FUR', 'F*', 'F??', '*R']:
                inv_ = inv.remove(network=network, station=station)
                assert len(inv_) == 2
                assert inv_[0].code == 'GR'
                assert len(inv_[0]) == 1
                for sta in inv_[0]:
                    assert len(sta) == 9
                    assert sta.code == 'WET'
                assert inv_[1].code == 'BW'
                assert len(inv_[1]) == 3
                for sta in inv_[1]:
                    assert len(sta) == 3
                    assert sta.code == 'RJOB'

        # remove one specific channel
        inv_ = inv.remove(channel='*Z')
        assert len(inv_) == 2
        assert inv_[0].code == 'GR'
        assert len(inv_[0]) == 2
        assert len(inv_[0][0]) == 8
        assert len(inv_[0][1]) == 6
        assert inv_[0][0].code == 'FUR'
        assert inv_[0][1].code == 'WET'
        assert inv_[1].code == 'BW'
        assert len(inv_[1]) == 3
        for sta in inv_[1]:
            assert len(sta) == 2
            assert sta.code == 'RJOB'
        for net in inv_:
            for sta in net:
                for cha in sta:
                    assert cha.code[2] != 'Z'

        # check keep_empty kwarg
        inv_ = inv.remove(station='R*')
        assert len(inv_) == 1
        assert inv_[0].code == 'GR'
        inv_ = inv.remove(station='R*', keep_empty=True)
        assert len(inv_) == 2
        assert inv_[0].code == 'GR'
        assert inv_[1].code == 'BW'
        assert len(inv_[1]) == 0

        inv_ = inv.remove(channel='EH*')
        assert len(inv_) == 1
        assert inv_[0].code == 'GR'
        inv_ = inv.remove(channel='EH*', keep_empty=True)
        assert len(inv_) == 2
        assert inv_[0].code == 'GR'
        assert inv_[1].code == 'BW'
        assert len(inv_[1]) == 3
        for sta in inv_[1]:
            assert sta.code == 'RJOB'
            assert len(sta) == 0

        # some remove calls that don't match anything and should not do
        # anything
        for kwargs in [dict(network='AA'),
                       dict(network='AA', station='FUR'),
                       dict(network='GR', station='ABCD'),
                       dict(network='GR', channel='EHZ')]:
            inv_ = inv.remove(**kwargs)
            assert inv_ == inv

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
        assert len(inv_net.networks) == 1
        # filter by the stations, this should also remove network BW
        inv_sta = copy.deepcopy(inv).remove(station='RJOB')
        assert len(inv_sta.networks) == 1
        assert len(inv_sta.networks[0].stations) == 2
        # but is keep empty is selected network BW should remain
        inv_sta = copy.deepcopy(inv).remove(station='RJOB', keep_empty=True)
        assert len(inv_sta.networks) == 2

    def test_inventory_select(self):
        """
        Test for the Inventory.select() method.
        """
        inv = read_inventory()

        # Currently contains 30 channels.
        assert sum_stations(inv) == 30

        # No arguments, everything should be selected.
        assert sum_stations(inv.select()) == 30

        # All networks.
        assert sum_stations(inv.select(network="*")) == 30

        # All stations.
        assert sum_stations(inv.select(station="*")) == 30

        # All locations.
        assert sum_stations(inv.select(location="*")) == 30

        # All channels.
        assert sum_stations(inv.select(channel="*")) == 30

        # Only BW network.
        assert sum_stations(inv.select(network="BW")) == 9
        assert sum_stations(inv.select(network="B?")) == 9

        # Only RJOB Station.
        assert sum_stations(inv.select(station="RJOB")) == 9
        assert sum_stations(inv.select(station="R?O*")) == 9

        out = inv.select(
            minlatitude=47.5, maxlatitude=47.9,
            minlongitude=11.9, maxlongitude=13.3
        )
        assert sum_stations(out) == 9
        assert sum(len(sta) for net in inv.select(
                latitude=48, longitude=13,
                maxradius=0.5) for sta in net) == \
            9

        # Only WET Station.
        assert sum(len(sta) for net in inv.select(
                latitude=48, longitude=13,
                minradius=0.5, maxradius=1.15) for sta in net) == \
            9

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
        assert p.call_args[1] == select_kwargs

        # Artificially set start-and end dates for the first network.
        inv[0].start_date = UTCDateTime(2000, 1, 1)
        inv[0].end_date = UTCDateTime(2015, 1, 1)

        # Nothing will stick around if keep_empty it False.
        assert len(inv.select(time=UTCDateTime(2001, 1, 1))) == 0
        # If given, both will stick around.
        assert len(inv.select(time=UTCDateTime(2001, 1, 1),
                              keep_empty=True)) == 2
        # Or only one.
        assert len(inv.select(time=UTCDateTime(1999, 1, 1),
                              keep_empty=True)) == 1

        # Also test the starttime and endtime parameters.
        assert len(inv.select(starttime=UTCDateTime(1999, 1, 1),
                              keep_empty=True)) == 2
        assert len(inv.select(starttime=UTCDateTime(2016, 1, 1),
                              keep_empty=True)) == 1
        assert len(inv.select(endtime=UTCDateTime(1999, 1, 1),
                              keep_empty=True)) == 1
        assert len(inv.select(endtime=UTCDateTime(2016, 1, 1),
                              keep_empty=True)) == 2

    def test_inventory_select_with_empty_networks(self):
        """
        Tests the behaviour of the Inventory.select() method with empty
        Network objects.
        """
        inv = read_inventory()

        # Empty all networks.
        for net in inv:
            net.stations = []

        assert len(inv) == 2
        assert sum(len(net) for net in inv) == 0

        # No arguments, everything should be selected.
        assert len(inv) == 2
        # Same if everything is selected.
        assert len(inv.select(network="*")) == 2
        # Select only one.
        assert len(inv.select(network="BW")) == 1
        assert len(inv.select(network="G?")) == 1
        # Should only be empty if trying to select something that does not
        # exist.
        assert len(inv.select(network="RR")) == 0

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
            assert expected_ == _unified_content_strings(contents_)

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
            assert expected_ == _unified_content_strings(contents_)

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
        expected_error_message = re.escape(exception_msg.format(doesnt_exist))
        for format in formats[:1]:
            with pytest.raises(IOError, match=expected_error_message):
                read_inventory(doesnt_exist, format=format)

    def test_inventory_can_be_initialized_with_no_arguments(self):
        """
        Source and networks need not be specified.
        """
        inv = Inventory()
        assert inv.networks == []
        assert inv.source == "ObsPy %s" % obspy.__version__

        # Should also be serializable.
        with io.BytesIO() as buf:
            # This actually would not be a valid StationXML file but there
            # might be uses for this.
            inv.write(buf, format="stationxml")
            buf.seek(0, 0)
            inv2 = read_inventory(buf)

        assert inv == inv2

    def test_copy(self):
        """
        Test for copying inventory.
        """
        inv = read_inventory()
        inv2 = inv.copy()
        assert inv is not inv2
        assert inv == inv2
        # make sure changing inv2 doesnt affect inv
        original_latitude = inv2[0][0][0].latitude
        inv2[0][0][0].latitude = original_latitude + 1
        assert inv[0][0][0].latitude == original_latitude
        assert inv2[0][0][0].latitude == original_latitude + 1
        assert inv[0][0][0].latitude != inv2[0][0][0].latitude

    def test_add(self):
        """
        Test shallow copies for inventory addition
        """
        inv1 = read_inventory()
        inv2 = read_inventory()

        # __add__ creates two shallow copies
        inv_sum = inv1 + inv2
        assert {id(net) for net in inv_sum} == \
               {id(net) for net in inv1} | {id(net) for net in inv2}

        # __iadd__ creates a shallow copy of other and keeps self
        ids1 = {id(net) for net in inv1}
        inv1 += inv2
        assert {id(net) for net in inv1} == ids1 | {id(net) for net in inv2}

        # __add__ with a network appends the network to a shallow copy of
        # the inventory
        net1 = Network('N1')
        inv_sum = inv1 + net1
        assert {id(net) for net in inv_sum} == \
               {id(net) for net in inv1} | {id(net1)}

        # __iadd__ with a network appends the network to the inventory
        net1 = Network('N1')
        ids1 = {id(net) for net in inv1}
        inv1 += net1
        assert {id(net) for net in inv1} == ids1 | {id(net1)}

    def test_extend_metadata(self):
        """
        Test that extend merges the metadata of the Inventories
        """
        inv1 = Inventory([], source='S1', sender='T1')
        inv2 = Inventory([], source='S2', sender='T2')

        inv1.extend(inv2)

        assert inv1.source == 'S1,S2'
        assert inv1.sender == 'T1,T2'

    def test_read_inventory_with_wildcard(self):
        """
        Tests the read_inventory() function with a filename wild card.
        """
        # without wildcard..
        expected = read_inventory(self.station_xml1)
        expected += read_inventory(self.station_xml2)
        # with wildcard
        got = read_inventory(os.path.join(self.path, "IU_*_00*.xml"))
        assert expected == got

    def test_read_inventory_with_path(self):
        """
        Tests that pathlib.Path objects works for input to read_inventory().
        """
        path1 = Path(self.station_xml1)
        inv1 = read_inventory(path1)
        assert inv1 == read_inventory(self.station_xml1)


@pytest.mark.usefixtures('ignore_numpy_errors')
@pytest.mark.skipif(not CARTOPY_VERSION, reason='cartopy not installed')
class TestInventoryCartopy:
    """
    Tests the for :meth:`~obspy.station.inventory.Inventory.plot` with Cartopy.
    """
    image_dir = os.path.join(os.path.dirname(__file__), 'images')

    def test_location_plot_global(self, image_path):
        """
        Tests the inventory location preview plot, default parameters, using
        Cartopy.
        """
        inv = read_inventory()
        inv.plot(method='cartopy', outfile=image_path)

    def test_location_plot_ortho(self, image_path):
        """
        Tests the inventory location preview plot, ortho projection, some
        non-default parameters, using Cartopy.
        """
        inv = read_inventory()
        inv.plot(method='cartopy', projection='ortho', resolution='c',
                 continent_fill_color='0.3', marker='d', label=False,
                 colormap='Set3', color_per_network=True, outfile=image_path)

    def test_location_plot_local(self, image_path):
        """
        Tests the inventory location preview plot, local projection, some more
        non-default parameters, using Cartopy.
        """
        inv = read_inventory()
        inv.plot(method='cartopy', projection='local', resolution='50m',
                 size=20**2, color_per_network={'GR': 'b', 'BW': 'green'},
                 outfile=image_path)

    def test_combined_station_event_plot(self, image_path):
        """
        Tests the combined plotting of inventory/event data in one plot,
        reusing the cartopy instance.
        """
        inv = read_inventory()
        cat = read_events()
        fig = inv.plot(show=False)
        cat.plot(outfile=image_path, fig=fig)
