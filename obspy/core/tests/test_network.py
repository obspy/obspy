#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the network class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
import unittest
import warnings

import numpy as np
from matplotlib import rcParams

import obspy
from obspy import UTCDateTime, read_inventory
from obspy.core.compatibility import mock
from obspy.core.util import (
    BASEMAP_VERSION, CARTOPY_VERSION, MATPLOTLIB_VERSION, PROJ4_VERSION)
from obspy.core.util.testing import ImageComparison
from obspy.core.inventory import (Channel, Inventory, Network, Response,
                                  Station)


class NetworkTestCase(unittest.TestCase):
    """
    Tests for the :class:`~obspy.core.inventory.network.Network` class.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

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
                              channels=channels_n1_s2),
                      Station(code='N2S1',
                              latitude=0.0,
                              longitude=0.0,
                              elevation=0.0,
                              channels=channels_n2_s1)]
        network = Network('N1', stations=stations_1)

        response = network.get_response('N1.N1S1..BHZ',
                                        UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, response_n1_s1)
        response = network.get_response('N1.N1S2..BHZ',
                                        UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, response_n1_s2)
        response = network.get_response('N1.N2S1..BHZ',
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
        network = Network('BW', stations=stations)
        # 1
        coordinates = network.get_coordinates('BW.RJOB..EHZ',
                                              UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(sorted(coordinates.items()), sorted(expected.items()))
        # 2 - without datetime
        coordinates = network.get_coordinates('BW.RJOB..EHZ')
        self.assertEqual(sorted(coordinates.items()), sorted(expected.items()))
        # 3 - unknown SEED ID should raise exception
        self.assertRaises(Exception, network.get_coordinates, 'BW.RJOB..XXX')

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.x:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 5, 0]:
            reltol = 2.0

        net = read_inventory()[0]
        t = UTCDateTime(2008, 7, 1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "network_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                net.plot_response(0.002, output="DISP", channel="B*E",
                                  time=t, outfile=ic.name)

    def test_response_plot_epoch_times_in_label(self):
        """
        Tests response plot with epoch times in labels switched on.
        """
        import matplotlib.pyplot as plt
        net = read_inventory().select(station='RJOB', channel='EHZ')[0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            fig = net.plot_response(0.01, label_epoch_dates=True, show=False)
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

    def test_len(self):
        """
        Tests the __len__ property.
        """
        net = read_inventory()[0]
        self.assertEqual(len(net), len(net.stations))
        self.assertEqual(len(net), 2)

    def test_network_select(self):
        """
        Test for the select() method of the network class.
        """
        net = read_inventory()[0]

        # Basic asserts to assert some things about the test data.
        self.assertEqual(len(net), 2)
        self.assertEqual(len(net[0]), 12)
        self.assertEqual(len(net[1]), 9)
        self.assertEqual(sum(len(i) for i in net), 21)

        # Artificially move the start time of the first station before the
        # channel start times.
        net[0].start_date = UTCDateTime(1999, 1, 1)

        # Nothing happens if nothing is specified or if everything is selected.
        self.assertEqual(sum(len(i) for i in net.select()), 21)
        self.assertEqual(sum(len(i) for i in net.select(station="*")), 21)
        self.assertEqual(sum(len(i) for i in net.select(location="*")), 21)
        self.assertEqual(sum(len(i) for i in net.select(channel="*")), 21)
        self.assertEqual(sum(len(i) for i in net.select(
            station="*", location="*", channel="*")), 21)

        # No matching station.
        self.assertEqual(sum(len(i) for i in net.select(station="RR")), 0)
        # keep_empty does not do anything in these cases.
        self.assertEqual(sum(len(i) for i in
                             net.select(station="RR", keep_empty=True)), 0)
        # Selecting only one station.
        self.assertEqual(sum(len(i) for i in
                             net.select(station="FUR", keep_empty=True)), 12)
        self.assertEqual(sum(len(i) for i in
                             net.select(station="F*", keep_empty=True)), 12)
        self.assertEqual(sum(len(i) for i in
                             net.select(station="WET", keep_empty=True)), 9)
        self.assertEqual(sum(len(i) for i in
                             net.select(
                                minlatitude=47.89, maxlatitude=48.39,
                                minlongitude=10.88, maxlongitude=11.98)), 12)
        self.assertEqual(sum(len(i) for i in
                             net.select(
                                latitude=48.12, longitude=12.24,
                                maxradius=1)), 12)

        # Test the keep_empty flag.
        net_2 = net.select(time=UTCDateTime(2000, 1, 1))
        self.assertEqual(len(net_2), 0)
        self.assertEqual(sum(len(i) for i in net_2), 0)
        # One is kept - it has no more channels but the station still has a
        # valid start time.
        net_2 = net.select(time=UTCDateTime(2000, 1, 1), keep_empty=True)
        self.assertEqual(len(net_2), 1)
        self.assertEqual(sum(len(i) for i in net_2), 0)

        # location, channel, time, starttime, endtime, and sampling_rate
        # and geographic parameters are also passed on to the station selector.
        select_kwargs = {
            "location": "00",
            "channel": "EHE",
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

        with mock.patch("obspy.core.inventory.station.Station.select") as p:
            p.return_value = obspy.core.inventory.station.Station("FUR", 1,
                                                                  2, 3)
            net.select(**select_kwargs)

        self.assertEqual(p.call_args[1], select_kwargs)

    def test_writing_network_before_1990(self):
        inv = obspy.Inventory(networks=[
            Network(code="XX", start_date=obspy.UTCDateTime(1880, 1, 1))],
            source="")
        with io.BytesIO() as buf:
            inv.write(buf, format="stationxml")
            buf.seek(0, 0)
            inv2 = read_inventory(buf)

        self.assertEqual(inv.networks[0], inv2.networks[0])

    def test_network_select_with_empty_stations(self):
        """
        Tests the behaviour of the Network.select() method for empty stations.
        """
        net = read_inventory()[0]

        # Delete all channels.
        for sta in net:
            sta.channels = []

        # 2 stations and 0 channels remain.
        self.assertEqual(len(net), 2)
        self.assertEqual(sum(len(sta) for sta in net), 0)

        # No arguments, everything should be selected.
        self.assertEqual(len(net.select()), 2)

        # Everything selected, nothing should happen.
        self.assertEqual(len(net.select(station="*")), 2)

        # Only select a single station.
        self.assertEqual(len(net.select(station="FUR")), 1)
        self.assertEqual(len(net.select(station="FU?")), 1)
        self.assertEqual(len(net.select(station="W?T")), 1)

        # Once again, this time with the time selection.
        self.assertEqual(len(net.select(time=UTCDateTime(2006, 1, 1))), 0)
        self.assertEqual(len(net.select(time=UTCDateTime(2007, 1, 1))), 1)
        self.assertEqual(len(net.select(time=UTCDateTime(2008, 1, 1))), 2)

    def test_empty_network_code(self):
        """
        Tests that an empty sring is acceptabble.
        """
        # An empty string is allowed.
        n = Network(code="")
        self.assertEqual(n.code, "")

        # But None is not allowed.
        with self.assertRaises(ValueError) as e:
            Network(code=None)
        self.assertEqual(e.exception.args[0], "A code is required")

        # Should still serialize to something.
        inv = Inventory(networks=[n])
        with io.BytesIO() as buf:
            inv.write(buf, format="stationxml", validate=True)
            buf.seek(0, 0)
            inv2 = read_inventory(buf)

        self.assertEqual(inv, inv2)


@unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
@unittest.skipIf(
    BASEMAP_VERSION >= [1, 1, 0] and MATPLOTLIB_VERSION == [3, 0, 1],
    'matplotlib 3.0.1 is not compatible with basemap')
class NetworkBasemapTestCase(unittest.TestCase):
    """
    Tests for the :meth:`~obspy.station.network.Network.plot` with Basemap.
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
        Tests the network location preview plot, default parameters, using
        Basemap.
        """
        net = read_inventory()[0]
        reltol = 1.3
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        if BASEMAP_VERSION < [1, 0, 7]:
            reltol = 3.0
        with ImageComparison(self.image_dir, 'network_location-basemap1.png',
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(method='basemap', outfile=ic.name)

    def test_location_plot_ortho(self):
        """
        Tests the network location preview plot, ortho projection, some
        non-default parameters, using Basemap.
        """
        net = read_inventory()[0]
        with ImageComparison(self.image_dir,
                             'network_location-basemap2.png') as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(method='basemap', projection='ortho', resolution='c',
                     continent_fill_color='0.5', marker='d',
                     color='yellow', label=False, outfile=ic.name)

    def test_location_plot_local(self):
        """
        Tests the network location preview plot, local projection, some more
        non-default parameters, using Basemap.
        """
        net = read_inventory()[0]
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        reltol = 2.0
        # Basemap smaller 1.0.4 has a serious issue with plotting. Thus the
        # tolerance must be much higher.
        if BASEMAP_VERSION < [1, 0, 4]:
            reltol = 100.0
        with ImageComparison(self.image_dir, 'network_location-basemap3.png',
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(method='basemap', projection='local', resolution='l',
                     size=13**2, outfile=ic.name)


@unittest.skipIf(not (CARTOPY_VERSION and CARTOPY_VERSION >= [0, 12, 0]),
                 'cartopy not installed')
class NetworkCartopyTestCase(unittest.TestCase):
    """
    Tests for the :meth:`~obspy.station.network.Network.plot` with Cartopy.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_location_plot_global(self):
        """
        Tests the network location preview plot, default parameters, using
        Cartopy.
        """
        net = read_inventory()[0]
        with ImageComparison(self.image_dir,
                             'network_location-cartopy1.png') as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(method='cartopy', outfile=ic.name)

    def test_location_plot_ortho(self):
        """
        Tests the network location preview plot, ortho projection, some
        non-default parameters, using Cartopy.
        """
        net = read_inventory()[0]
        with ImageComparison(self.image_dir,
                             'network_location-cartopy2.png') as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(method='cartopy', projection='ortho', resolution='c',
                     continent_fill_color='0.5', marker='d',
                     color='yellow', label=False, outfile=ic.name)

    def test_location_plot_local(self):
        """
        Tests the network location preview plot, local projection, some more
        non-default parameters, using Cartopy.
        """
        net = read_inventory()[0]
        with ImageComparison(self.image_dir,
                             'network_location-cartopy3.png') as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(method='cartopy', projection='local', resolution='50m',
                     size=13**2, outfile=ic.name)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NetworkTestCase, 'test'))
    suite.addTest(unittest.makeSuite(NetworkBasemapTestCase, 'test'))
    suite.addTest(unittest.makeSuite(NetworkCartopyTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
