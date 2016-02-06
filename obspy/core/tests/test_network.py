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

import os
import unittest
import warnings

import numpy as np
from matplotlib import rcParams

from obspy import UTCDateTime, read_inventory
from obspy.core.util.base import get_basemap_version, get_cartopy_version
from obspy.core.util.testing import ImageComparison, get_matplotlib_version
from obspy.core.inventory import Channel, Network, Response, Station


BASEMAP_VERSION = get_basemap_version()
CARTOPY_VERSION = get_cartopy_version()
MATPLOTLIB_VERSION = get_matplotlib_version()


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


@unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
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

    def test_location_plot_global(self):
        """
        Tests the network location preview plot, default parameters, using
        Basemap.
        """
        net = read_inventory()[0]
        reltol = 1.0
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
            net.plot(method='basemap', projection='local', resolution='i',
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
