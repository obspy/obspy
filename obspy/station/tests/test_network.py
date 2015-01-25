#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the network class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest
import os
import warnings
import numpy as np

from obspy.station import Network, Station, Channel, Response
from obspy import UTCDateTime, read_inventory
from obspy.core.util.base import getBasemapVersion
from obspy.core.util.testing import ImageComparison, getMatplotlibVersion
from obspy.core.util.decorator import skipIf

# checking for matplotlib/basemap
try:
    from matplotlib import rcParams
    import mpl_toolkits.basemap
    # avoid flake8 complaining about unused import
    mpl_toolkits.basemap
    HAS_BASEMAP = True
except ImportError:
    HAS_BASEMAP = False

BASEMAP_VERSION = getBasemapVersion()
MATPLOTLIB_VERSION = getMatplotlibVersion()


class NetworkTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.network.Network` class.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_get_response(self):
        responseN1S1 = Response('RESPN1S1')
        responseN1S2 = Response('RESPN1S2')
        responseN2S1 = Response('RESPN2S1')
        channelsN1S1 = [Channel(code='BHZ',
                                location_code='',
                                latitude=0.0,
                                longitude=0.0,
                                elevation=0.0,
                                depth=0.0,
                                response=responseN1S1)]
        channelsN1S2 = [Channel(code='BHZ',
                                location_code='',
                                latitude=0.0,
                                longitude=0.0,
                                elevation=0.0,
                                depth=0.0,
                                response=responseN1S2)]
        channelsN2S1 = [Channel(code='BHZ',
                                location_code='',
                                latitude=0.0,
                                longitude=0.0,
                                elevation=0.0,
                                depth=0.0,
                                response=responseN2S1)]
        stations1 = [Station(code='N1S1',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN1S1),
                     Station(code='N1S2',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN1S2),
                     Station(code='N2S1',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN2S1)]
        network = Network('N1', stations=stations1)

        response = network.get_response('N1.N1S1..BHZ',
                                        UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN1S1)
        response = network.get_response('N1.N1S2..BHZ',
                                        UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN1S2)
        response = network.get_response('N1.N2S1..BHZ',
                                        UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN2S1)

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

    @skipIf(not HAS_BASEMAP, 'basemap not installed')
    def test_location_plot_cylindrical(self):
        """
        Tests the network location preview plot, default parameters.
        """
        net = read_inventory()[0]
        with ImageComparison(self.image_dir, "network_location1.png") as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(outfile=ic.name)

    @skipIf(not HAS_BASEMAP, 'basemap not installed')
    def test_location_plot_ortho(self):
        """
        Tests the network location preview plot, ortho projection, some
        non-default parameters.
        """
        net = read_inventory()[0]
        with ImageComparison(self.image_dir, "network_location2.png") as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(projection="ortho", resolution="c",
                     continent_fill_color="0.5", marker="d",
                     color="yellow", label=False, outfile=ic.name)

    @skipIf(not HAS_BASEMAP, 'basemap not installed')
    def test_location_plot_local(self):
        """
        Tests the network location preview plot, local projection, some more
        non-default parameters.
        """
        net = read_inventory()[0]
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        reltol = 2.0
        # Basemap smaller 1.0.4 has a serious issue with plotting. Thus the
        # tolerance must be much higher.
        if BASEMAP_VERSION < [1, 0, 4]:
            reltol = 100.0
        with ImageComparison(self.image_dir, "network_location3.png",
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            net.plot(projection="local", resolution="i", size=13**2,
                     outfile=ic.name)

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.2:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 4, 2]:
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


def suite():
    return unittest.makeSuite(NetworkTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
