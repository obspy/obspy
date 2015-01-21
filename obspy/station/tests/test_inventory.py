#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the inventory class.

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
import numpy as np

from obspy.station import Inventory, Network, Station, Channel, Response
from obspy import UTCDateTime, read_inventory
from obspy.core.util.base import getBasemapVersion
from obspy.core.util.testing import ImageComparison, getMatplotlibVersion
from obspy.core.util.decorator import skipIf
import warnings

# checking for matplotlib/basemap
try:
    from matplotlib import rcParams
    import mpl_toolkits.basemap
    # avoid flake8 complaining about unused import
    mpl_toolkits.basemap
    HAS_BASEMAP = True
except ImportError:
    HAS_BASEMAP = False

MATPLOTLIB_VERSION = getMatplotlibVersion()
BASEMAP_VERSION = getBasemapVersion()


class InventoryTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.inventory.Inventory` class.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

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
        self.assertTrue(inv.created - dt <= 10.0)

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
                             channels=channelsN1S2)]
        stations2 = [Station(code='N2S1',
                             latitude=0.0,
                             longitude=0.0,
                             elevation=0.0,
                             channels=channelsN2S1)]
        networks = [Network('N1', stations=stations1),
                    Network('N2', stations=stations2)]
        inv = Inventory(networks=networks, source='TEST')

        response = inv.get_response('N1.N1S1..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN1S1)
        response = inv.get_response('N1.N1S2..BHZ',
                                    UTCDateTime('2010-01-01T12:00'))
        self.assertEqual(response, responseN1S2)
        response = inv.get_response('N2.N2S1..BHZ',
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

    @skipIf(not HAS_BASEMAP, 'basemap not installed')
    def test_location_plot_cylindrical(self):
        """
        Tests the inventory location preview plot, default parameters.
        """
        inv = read_inventory()
        with ImageComparison(self.image_dir, "inventory_location1.png") as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(outfile=ic.name)

    @skipIf(not HAS_BASEMAP, 'basemap not installed')
    def test_location_plot_ortho(self):
        """
        Tests the inventory location preview plot, ortho projection, some
        non-default parameters.
        """
        inv = read_inventory()
        with ImageComparison(self.image_dir, "inventory_location2.png") as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(projection="ortho", resolution="c",
                     continent_fill_color="0.3", marker="d",
                     label=False, outfile=ic.name, colormap="hsv",
                     color_per_network=True)

    @skipIf(not HAS_BASEMAP, 'basemap not installed')
    def test_location_plot_local(self):
        """
        Tests the inventory location preview plot, local projection, some more
        non-default parameters.
        """
        inv = read_inventory()
        # Coordinate lines might be slightly off, depending on the basemap
        # version.
        reltol = 2.0
        # Basemap smaller 1.0.4 has a serious issue with plotting. Thus the
        # tolerance must be much higher.
        if BASEMAP_VERSION < [1, 0, 4]:
            reltol = 100.0
        with ImageComparison(self.image_dir, "inventory_location3.png",
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            inv.plot(projection="local", resolution="i", size=20**2,
                     color_per_network={"GR": "b", "BW": "green"},
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

        inv = read_inventory()
        t = UTCDateTime(2008, 7, 1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "inventory_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                inv.plot_response(0.01, output="ACC", channel="*N",
                                  station="[WR]*", time=t, outfile=ic.name)

    def test_inventory_merging_metadata_update(self):
        """
        Tests the metadata update during merging of inventory objects.
        """
        inv_1 = read_inventory()
        inv_2 = read_inventory()

        inv_1 += inv_2

        self.assertEqual(inv_1.source, inv_2.source)
        self.assertEqual(inv_1.sender, inv_2.sender)
        self.assertTrue("ObsPy" in inv_1.module)
        self.assertTrue("obspy.org" in inv_1.module_uri)
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
        self.assertTrue("ObsPy" in inv_1.module)
        self.assertTrue("obspy.org" in inv_1.module_uri)
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
        self.assertTrue("ObsPy" in inv_1.module)
        self.assertTrue("obspy.org" in inv_1.module_uri)
        self.assertTrue((UTCDateTime() - inv_1.created) < 5)


def suite():
    return unittest.makeSuite(InventoryTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
