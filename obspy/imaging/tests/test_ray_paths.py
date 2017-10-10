# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

import obspy
from obspy.imaging.ray_paths import get_ray_paths

import obspy.geodetics.base as geodetics


class PathPlottingTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    def setUp(self):
        # load an inventory and an event catalog to test
        # the ray path routines. Careful, the full catalog
        # test is quite long and is therefore commented out
        # by default
        self.path = os.path.join(os.path.dirname(__file__), 'images')
        pass

    @unittest.skipIf(not geodetics.HAS_GEOGRAPHICLIB,
                     'geographiclib is not installed or doesn\'t run')
    def test_compute_ray_paths(self):
        # careful, the full inventory, catalog test is long (1min)
        # greatcircles = get_ray_paths(
        #        inventory=self.inventory, catalog=self.catalog,
        #        phase_list=['P'], coordinate_system='XYZ',
        #        taup_model='iasp91')

        # this test checks if we get a single P wave greatcircle
        station = obspy.core.inventory.Station(
                    code='STA', latitude=0., longitude=30., elevation=0.)
        network = obspy.core.inventory.Network(
                    code='NET', stations=[station])
        inventory = obspy.core.inventory.Inventory(
                source='ME', networks=[network])

        otime = obspy.UTCDateTime()
        origin = obspy.core.event.Origin(latitude=0., longitude=90.,
                                         depth=100000., time=otime)
        magnitude = obspy.core.event.Magnitude(mag=7.)
        event = obspy.core.event.Event(origins=[origin],
                                       magnitudes=[magnitude])
        catalog = obspy.core.event.Catalog(events=[event])

        greatcircles = get_ray_paths(inventory, catalog, phase_list=['P'],
                                     coordinate_system='XYZ',
                                     taup_model='iasp91')
        self.assertEqual(len(greatcircles), 1)
        self.assertEqual(greatcircles[0][1], 'P')


def suite():
    return unittest.makeSuite(PathPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
