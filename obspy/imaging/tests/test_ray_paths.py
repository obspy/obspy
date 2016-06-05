# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

from obspy import read_inventory, read_events
from obspy.imaging.ray_paths import plot_rays, get_ray_paths


class PathPlottingTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    def setUp(self):
        # load an inventory and an event catalog to test
        # the ray path routines. Careful, the full catalog
        # test is quite long and is therefore commented out
        # by default
        # filedir, filename = os.path.split(__file__)
        # data_path = os.path.join(filedir, 'data', 'IU.xml')
        # self.inventory = read_inventory(data_path)
        # self.catalog = read_events()
        pass

    def test_getraypaths(self):
        # careful, the full inventory, catalog test is long (1min)
        # greatcircles = get_ray_paths(
        #        inventory=self.inventory, catalog=self.catalog,
        #        phase_list=['P'], coordinate_system='XYZ',
        #        taup_model='iasp91')

        # this test checks if we get a single P wave greatcircle
        greatcircles = get_ray_paths(
                stlat=0., stlon=30., evlat=0., evlon=90.,
                evdepth_km=100., phase_list=['P'], coordinate_system='XYZ',
                taup_model='iasp91')
        self.assertEqual(len(greatcircles), 1)
        self.assertEqual(greatcircles[0][1], 'P')

    @unittest.skip('Needs Mayavi to run')
    def test_pathplotting(self):
        # this test uses station lon/lat and event lon/lat/depth input
        # and test the resampling method along the CMB
        plot_rays(station_latitude=0., station_longitude=30,
                  event_latitude=0.,
                  event_longitude=170, event_depth_in_km=200.,
                  phase_list=['Pdiff'], colorscheme='dark', kind='mayavi')

        # uncomment the following to read the global network inventory and
        # a basic catalog that are used by the commented tests:
        #
        filedir, filename = os.path.split(__file__)
        data_path = os.path.join(filedir, 'data', 'IU.xml')
        inventory = read_inventory(data_path)
        catalog = read_events()

        # catalog and inventory test with a phase that doesn't
        # have too many paths (PKIKP):
        #
        plot_rays(inventory=inventory, catalog=catalog,
                  phase_list=['PKIKP'],
                  kind='mayavi', colorscheme='dark')

        # the following test is for an animated mayavi windows
        # and movie plotting.
        # Needs third party coastlines by default !
        #
        # plot_rays(inventory=inv, catalog=cat,
        #           phase_list=['PKP'], animate=True, savemovie=False,
        #           kind='mayavi', figsize=(1920, 1080),
        #           coastlines='data/coastlines.vtk')


def suite():
    return unittest.makeSuite(PathPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
