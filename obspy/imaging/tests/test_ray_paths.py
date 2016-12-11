# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import os

from obspy.core.util.testing import ImageComparison
from obspy import read_inventory, read_events
from obspy.imaging.ray_paths import plot_rays, get_ray_paths


try:
    from mayavi import mlab  # @UnusedImport # NOQA
    HAS_MAYAVI = True
except ImportError:
    HAS_MAYAVI = False


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

    @unittest.skipIf(not HAS_MAYAVI,
                     'Module mayavi is not installed or doesn\'t run')
    def test_path_plotting(self):
        # uncomment the following to read the global network inventory and
        # a basic catalog that are used by the commented tests:
        filedir = os.path.dirname(__file__)
        data_path = os.path.join(filedir, 'data', 'IU_stations.txt')
        # image_path = os.path.join(filedir, 'images', 'ray_paths.png')
        inventory = read_inventory(data_path)
        # inventory = read_inventory()
        catalog = read_events()

        # this test uses the resampling method along the CMB
        view_dict = {'elevation': 80, 'azimuth': -20, 'distance': 4.,
                     'focalpoint': (0., 0., 0.)}
        with ImageComparison(self.path, 'ray_paths.png', reltol=1.5) as ic:
            plot_rays(inventory=inventory, catalog=catalog,
                      phase_list=['Pdiff'], colorscheme='dark',
                      kind='mayavi', view_dict=view_dict, icol=2,
                      fname_out=ic.name)

        # the following test is for an animated mayavi windows
        # and movie plotting.
        # Needs third party coastlines by default, otherwise
        # remove the coastlines keyword !
        #
        # plot_rays(inventory=inv, catalog=cat,
        #           phase_list=['PKP'], animate=True, savemovie=False,
        #           kind='mayavi', figsize=(1920, 1080),
        #           coastlines='data/coastlines.vtk')


def suite():
    return unittest.makeSuite(PathPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
