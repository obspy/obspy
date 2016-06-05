# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy import read_inventory, read_events
from obspy.imaging.ray_paths import plot_rays


class PathPlottingTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    @unittest.skip('Needs Mayavi to run')
    def test_pathplotting(self):
        inv = read_inventory('data/IU.xml')
        cat = read_events()
        # uncomment the following for mayavi tests
        plot_rays(inventory=inv, catalog=cat, phase_list=['PcP'],
                  kind='mayavi', colorscheme='dark')
        # plot_rays(inventory=inv, catalog=cat,
        #           phase_list=['PKP'], animate=False, savemovie=False,
        #           kind='mayavi', figsize=(1920, 1080),
        #           coastlines='data/coastlines.vtk')
        # plot_rays(station_latitude=0., station_longitude=30,
        #           event_latitude=0.,
        #           event_longitude=170, event_depth_in_km=200.,
        #           phase_list=['Pdiff'], colorscheme='dark', kind='mayavi')


def suite():
    return unittest.makeSuite(PathPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
