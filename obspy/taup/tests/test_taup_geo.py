#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.taup.tau import TauPyModel
import obspy.geodetics.base as geodetics


class TaupGeoTestCase(unittest.TestCase):
    """
    Test suite for the SeismicPhase class.
    """
    def setUp(self):
        self.model = TauPyModel('iasp91')

    @unittest.skipIf(not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
                     'Module geographiclib is not installed or too old.')
    def test_path_geo(self):
        evlat, evlon = 0., 20.
        evdepth = 10.
        stlat, stlon = 0., -80.
        arrivals = self.model.get_ray_paths_geo(evdepth, evlat, evlon, stlat,
                                                stlon)
        for arr in arrivals:
            stlat_path = arr.path['lat'][-1]
            stlon_path = arr.path['lon'][-1]
            self.assertAlmostEqual(stlat, stlat_path, delta=0.1)
            self.assertAlmostEqual(stlon, stlon_path, delta=0.1)


def suite():
    return unittest.makeSuite(TaupGeoTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
