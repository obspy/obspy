#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the station handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.station import read_inventory
import os
import numpy as np
import unittest
import warnings
from obspy.core.util.testing import ImageComparison, HAS_COMPARE_IMAGE
from obspy.core.util.decorator import skipIf

if HAS_COMPARE_IMAGE:
    from matplotlib import rcParams


class StationTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.station.Station` class.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    @skipIf(not HAS_COMPARE_IMAGE, 'nose not installed or matplotlib too old')
    def test_response_plot(self):
        """
        Tests the response plot.
        """
        sta = read_inventory()[0][0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "station_response.png") as ic:
                rcParams['savefig.dpi'] = 72
                sta.plot(0.05, channel="*[NE]", outfile=ic.name)


def suite():
    return unittest.makeSuite(StationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
