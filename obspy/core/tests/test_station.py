#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the station handling.

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

from obspy import read_inventory, UTCDateTime
from obspy.core.util import MATPLOTLIB_VERSION
from obspy.core.util.testing import ImageComparison


class StationTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.core.inventory.station.Station` class.
    """
    def setUp(self):
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.x:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 5, 0]:
            reltol = 2.0

        sta = read_inventory()[0][0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "station_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                sta.plot(0.05, channel="*[NE]", outfile=ic.name)

    def test_response_plot_degrees(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.x:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 5, 0]:
            reltol = 2.0

        sta = read_inventory()[0][0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir,
                                 "station_response_degrees.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                sta.plot(0.05, channel="*[NE]", plot_degrees=True,
                         outfile=ic.name)

    def test_len(self):
        """
        Tests the __len__ property.
        """
        sta = read_inventory()[0][0]

        self.assertEqual(len(sta), len(sta.channels))
        self.assertEqual(len(sta), 12)

    def test_station_select(self):
        """
        Tests the select() method on station objects.
        """
        sta = read_inventory()[0][0]

        # Basic assertions to make sure the test data does not change.
        self.assertEqual(len(sta), 12)
        self.assertEqual(sta.code, "FUR")
        self.assertEqual(sorted(["%s.%s" % (_i.location_code, _i.code) for _i
                                 in sta]),
                         ['.BHE', '.BHN', '.BHZ', '.HHE', '.HHN', '.HHZ',
                          '.LHE', '.LHN', '.LHZ', '.VHE', '.VHN', '.VHZ'])

        self.assertEqual(sta[0].code, "HHZ")
        # Manually set the end-date of the first one.
        sta[0].end_date = UTCDateTime(2010, 1, 1)

        # If nothing is given, nothing should change.
        sta_2 = sta.select()
        self.assertEqual(len(sta_2), 12)
        self.assertEqual(sta_2.code, "FUR")

        # Only select vertical channels.
        sta_2 = sta.select(channel="*Z")
        self.assertEqual(len(sta_2), 4)
        self.assertEqual(sta_2.code, "FUR")
        self.assertEqual(sorted(["%s.%s" % (_i.location_code, _i.code) for _i
                                 in sta_2]), ['.BHZ', '.HHZ', '.LHZ', '.VHZ'])

        # Only BH channels.
        sta_2 = sta.select(channel="BH?")
        self.assertEqual(len(sta_2), 3)
        self.assertEqual(sta_2.code, "FUR")
        self.assertEqual(sorted(["%s.%s" % (_i.location_code, _i.code) for _i
                                 in sta_2]), ['.BHE', '.BHN', '.BHZ'])

        # All location codes.
        sta_2 = sta.select(location="*")
        self.assertEqual(len(sta_2), 12)
        self.assertEqual(sta_2.code, "FUR")

        sta_2 = sta.select(location="")
        self.assertEqual(len(sta_2), 12)
        self.assertEqual(sta_2.code, "FUR")

        # None exist with this code.
        sta_2 = sta.select(location="10")
        self.assertEqual(len(sta_2), 0)
        self.assertEqual(sta_2.code, "FUR")

        # The time parameter selects channels active at that particular
        # time. All channels start 2006-12-16 and only the first ends in
        # 2010-1-1. All others don't have an end-date set.
        self.assertEqual(len(sta.select(time=UTCDateTime(2005, 1, 1))), 0)
        self.assertEqual(len(sta.select(time=UTCDateTime(2007, 1, 1))), 12)
        self.assertEqual(len(sta.select(time=UTCDateTime(2006, 12, 15))), 0)
        self.assertEqual(len(sta.select(time=UTCDateTime(2006, 12, 17))), 12)
        self.assertEqual(len(sta.select(time=UTCDateTime(2012, 1, 1))), 11)

        # Test starttime parameter.
        self.assertEqual(
            len(sta.select(starttime=UTCDateTime(2005, 1, 1))), 12)
        self.assertEqual(
            len(sta.select(starttime=UTCDateTime(2009, 1, 1))), 12)
        self.assertEqual(
            len(sta.select(starttime=UTCDateTime(2011, 1, 1))), 11)
        self.assertEqual(
            len(sta.select(starttime=UTCDateTime(2016, 1, 1))), 11)

        # Test endtime parameter.
        self.assertEqual(
            len(sta.select(endtime=UTCDateTime(2005, 1, 1))), 0)
        self.assertEqual(
            len(sta.select(endtime=UTCDateTime(2009, 1, 1))), 12)
        self.assertEqual(
            len(sta.select(endtime=UTCDateTime(2011, 1, 1))), 12)
        self.assertEqual(
            len(sta.select(endtime=UTCDateTime(2016, 1, 1))), 12)

        # Sampling rate parameter.
        self.assertEqual(len(sta.select(sampling_rate=33.0)), 0)
        self.assertEqual(len(sta.select(sampling_rate=100.0)), 3)
        self.assertEqual(len(sta.select(sampling_rate=20.0)), 3)
        self.assertEqual(len(sta.select(sampling_rate=1.0)), 3)
        self.assertEqual(len(sta.select(sampling_rate=0.1)), 3)

        self.assertEqual(sorted(["%s.%s" % (_i.location_code, _i.code) for _i
                                 in sta.select(sampling_rate=100.0)]),
                         ['.HHE', '.HHN', '.HHZ'])

        # Check tolerances.
        self.assertEqual(len(sta.select(sampling_rate=33.0 + 1E-6)), 0)
        self.assertEqual(len(sta.select(sampling_rate=100.0 + 1E-6)), 3)
        self.assertEqual(len(sta.select(sampling_rate=20.0 - 1E-6)), 3)
        self.assertEqual(len(sta.select(sampling_rate=1.0 + 1E-6)), 3)
        self.assertEqual(len(sta.select(sampling_rate=0.1 - 1E-6)), 3)

        # Artificially set different coordinates for a channel of RJOB.
        sta = read_inventory()[1][0]
        sta[0].latitude = 47.9
        sta[0].longitude = 12.9
        self.assertEqual(len(sta.select(
            minlatitude=47.8, maxlatitude=48,
            minlongitude=12.8, maxlongitude=13)), 1)
        self.assertEqual(len(sta.select(
            latitude=47.95, longitude=12.95, maxradius=0.1)), 1)
        self.assertEqual(len(sta.select(
            latitude=47.95, longitude=12.95, minradius=0.1)), 2)
        self.assertEqual(len(sta.select(
            latitude=47.95, longitude=12.95,
            minradius=0.08, maxradius=0.1)), 0)


def suite():
    return unittest.makeSuite(StationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
