#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the channel handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import unittest
import warnings

import numpy as np
from matplotlib import rcParams

from obspy.core.util import MATPLOTLIB_VERSION
from obspy.core.util.testing import ImageComparison
from obspy import read_inventory
from obspy.core.inventory import Channel, Equipment


class ChannelTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.core.inventory.channel.Channel` class.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
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

        cha = read_inventory()[0][0][0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "channel_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                cha.plot(0.005, outfile=ic.name)

    def test_channel_str(self):
        """
        Tests the __str__ method of the channel object.
        """
        c = Channel(code="BHE", location_code="10", latitude=1, longitude=2,
                    elevation=3, depth=4, azimuth=5, dip=6)
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n")

        # Adding channel types.
        c.types = ["A", "B"]
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n")

        # Adding channel types.
        c.sample_rate = 10.0
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n")

        # "Adding" response
        c.response = True
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n"
            "\tResponse information available"
        )

        # Adding an empty sensor.
        c.sensor = Equipment(type=None)
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n"
            "\tSensor (Description): None (None)\n"
            "\tResponse information available"
        )

        # Adding a sensor with only a type.
        c.sensor = Equipment(type="random")
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n"
            "\tSensor (Description): random (None)\n"
            "\tResponse information available"
        )

        # Adding a sensor with only a description
        c.sensor = Equipment(description="some description")
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n"
            "\tSensor (Description): None (some description)\n"
            "\tResponse information available"
        )

        # Adding a sensor with type and description
        c.sensor = Equipment(type="random", description="some description")
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.00, Longitude: 2.00, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n"
            "\tSensor (Description): random (some description)\n"
            "\tResponse information available"
        )


def suite():
    return unittest.makeSuite(ChannelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
