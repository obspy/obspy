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
import pytest
from matplotlib import rcParams

from obspy import read_inventory
from obspy.core.inventory import Channel, Equipment
from obspy.core.util.testing import WarningsCapture


@pytest.mark.usefixtures('ignore_numpy_errors')
class TestChannel:
    """
    Tests the for :class:`~obspy.core.inventory.channel.Channel` class.
    """
    def test_response_plot(self, image_path):
        """
        Tests the response plot.
        """
        cha = read_inventory()[0][0][0]
        with WarningsCapture():
            rcParams['savefig.dpi'] = 72
            cha.plot(0.005, outfile=image_path)

    def test_channel_str(self):
        """
        Tests the __str__ method of the channel object.
        """
        c = Channel(code="BHE", location_code="10", latitude=1, longitude=2,
                    elevation=3, depth=4, azimuth=5, dip=6)
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n")

        # Adding channel types.
        c.types = ["A", "B"]
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n")

        # Adding channel types.
        c.sample_rate = 10.0
        assert str(c) == (
            "Channel 'BHE', Location '10' \n"
            "\tTime range: -- - --\n"
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
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
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
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
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
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
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
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
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
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
            "\tLatitude: 1.0000, Longitude: 2.0000, Elevation: 3.0 m, "
            "Local Depth: 4.0 m\n"
            "\tAzimuth: 5.00 degrees from north, clockwise\n"
            "\tDip: 6.00 degrees down from horizontal\n"
            "\tChannel types: A, B\n"
            "\tSampling Rate: 10.00 Hz\n"
            "\tSensor (Description): random (some description)\n"
            "\tResponse information available"
        )
