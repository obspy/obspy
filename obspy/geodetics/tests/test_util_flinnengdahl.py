#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest

from obspy.scripts.flinnengdahl import main as obspy_flinnengdahl
from obspy.geodetics import FlinnEngdahl
from obspy.core.util.misc import CatchOutput


class TestUtilFlinnEngdahl:

    @pytest.fixture()
    def flinnengdahl(self):
        """Return instance of FlinnEngdahl."""
        return FlinnEngdahl()

    @pytest.fixture(scope='class')
    def sample_file_path(self):
        """A path to the sample files"""
        sample_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'flinnengdahl.csv'
        )
        return sample_file

    def test_coordinates(self, flinnengdahl, sample_file_path):
        with open(sample_file_path, 'r') as fh:
            for line in fh:
                longitude, latitude, checked_region = line.strip().split('\t')
                longitude = float(longitude)
                latitude = float(latitude)

                region = flinnengdahl.get_region(longitude, latitude)
                assert region == \
                    checked_region, \
                    "(%f, %f) got %s instead of %s" % (
                        longitude,
                        latitude,
                        region,
                        checked_region
                    )

    def test_script(self, sample_file_path):
        with open(sample_file_path, 'r') as fh:
            # Testing once is sufficient.
            line = fh.readline()
            longitude, latitude, checked_region = line.strip().split('\t')

            with CatchOutput() as out:
                obspy_flinnengdahl([longitude, latitude])
            region = out.stdout.strip()

            assert region == \
                checked_region, \
                '(%s, %s) got %s instead of %s' % (
                    longitude,
                    latitude,
                    region,
                    checked_region
                )
