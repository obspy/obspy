#! /usr/bin/env python
# -*- coding: utf-8 -*-
from obspy.scripts.flinnengdahl import main as obspy_flinnengdahl
from obspy.geodetics import FlinnEngdahl
from obspy.core.util.misc import CatchOutput


class TestUtilFlinnEngdahl:

    def test_coordinates(self, testdata):
        flinn_engdahl = FlinnEngdahl()
        with open(testdata['flinnengdahl.csv'], 'r') as fh:
            for line in fh:
                longitude, latitude, checked_region = line.strip().split('\t')
                longitude = float(longitude)
                latitude = float(latitude)

                region = flinn_engdahl.get_region(longitude, latitude)
                assert region == \
                    checked_region, \
                    "(%f, %f) got %s instead of %s" % (
                        longitude,
                        latitude,
                        region,
                        checked_region
                    )

    def test_script(self, testdata):
        with open(testdata['flinnengdahl.csv'], 'r') as fh:
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
