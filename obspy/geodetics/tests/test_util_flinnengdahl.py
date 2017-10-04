#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

from obspy.scripts.flinnengdahl import main as obspy_flinnengdahl
from obspy.geodetics import FlinnEngdahl
from obspy.core.util.misc import CatchOutput


class UtilFlinnEngdahlTestCase(unittest.TestCase):
    def setUp(self):
        self.flinnengdahl = FlinnEngdahl()
        self.samples_file = os.path.join(
            os.path.dirname(__file__),
            'data',
            'flinnengdahl.csv'
        )

    def test_coordinates(self):
        with open(self.samples_file, 'r') as fh:
            for line in fh:
                longitude, latitude, checked_region = line.strip().split('\t')
                longitude = float(longitude)
                latitude = float(latitude)

                region = self.flinnengdahl.get_region(longitude, latitude)
                self.assertEqual(
                    region,
                    checked_region,
                    msg="(%f, %f) got %s instead of %s" % (
                        longitude,
                        latitude,
                        region,
                        checked_region
                    )
                )

    def test_script(self):
        with open(self.samples_file, 'r') as fh:
            # Testing once is sufficient.
            line = fh.readline()
            longitude, latitude, checked_region = line.strip().split('\t')

            with CatchOutput() as out:
                obspy_flinnengdahl([longitude, latitude])
            region = out.stdout.strip()

            self.assertEqual(
                region,
                checked_region,
                msg='(%s, %s) got %s instead of %s' % (
                    longitude,
                    latitude,
                    region,
                    checked_region
                )
            )


def suite():
    return unittest.makeSuite(UtilFlinnEngdahlTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
