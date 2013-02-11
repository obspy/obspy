#! /usr/bin/env python
# -*- coding: utf-8 -*-

from obspy.core.util import FlinnEngdahl
import os
import unittest


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
                    msg="%f, %f got %s instead of %s" % (
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
