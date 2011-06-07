# -*- coding: utf-8 -*-
"""
The obspy.taup test suite.
"""
from obspy.taup.taup import getTravelTimes, kilometer2degrees
from obspy.taup.taup import locations2degrees
import math
import os
import unittest


class TauPTestCase(unittest.TestCase):
    """
    Test suite for obspy.taup.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_getTravelTimesAK135(self):
        """
        Tests getTravelTimes method using model ak135.
        """
        # read output results from original program
        file = os.path.join(self.path, 'sample_ttimes_ak135.lst')
        data = open(file, 'rt').readlines()
        #1
        tt = getTravelTimes(delta=52.474, depth=611.0, model='ak135')
        lines = data[5:29]
        self.assertEquals(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEquals(item['phase_name'], parts[0].strip())
            self.assertAlmostEquals(item['time'], float(parts[1].strip()), 3)
            self.assertAlmostEquals(item['take-off angle'],
                                    float(parts[2].strip()), 3)
            self.assertAlmostEquals(item['dT/dD'], float(parts[3].strip()), 3)
            self.assertAlmostEquals(item['dT/dh'], float(parts[4].strip()), 3)
            self.assertAlmostEquals(item['d2T/dD2'],
                                    float(parts[5].strip()), 3)
        #2
        tt = getTravelTimes(delta=50.0, depth=300.0, model='ak135')
        lines = data[34:59]
        self.assertEquals(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEquals(item['phase_name'], parts[0].strip())
            self.assertAlmostEquals(item['time'], float(parts[1].strip()), 3)
            self.assertAlmostEquals(item['take-off angle'],
                                    float(parts[2].strip()), 3)
            self.assertAlmostEquals(item['dT/dD'], float(parts[3].strip()), 3)
            self.assertAlmostEquals(item['dT/dh'], float(parts[4].strip()), 3)
            self.assertAlmostEquals(item['d2T/dD2'],
                                    float(parts[5].strip()), 3)
        #3
        tt = getTravelTimes(delta=150.0, depth=300.0, model='ak135')
        lines = data[61:88]
        self.assertEquals(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEquals(item['phase_name'], parts[0].strip())
            self.assertAlmostEquals(item['time'], float(parts[1].strip()), 3)
            self.assertAlmostEquals(item['take-off angle'],
                                    float(parts[2].strip()), 3)
            self.assertAlmostEquals(item['dT/dD'], float(parts[3].strip()), 3)
            self.assertAlmostEquals(item['dT/dh'], float(parts[4].strip()), 3)
            self.assertAlmostEquals(item['d2T/dD2'],
                                    float(parts[5].strip()), 2)

    def test_getTravelTimesIASP91(self):
        """
        Tests getTravelTimes method using model iasp91.
        """
        # read output results from original program
        file = os.path.join(self.path, 'sample_ttimes_iasp91.lst')
        data = open(file, 'rt').readlines()
        #1
        tt = getTravelTimes(delta=52.474, depth=611.0, model='iasp91')
        lines = data[5:29]
        self.assertEquals(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEquals(item['phase_name'], parts[0].strip())
            self.assertAlmostEquals(item['time'], float(parts[1].strip()), 3)
            self.assertAlmostEquals(item['take-off angle'],
                                    float(parts[2].strip()), 2)
            self.assertAlmostEquals(item['dT/dD'], float(parts[3].strip()), 3)
            self.assertAlmostEquals(item['dT/dh'], float(parts[4].strip()), 3)
            self.assertAlmostEquals(item['d2T/dD2'],
                                    float(parts[5].strip()), 3)
        #2
        tt = getTravelTimes(delta=50.0, depth=300.0, model='iasp91')
        lines = data[34:59]
        self.assertEquals(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEquals(item['phase_name'], parts[0].strip())
            self.assertAlmostEquals(item['time'], float(parts[1].strip()), 3)
            self.assertAlmostEquals(item['take-off angle'],
                                    float(parts[2].strip()), 2)
            self.assertAlmostEquals(item['dT/dD'], float(parts[3].strip()), 3)
            self.assertAlmostEquals(item['dT/dh'], float(parts[4].strip()), 3)
            self.assertAlmostEquals(item['d2T/dD2'],
                                    float(parts[5].strip()), 3)
        #3
        tt = getTravelTimes(delta=150.0, depth=300.0, model='iasp91')
        lines = data[61:89]
        self.assertEquals(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEquals(item['phase_name'], parts[0].strip())
            self.assertAlmostEquals(item['time'], float(parts[1].strip()), 3)
            self.assertAlmostEquals(item['take-off angle'],
                                    float(parts[2].strip()), 3)
            self.assertAlmostEquals(item['dT/dD'], float(parts[3].strip()), 3)
            self.assertAlmostEquals(item['dT/dh'], float(parts[4].strip()), 3)
            self.assertAlmostEquals(item['d2T/dD2'],
                                    float(parts[5].strip()), 2)

    def test_kilometer2degrees(self):
        """
        Simple test of the convenience function.
        """
        # Test if it works.
        self.assertEqual(kilometer2degrees(111.19492664455873, radius=6371),
                         1.0)
        # Test if setting the radius actually does something. Round to avoid
        # some precision problems on different machines.
        self.assertEqual(round(kilometer2degrees(111.19492664455873,
                         radius=6381), 5), round(0.99843284751606332, 5))

    def test_locations2degrees(self):
        """
        Test the location 2 degree conversion.
        """
        # Inline method to avoid messy code.
        def assertLoc(lat1, long1, lat2, long2, approx_distance):
            self.assertTrue( \
            abs(math.radians(locations2degrees(lat1, long1, lat2, long2)) \
                * 6371 - approx_distance) <= 20)

        # Approximate values from the Great Circle Calculator:
        #   http://williams.best.vwh.net/gccalc.htm

        # Random location.
        assertLoc(36.12, -86.67, 33.94, -118.40, 2893)
        # Test several combinations of quadrants.
        assertLoc(11.11, 22.22, 33.33, 44.44, 3346)
        assertLoc(-11.11, -22.22, -33.33, -44.44, 3346)
        assertLoc(11.11, 22.22, -33.33, -44.44, 8596)
        assertLoc(-11.11, -22.22, 33.33, 44.44, 8596)
        assertLoc(11.11, -22.22, 33.33, -44.44, 3346)
        assertLoc(-11.11, 22.22, 33.33, 44.44, 5454)
        assertLoc(11.11, -22.22, 33.33, 44.44, 7177)
        assertLoc(11.11, 22.22, -33.33, 44.44, 5454)
        assertLoc(11.11, 22.22, 33.33, -44.44, 7177)
        # Test some extreme values.
        assertLoc(90, 0, 0, 0, 10018)
        assertLoc(180, 0, 0, 0, 20004)
        assertLoc(0, 90, 0, 0, 10018)
        assertLoc(0, 180, 0, 0, 20004)
        assertLoc(0, 0, 90, 0, 10018)
        assertLoc(0, 0, 180, 0, 20004)
        assertLoc(0, 0, 0, 90, 10018)
        assertLoc(0, 0, 0, 180, 20004)
        assertLoc(11, 55, 11, 55, 0)


def suite():
    return unittest.makeSuite(TauPTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
