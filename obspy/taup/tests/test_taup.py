# -*- coding: utf-8 -*-
"""
The obspy.taup test suite.
"""
from __future__ import unicode_literals
from future.builtins import range
from future.builtins import open
from obspy.taup.taup import getTravelTimes
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
        filename = os.path.join(self.path, 'sample_ttimes_ak135.lst')
        with open(filename, 'rt') as fp:
            data = fp.readlines()
        #1
        tt = getTravelTimes(delta=52.474, depth=611.0, model='ak135')
        lines = data[5:29]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 2)
            self.assertAlmostEqual(item['take-off angle'],
                                   float(parts[2].strip()), 2)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3].strip()), 2)
            self.assertAlmostEqual(item['dT/dh'], float(parts[4].strip()), 2)
            self.assertAlmostEqual(item['d2T/dD2'],
                                   float(parts[5].strip()), 2)
        #2
        tt = getTravelTimes(delta=50.0, depth=300.0, model='ak135')
        lines = data[34:59]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 2)
            self.assertAlmostEqual(item['take-off angle'],
                                   float(parts[2].strip()), 2)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3].strip()), 2)
            self.assertAlmostEqual(item['dT/dh'], float(parts[4].strip()), 2)
            self.assertAlmostEqual(item['d2T/dD2'],
                                   float(parts[5].strip()), 2)
        #3
        tt = getTravelTimes(delta=150.0, depth=300.0, model='ak135')
        lines = data[61:88]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 2)
            self.assertAlmostEqual(item['take-off angle'],
                                   float(parts[2].strip()), 2)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3].strip()), 2)
            self.assertAlmostEqual(item['dT/dh'], float(parts[4].strip()), 2)
            self.assertAlmostEqual(item['d2T/dD2'],
                                   float(parts[5].strip()), 2)

    def test_getTravelTimesIASP91(self):
        """
        Tests getTravelTimes method using model iasp91.
        """
        # read output results from original program
        filename = os.path.join(self.path, 'sample_ttimes_iasp91.lst')
        with open(filename, 'rt') as fp:
            data = fp.readlines()
        #1
        tt = getTravelTimes(delta=52.474, depth=611.0, model='iasp91')
        lines = data[5:29]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 2)
            self.assertAlmostEqual(item['take-off angle'],
                                   float(parts[2].strip()), 2)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3].strip()), 2)
            self.assertAlmostEqual(item['dT/dh'], float(parts[4].strip()), 2)
            self.assertAlmostEqual(item['d2T/dD2'],
                                   float(parts[5].strip()), 2)
        #2
        tt = getTravelTimes(delta=50.0, depth=300.0, model='iasp91')
        lines = data[34:59]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 2)
            self.assertAlmostEqual(item['take-off angle'],
                                   float(parts[2].strip()), 2)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3].strip()), 2)
            self.assertAlmostEqual(item['dT/dh'], float(parts[4].strip()), 2)
            self.assertAlmostEqual(item['d2T/dD2'],
                                   float(parts[5].strip()), 2)
        #3
        tt = getTravelTimes(delta=150.0, depth=300.0, model='iasp91')
        lines = data[61:89]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for i in range(len(lines)):
            parts = lines[i][13:].split()
            item = tt[i]
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 2)
            self.assertAlmostEqual(item['take-off angle'],
                                   float(parts[2].strip()), 2)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3].strip()), 2)
            self.assertAlmostEqual(item['dT/dh'], float(parts[4].strip()), 2)
            self.assertAlmostEqual(item['d2T/dD2'],
                                   float(parts[5].strip()), 2)


def suite():
    return unittest.makeSuite(TauPTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
