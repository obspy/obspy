# -*- coding: utf-8 -*-
"""
Tests the wrapper of ObsPy around obspy.taup.TauPyModel.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest
import warnings

from obspy.taup.taup import getTravelTimes


class ObsPyTauPWrapperTestCase(unittest.TestCase):
    """
    Test suite for the ObsPy TauPy wrapper.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_get_travel_times_ak135(self):
        """
        Tests getTravelTimes method using model ak135.
        """
        # read output results from original program
        filename = os.path.join(self.path, 'sample_ttimes_ak135.lst')
        with open(filename, 'rt') as fp:
            data = fp.readlines()

        # 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tt = getTravelTimes(delta=52.474, depth=611.0, model='ak135')[:16]
        lines = data[5:21]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for line, item in zip(lines, tt):
            parts = line[13:].split()
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1]), 1)
            # The takeoff angle is defined a bit differently in the new
            # version.
            if item["take-off angle"] < 0.0:
                item["take-off angle"] += 180.0
            self.assertAlmostEqual(item["take-off angle"], float(parts[2]), 0)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3]), 1)

        # 2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tt = getTravelTimes(delta=50.0, depth=300.0, model='ak135')[:17]
        lines = data[26:43]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for line, item in zip(lines, tt):
            parts = line[13:].split()
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1]), 1)
            if item["take-off angle"] < 0.0:
                item["take-off angle"] += 180.0
            self.assertAlmostEqual(item['take-off angle'], float(parts[2]), 0)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3]), 1)

    def test_get_travel_times_iasp_91(self):
        """
        Tests getTravelTimes method using model iasp91.
        """
        # read output results from original program
        filename = os.path.join(self.path, 'sample_ttimes_iasp91.lst')
        with open(filename, 'rt') as fp:
            data = fp.readlines()

        # 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tt = getTravelTimes(delta=52.474, depth=611.0, model='iasp91')[:16]
        lines = data[5:21]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for line, item in zip(lines, tt):
            parts = line[13:].split()
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1].strip()), 1)
            if item["take-off angle"] < 0.0:
                item["take-off angle"] += 180.0
            self.assertAlmostEqual(item['take-off angle'], float(parts[2]), 0)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3]), 1)

        # 2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tt = getTravelTimes(delta=50.0, depth=300.0, model='iasp91')[:19]
        lines = data[26:45]
        self.assertEqual(len(tt), len(lines))
        # check calculated tt against original
        for line, item in zip(lines, tt):
            parts = line[13:].split()
            self.assertEqual(item['phase_name'], parts[0].strip())
            self.assertAlmostEqual(item['time'], float(parts[1]), 1)
            if item["take-off angle"] < 0.0:
                item["take-off angle"] += 180.0
            self.assertAlmostEqual(item['take-off angle'], float(parts[2]), 0)
            self.assertAlmostEqual(item['dT/dD'], float(parts[3]), 1)

    def test_issue_with_global_state(self):
        """
        Minimal test case for an issue with global state that results in
        different results for the same call to getTravelTimes() in some
        circumstances.

        See #728 for more details.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tt_1 = getTravelTimes(delta=100, depth=0, model="ak135")

            # Some other calculation in between.
            getTravelTimes(delta=100, depth=200, model="ak135")

            tt_2 = getTravelTimes(delta=100, depth=0, model="ak135")

        # Both should be equal if everything is alright.
        self.assertEqual(tt_1, tt_2)


def suite():
    return unittest.makeSuite(ObsPyTauPWrapperTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
