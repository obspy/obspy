#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np

from obspy.taup.tau_model import TauModel


class SplitTauModelTestCase(unittest.TestCase):
    """
    Test suite for splitting of the TauModel class.
    """

    def test_split(self):
        depth = 110
        tMod = TauModel.from_file('iasp91')
        splitTMod = tMod.splitBranch(depth)
        self.assertEqual(tMod.tauBranches.shape[1] + 1,
                         splitTMod.tauBranches.shape[1])
        self.assertEqual(len(tMod.ray_params) + 2, len(splitTMod.ray_params))

        branch_count = tMod.tauBranches.shape[1]
        split_branch_index = tMod.findBranch(depth)

        new_P_ray_param = splitTMod.sMod.getSlownessLayer(
            splitTMod.sMod.layerNumberAbove(depth, True), True)['botP']
        new_S_ray_param = splitTMod.sMod.getSlownessLayer(
            splitTMod.sMod.layerNumberAbove(depth, False), False)['botP']

        pIndex = sIndex = len(splitTMod.ray_params)
        ray_params = splitTMod.ray_params
        for j in range(len(ray_params)):
            if new_P_ray_param == ray_params[j]:
                pIndex = j
            if new_S_ray_param == ray_params[j]:
                sIndex = j

        self.assertTrue(pIndex == len(splitTMod.ray_params) or sIndex < pIndex)

        for b in range(branch_count):
            orig = tMod.getTauBranch(b, True)
            depthBranch = None
            if b < split_branch_index:
                depthBranch = splitTMod.getTauBranch(b, True)
                self.assertGreater(depthBranch.dist[pIndex], 0)
                self.assertGreater(depthBranch.time[pIndex], 0)
            elif b > split_branch_index:
                depthBranch = splitTMod.getTauBranch(b + 1, True)
                self.assertAlmostEqual(depthBranch.dist[pIndex], 0,
                                       delta=0.00000001)
                self.assertAlmostEqual(depthBranch.time[pIndex], 0,
                                       delta=0.00000001)
            else:
                # the split one
                continue

            np.testing.assert_allclose(orig.dist[0:sIndex],
                                       depthBranch.dist[0:sIndex],
                                       atol=0.00000001)
            np.testing.assert_allclose(orig.time[0:sIndex],
                                       depthBranch.time[0:sIndex],
                                       atol=0.00000001)
            orig_len = len(orig.dist)
            if sIndex < orig_len:
                self.assertEqual(orig_len + 2, len(depthBranch.dist))
                np.testing.assert_allclose(orig.dist[sIndex:pIndex - 1],
                                           depthBranch.dist[sIndex + 1:pIndex],
                                           atol=0.00000001)
                np.testing.assert_allclose(orig.time[sIndex:pIndex - 1],
                                           depthBranch.time[sIndex + 1:pIndex],
                                           atol=0.00000001)
                np.testing.assert_allclose(
                    orig.dist[pIndex:orig_len-sIndex-2],
                    depthBranch.dist[pIndex+2:orig_len-sIndex],
                    atol=0.00000001)
                np.testing.assert_allclose(
                    orig.time[pIndex:orig_len-sIndex-2],
                    depthBranch.time[pIndex+2:orig_len-sIndex],
                    atol=0.00000001)

        # now check branch split
        orig = tMod.getTauBranch(split_branch_index, True)
        above = splitTMod.getTauBranch(split_branch_index, True)
        below = splitTMod.getTauBranch(split_branch_index + 1, True)
        self.assertAlmostEqual(above.minRayParam, below.maxRayParam, 8)
        for i in range(len(above.dist)):
            if i < sIndex:
                self.assertAlmostEqual(orig.dist[i], above.dist[i],
                                       delta=0.000000001)
            elif i == sIndex:
                # new value should be close to average of values to either side
                self.assertAlmostEqual((orig.dist[i - 1] + orig.dist[i]) / 2,
                                       above.dist[i], delta=0.00001)
            elif i > sIndex and i < pIndex:
                self.assertAlmostEqual(orig.dist[i - 1],
                                       above.dist[i] + below.dist[i],
                                       delta=0.000000001)
            elif i == pIndex:
                # new value should be close to average of values to either side
                self.assertAlmostEqual(
                    (orig.dist[i - 2] + orig.dist[i - 1]) / 2,
                    above.dist[i] + below.dist[i],
                    delta=0.0001)
            else:
                self.assertAlmostEqual(orig.dist[i - 2],
                                       above.dist[i] + below.dist[i],
                                       delta=0.000000001)


def suite():
    return unittest.makeSuite(SplitTauModelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
