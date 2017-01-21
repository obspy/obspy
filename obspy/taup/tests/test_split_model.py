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
        tau_model = TauModel.from_file('iasp91')
        split_t_mod = tau_model.split_branch(depth)
        self.assertEqual(tau_model.tau_branches.shape[1] + 1,
                         split_t_mod.tau_branches.shape[1])
        self.assertEqual(len(tau_model.ray_params) + 2,
                         len(split_t_mod.ray_params))

        branch_count = tau_model.tau_branches.shape[1]
        split_branch_index = tau_model.find_branch(depth)

        new_p_ray_param = split_t_mod.s_mod.get_slowness_layer(
            split_t_mod.s_mod.layer_number_above(depth, True), True)['bot_p']
        new_s_ray_param = split_t_mod.s_mod.get_slowness_layer(
            split_t_mod.s_mod.layer_number_above(depth, False), False)['bot_p']

        p_index = s_index = len(split_t_mod.ray_params)
        ray_params = split_t_mod.ray_params
        for j in range(len(ray_params)):
            if new_p_ray_param == ray_params[j]:
                p_index = j
            if new_s_ray_param == ray_params[j]:
                s_index = j

        self.assertTrue(p_index == len(split_t_mod.ray_params) or
                        s_index < p_index)

        for b in range(branch_count):
            orig = tau_model.get_tau_branch(b, True)
            if b < split_branch_index:
                depth_branch = split_t_mod.get_tau_branch(b, True)
                self.assertGreater(depth_branch.dist[p_index], 0)
                self.assertGreater(depth_branch.time[p_index], 0)
            elif b > split_branch_index:
                depth_branch = split_t_mod.get_tau_branch(b + 1, True)
                self.assertAlmostEqual(depth_branch.dist[p_index], 0,
                                       delta=0.00000001)
                self.assertAlmostEqual(depth_branch.time[p_index], 0,
                                       delta=0.00000001)
            else:
                # the split one
                continue

            np.testing.assert_allclose(orig.dist[0:s_index],
                                       depth_branch.dist[0:s_index],
                                       atol=0.00000001)
            np.testing.assert_allclose(orig.time[0:s_index],
                                       depth_branch.time[0:s_index],
                                       atol=0.00000001)
            orig_len = len(orig.dist)
            if s_index < orig_len:
                self.assertEqual(orig_len + 2, len(depth_branch.dist))
                np.testing.assert_allclose(
                    orig.dist[s_index:p_index - 1],
                    depth_branch.dist[s_index + 1:p_index],
                    atol=0.00000001)
                np.testing.assert_allclose(
                    orig.time[s_index:p_index - 1],
                    depth_branch.time[s_index + 1:p_index],
                    atol=0.00000001)
                np.testing.assert_allclose(
                    orig.dist[p_index:orig_len - s_index - 2],
                    depth_branch.dist[p_index + 2:orig_len - s_index],
                    atol=0.00000001)
                np.testing.assert_allclose(
                    orig.time[p_index:orig_len - s_index - 2],
                    depth_branch.time[p_index + 2:orig_len - s_index],
                    atol=0.00000001)

        # now check branch split
        orig = tau_model.get_tau_branch(split_branch_index, True)
        above = split_t_mod.get_tau_branch(split_branch_index, True)
        below = split_t_mod.get_tau_branch(split_branch_index + 1, True)
        self.assertAlmostEqual(above.min_ray_param, below.max_ray_param, 8)
        for i in range(len(above.dist)):
            if i < s_index:
                self.assertAlmostEqual(orig.dist[i], above.dist[i],
                                       delta=0.000000001)
            elif i == s_index:
                # new value should be close to average of values to either side
                self.assertAlmostEqual((orig.dist[i - 1] + orig.dist[i]) / 2,
                                       above.dist[i], delta=0.00001)
            elif i > s_index and i < p_index:
                self.assertAlmostEqual(orig.dist[i - 1],
                                       above.dist[i] + below.dist[i],
                                       delta=0.000000001)
            elif i == p_index:
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
