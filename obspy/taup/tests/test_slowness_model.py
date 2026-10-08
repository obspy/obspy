#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import pytest

from obspy.taup import _DEFAULT_VALUES
from obspy.taup.slowness_layer import (SlownessLayer, SlownessModelError,
                                       bullen_depth_for, create_from_vlayer,
                                       evaluate_at_bullen)
from obspy.taup.velocity_layer import VelocityLayer


class TestTauPySlownessModel:
    def test_slowness_layer(self):
        v_layer = np.array([(10, 31, 3, 5, 2, 4,
                            _DEFAULT_VALUES["density"],
                            _DEFAULT_VALUES["density"],
                            _DEFAULT_VALUES["qp"],
                            _DEFAULT_VALUES["qp"],
                            _DEFAULT_VALUES["qs"],
                            _DEFAULT_VALUES["qs"])],
                           dtype=VelocityLayer)
        a = create_from_vlayer(v_layer, True, radius_of_planet=6371.0)
        assert a['bot_p'] == 1268.0
        assert a['bot_depth'] == 31.0
        b = create_from_vlayer(v_layer, False, radius_of_planet=6371.0)
        assert b['top_p'] == 3180.5


class TestBullenDepth:
    def test_overflow(self):
        sl = np.array([(2548.4, 6.546970605878823, 1846.2459389213773,
                        13.798727310994103)], dtype=SlownessLayer)
        try:
            depth = bullen_depth_for(sl, 2197.322969460689, 6371)
        except SlownessModelError:
            pytest.fail('SlownessModelError was incorrectly raised.')
        assert not np.isnan(depth)


class TestEvaluateAtBullen:
    @pytest.mark.parametrize('depth', [11.0, np.float64(11.0)])
    def test_overflow(self, depth):
        """
        A layer thicker than 2 km with a large power law exponent overflows
        the power law and must fall back to linear interpolation (see #3433).
        """
        sl = np.array([(1100.0, 10.0, 1048.5, 12.01)], dtype=SlownessLayer)
        p = evaluate_at_bullen(sl[0], depth, 6371.0)
        expected = 1100.0 + (1048.5 - 1100.0) / 2.01 * (11.0 - 10.0)
        assert p == pytest.approx(expected)

    def test_power_law(self):
        sl = np.array([(2000.0, 10.0, 1990.0, 50.0)], dtype=SlownessLayer)
        b = np.log(2000.0 / 1990.0) / np.log(6361.0 / 6321.0)
        expected = 2000.0 / 6361.0 ** b * 6341.0 ** b
        p = evaluate_at_bullen(sl[0], 30.0, 6371.0)
        assert p == pytest.approx(expected)
