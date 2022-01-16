#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import pytest

from obspy.taup import _DEFAULT_VALUES
from obspy.taup.slowness_layer import (SlownessLayer, SlownessModelError,
                                       bullen_depth_for, create_from_vlayer)
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
