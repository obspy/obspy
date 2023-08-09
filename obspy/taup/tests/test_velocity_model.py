#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os

import numpy as np

from obspy.taup.velocity_model import VelocityModel


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


class TestTauPyVelocityModel:
    def test_read_velocity_model(self):
        for filename in ['iasp91.tvel', 'iasp91_w_comment.tvel', 'iasp91.nd',
                         'iasp91_w_comment.nd']:
            velocity_model = os.path.join(DATA, filename)
            test2 = VelocityModel.read_velocity_file(velocity_model)

            assert len(test2.layers) == 129
            assert len(test2) == 129

            assert test2.radius_of_planet == 6371.0
            assert test2.moho_depth == 35
            assert test2.cmb_depth == 2889.0
            assert test2.iocb_depth == 5153.9
            assert test2.min_radius == 0.0
            assert test2.max_radius == 6371.0

            assert test2.validate()

            np.testing.assert_equal(
                test2.get_discontinuity_depths(),
                [0.0, 20.0, 35.0, 210.0, 410.0, 660.0, 2889.0, 5153.9, 6371.0])

            # check boundary cases
            assert test2.layer_number_above(6371) == 128
            assert test2.layer_number_below(0) == 0

            # evaluate at cmb
            assert test2.evaluate_above(2889.0, 'p') == 13.6908
            assert test2.evaluate_below(2889.0, 'D') == 9.9145
            assert test2.depth_at_top(50) == 2393.5
            assert test2.depth_at_bottom(50) == 2443.0
            assert not test2.fix_discontinuity_depths()
