#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import unittest

import numpy as np

from obspy.taup.velocity_model import VelocityModel


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


class TauPyVelocityModelTestCase(unittest.TestCase):
    def test_read_velocity_model(self):
        for filename in ['iasp91.tvel', 'iasp91_w_comment.tvel', 'iasp91.nd',
                         'iasp91_w_comment.nd']:
            velocity_model = os.path.join(DATA, filename)
            test2 = VelocityModel.read_velocity_file(velocity_model)

            self.assertEqual(len(test2.layers), 129)
            self.assertEqual(len(test2), 129)

            self.assertEqual(test2.radius_of_planet, 6371.0)
            self.assertEqual(test2.moho_depth, 35)
            self.assertEqual(test2.cmb_depth, 2889.0)
            self.assertEqual(test2.iocb_depth, 5153.9)
            self.assertEqual(test2.min_radius, 0.0)
            self.assertEqual(test2.max_radius, 6371.0)

            self.assertEqual(test2.validate(), True)

            np.testing.assert_equal(
                test2.get_discontinuity_depths(),
                [0.0, 20.0, 35.0, 210.0, 410.0, 660.0, 2889.0, 5153.9, 6371.0])

            # check boundary cases
            self.assertEqual(test2.layer_number_above(6371), 128)
            self.assertEqual(test2.layer_number_below(0), 0)

            # evaluate at cmb
            self.assertEqual(test2.evaluate_above(2889.0, 'p'), 13.6908)
            self.assertEqual(test2.evaluate_below(2889.0, 'D'), 9.9145)
            self.assertEqual(test2.depth_at_top(50), 2393.5)
            self.assertEqual(test2.depth_at_bottom(50), 2443.0)
            self.assertEqual(test2.fix_discontinuity_depths(), False)


def suite():
    return unittest.makeSuite(TauPyVelocityModelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
