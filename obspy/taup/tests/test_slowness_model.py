#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np

from obspy.taup import _DEFAULT_VALUES
from obspy.taup.slowness_layer import (SlownessLayer, SlownessModelError,
                                       bullen_depth_for, create_from_vlayer)
from obspy.taup.velocity_layer import VelocityLayer


class TauPySlownessModelTestCase(unittest.TestCase):
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
        self.assertEqual(a['bot_p'], 1268.0)
        self.assertEqual(a['bot_depth'], 31.0)
        b = create_from_vlayer(v_layer, False, radius_of_planet=6371.0)
        self.assertEqual(b['top_p'], 3180.5)


class BullenDepthForTestCase(unittest.TestCase):
    def test_overflow(self):
        sl = np.array([(2548.4, 6.546970605878823, 1846.2459389213773,
                        13.798727310994103)], dtype=SlownessLayer)
        try:
            depth = bullen_depth_for(sl, 2197.322969460689, 6371)
        except SlownessModelError:
            self.fail('SlownessModelError was incorrectly raised.')
        self.assertFalse(np.isnan(depth))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TauPySlownessModelTestCase, 'test'))
    suite.addTest(unittest.makeSuite(BullenDepthForTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
