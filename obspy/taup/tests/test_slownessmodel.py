#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np

from obspy.taup.slowness_layer import (SlownessLayer, SlownessModelError,
                                       bullenDepthFor, create_from_vlayer)
from obspy.taup.velocity_layer import (DEFAULT_DENSITY, DEFAULT_QP, DEFAULT_QS,
                                       VelocityLayer)


class TauPySlownessModelTestCase(unittest.TestCase):
    def test_slownesslayer(self):
        vLayer = np.array([(10, 31, 3, 5, 2, 4,
                            DEFAULT_DENSITY, DEFAULT_DENSITY,
                            DEFAULT_QP, DEFAULT_QP,
                            DEFAULT_QS, DEFAULT_QS)],
                          dtype=VelocityLayer)
        a = create_from_vlayer(vLayer, True)
        self.assertEqual(a['botP'], 1268.0)
        self.assertEqual(a['botDepth'], 31.0)
        b = create_from_vlayer(vLayer, False)
        self.assertEqual(b['topP'], 3180.5)


class bullenDepthForTestCase(unittest.TestCase):
    def test_overflow(self):
        sl = np.array([(2548.4, 6.546970605878823, 1846.2459389213773,
                        13.798727310994103)], dtype=SlownessLayer)
        try:
            depth = bullenDepthFor(sl, 2197.322969460689, 6371)
        except SlownessModelError:
            self.fail('SlownessModelError was incorrectly raised.')
        self.assertFalse(np.isnan(depth))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TauPySlownessModelTestCase, 'test'))
    suite.addTest(unittest.makeSuite(bullenDepthForTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
