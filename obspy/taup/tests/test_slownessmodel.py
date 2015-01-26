#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.taup.slowness_layer import create_from_vlayer
from obspy.taup.velocity_layer import VelocityLayer


class TauPySlownessModelTestCase(unittest.TestCase):
    def test_slownesslayer(self):
        vLayer = VelocityLayer(1, 10, 31, 3, 5, 2, 4)
        a = create_from_vlayer(vLayer, True)
        self.assertEqual(a.botP, 1268.0)
        self.assertEqual(a.botDepth, 31.0)
        b = create_from_vlayer(vLayer, False)
        self.assertEqual(b.topP, 3180.5)


def suite():
    return unittest.makeSuite(TauPySlownessModelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
