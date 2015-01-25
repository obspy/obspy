#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *
import unittest

from taupy.SlownessLayer import create_from_vlayer
from taupy.VelocityLayer import VelocityLayer


class TestSlownessModel(unittest.TestCase):
    """"
    WARNING: The values I'm testing can't be right. Half of the methods
    needed by createSample aren't implemented yet! However, as that is
    needed in the constructor of the SlownessModel, the other methods can't
    be tested independently of it. So I can test some (probably) always true
    boundary, but the intermediate testing values should at some point,
    as work progresses, start to throw errors. I could true unit tests,
    but the effort doesn't seem worth it at the moment.
    """

    # noinspection PyCallByClass
    def test_slownesslayer(self):
        vLayer = VelocityLayer(1, 10, 31, 3, 5, 2, 4)
        a = create_from_vlayer(vLayer, True)
        self.assertEqual(a.botP, 1268.0)
        self.assertEqual(a.botDepth, 31.0)
        b = create_from_vlayer(vLayer, False)
        self.assertEqual(b.topP, 3180.5)

if __name__ == '__main__':
    unittest.main(buffer=True)
