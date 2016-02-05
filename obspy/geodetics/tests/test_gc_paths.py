# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import unittest
import warnings

from obspy.geodetics.gc_paths import plot_rays
from obspy import read_inventory


class GcPathsTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    def test_gc_paths(self):
        inv = read_inventory('data/IU.xml')
        evlat, evlon, evdepth = 20., 30., 100.
        plot_rays(evlat, evlon, evdepth, inv)


def suite():
    return unittest.makeSuite(GcPathsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
