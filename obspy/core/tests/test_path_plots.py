# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy import read_inventory


class PathPlottingTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.geodetics
    """
    def test_pathplotting(self):
        inv = read_inventory('data/IU.xml')
        evcoords = 20., 30., 100.
        inv.plot_rays(evcoords, kind='mayavi')


def suite():
    return unittest.makeSuite(PathPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
