#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the channel handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
from obspy.station import read_inventory
import os
from matplotlib import rcParams
import numpy as np
import unittest
from obspy.core.util.testing import ImageComparison, getMatplotlibVersion
import warnings


MATPLOTLIB_VERSION = getMatplotlibVersion()


class ChannelTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.channel.Channel` class.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.2:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 4, 2]:
            reltol = 2.0

        cha = read_inventory()[0][0][0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "channel_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                cha.plot(0.005, outfile=ic.name)


def suite():
    return unittest.makeSuite(ChannelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
