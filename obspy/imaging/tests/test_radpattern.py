# -*- coding: utf-8 -*-
"""
The obspy.imaging.radiation_pattern test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import matplotlib.pyplot as plt
from obspy.imaging.radpattern import farfieldP,farfieldS
from mpl_toolkits.mplot3d import Axes3D


class RadPatternTestCase(unittest.TestCase):
    """
    Test cases for radiation_pattern.
    """

    def test_farfield(self):
        """
        Tests to plot P/S wave farfield radiation pattern
        """
        vlength = 0.05
        mt = [0.91, -0.89, -0.02, 1.78, -1.55, 0.47]
        pointsp,dispp = farfieldP(mt)
        pointss,disps = farfieldS(mt)

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.quiver(pointsp[0],pointsp[1],pointsp[2],dispp[0],dispp[1],dispp[2],length=vlength)

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.quiver(pointss[0],pointss[1],pointss[2],disps[0],disps[1],disps[2],length=vlength)

        plt.show()

def suite():
    return unittest.makeSuite(RadPatternTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
