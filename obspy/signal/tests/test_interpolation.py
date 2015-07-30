#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The interpolation test suite for ObsPy.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import numpy as np
from obspy.signal.interpolation import (lanczos_interpolation,
                                        calculate_lanczos_kernel)


class InterpolationTestCase(unittest.TestCase):
    """
    Interpolation test case
    """
    def test_calculate_lanczos_kernel(self):
        """
        Tests the kernels implemented in C against their numpy counterpart.
        """
        x = np.linspace(-5, 5, 11)

        values = calculate_lanczos_kernel(x, 5, "hanning")
        np.testing.assert_allclose(
            values["only_sinc"], np.sinc(x), atol=1E-9)
        np.testing.assert_allclose(
            values["only_taper"], np.hanning(len(x)), atol=1E-9)
        np.testing.assert_allclose(
            values["full_kernel"], np.sinc(x) * np.hanning(len(x)),
            atol=1E-9)

        values = calculate_lanczos_kernel(x, 5, "blackman")
        np.testing.assert_allclose(
            values["only_sinc"], np.sinc(x), atol=1E-9)
        np.testing.assert_allclose(
            values["only_taper"], np.blackman(len(x)), atol=1E-9)
        np.testing.assert_allclose(
            values["full_kernel"], np.sinc(x) * np.blackman(len(x)),
            atol=1E-9)

        values = calculate_lanczos_kernel(x, 5, "lanczos")
        np.testing.assert_allclose(
            values["only_sinc"], np.sinc(x), atol=1E-9)
        np.testing.assert_allclose(
            values["only_taper"], np.sinc(x / 5.0), atol=1E-9)
        np.testing.assert_allclose(
            values["full_kernel"], np.sinc(x) * np.sinc(x / 5.0),
            atol=1E-9)


def suite():
    return unittest.makeSuite(InterpolationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
