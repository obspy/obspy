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

    def test_lanczos_interpolation(self):
        """
        Tests against the instaseis implementation which should work well
        enough.
        """
        data = np.array([0.92961609, 0.31637555, 0.18391881, 0.20456028,
                         0.56772503, 0.5955447, 0.96451452, 0.6531771,
                         0.74890664, 0.65356987])
        dt = 1.0

        # Scenario 1.
        new_dt = 0.45
        a = 1

        expected_output = np.array([
            0.92961609, 0.55712768, 0.31720733, 0.24275977, 0.17825931,
            0.16750234, 0.17561933, 0.20626905, 0.37726064, 0.5647072,
            0.47145546, 0.59222238, 0.58665834, 0.91241347, 0.79909224,
            0.61631275, 0.61258393, 0.61611633, 0.73239733, 0.56371682,
            0.65356987])

        output = lanczos_interpolation(
            data, old_dt=dt, new_start=0.0, old_start=0.0, new_dt=new_dt,
            new_npts=21, a=a)
        np.testing.assert_allclose(output, expected_output, atol=1E-9)

        # Scenario 2.
        new_dt = 0.72
        a = 12

        expected_output = np.array([
            0.92961609, 0.54632548, 0.14335148, 0.19675436, 0.19030867,
            0.41722415, 0.60644459, 0.6018648,  0.88751628, 0.90970863,
            0.58602723, 0.71521445, 0.83288791])

        output = lanczos_interpolation(
            data, old_dt=dt, new_start=0.0, old_start=0.0, new_dt=new_dt,
            new_npts=13, a=a)
        np.testing.assert_allclose(output, expected_output, atol=1E-9)

    def test_lanczos_interpolation_units(self):
        """
        Regression test for a bug that manifested when the original sampling
        rate is not 1 Hertz and new and old start times are not identical.
        """
        # Generate a highly oversampled sine curve. Upsample and downsample
        # it again and it should not change a lot except at the boundaries.
        # Also shift a bit to trigger the bug.
        original_dt = 13.333
        new_dt = 17.23

        data = np.sin(np.linspace(0, 2 * np.pi, 1000))

        output = lanczos_interpolation(
            data, old_dt=original_dt, new_start=10 * original_dt,
            old_start=0.0, a=20,
            new_dt=new_dt, new_npts=int(990 * original_dt / new_dt))
        output = lanczos_interpolation(
            output, old_dt=new_dt, new_start=10 * original_dt,
            old_start=0.0, a=20,
            new_dt=original_dt, new_npts=int(980 * original_dt / new_dt) - 1)

        np.testing.assert_allclose(data[220:620], output[200:600], atol=1E-4,
                                   rtol=1E-4)


def suite():
    return unittest.makeSuite(InterpolationTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
