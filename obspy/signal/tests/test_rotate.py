#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Rotate test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2

import gzip
import itertools
import os
import unittest

import numpy as np

from obspy.signal.rotate import (rotate_lqt_zne, rotate_ne_rt, rotate_rt_ne,
                                 rotate_zne_lqt, _dip_azimuth2zne_base_vector,
                                 rotate2zne)


class RotateTestCase(unittest.TestCase):
    """
    Test cases for Rotate.
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_rotate_ne_rt_vs_pitsa(self):
        """
        Test horizontal component rotation against PITSA.
        """
        # load test files
        with gzip.open(os.path.join(self.path, 'rjob_20051006_n.gz')) as f:
            data_n = np.loadtxt(f)
        with gzip.open(os.path.join(self.path, 'rjob_20051006_e.gz')) as f:
            data_e = np.loadtxt(f)
        # test different angles, one from each sector
        for angle in [30, 115, 185, 305]:
            # rotate traces
            datcorr_r, datcorr_t = rotate_ne_rt(data_n, data_e, angle)
            # load pitsa files
            with gzip.open(os.path.join(self.path,
                                        'rjob_20051006_r_%sdeg.gz' %
                                        angle)) as f:
                data_pitsa_r = np.loadtxt(f)
            with gzip.open(os.path.join(self.path,
                                        'rjob_20051006_t_%sdeg.gz' %
                                        angle)) as f:
                data_pitsa_t = np.loadtxt(f)
            # Assert.
            self.assertTrue(np.allclose(datcorr_r, data_pitsa_r, rtol=1E-3,
                                        atol=1E-5))
            self.assertTrue(np.allclose(datcorr_t, data_pitsa_t, rtol=1E-3,
                                        atol=1E-5))

    def test_rotate_zne_lqt_vs_pitsa(self):
        """
        Test LQT component rotation against PITSA. Test back-rotation.
        """
        # load test files
        with gzip.open(os.path.join(self.path, 'rjob_20051006.gz')) as f:
            data_z = np.loadtxt(f)
        with gzip.open(os.path.join(self.path, 'rjob_20051006_n.gz')) as f:
            data_n = np.loadtxt(f)
        with gzip.open(os.path.join(self.path, 'rjob_20051006_e.gz')) as f:
            data_e = np.loadtxt(f)
        # test different backazimuth/incidence combinations
        for ba, inci in ((60, 130), (210, 60)):
            # rotate traces
            data_l, data_q, data_t = \
                rotate_zne_lqt(data_z, data_n, data_e, ba, inci)
            # rotate traces back to ZNE
            data_back_z, data_back_n, data_back_e = \
                rotate_lqt_zne(data_l, data_q, data_t, ba, inci)
            # load pitsa files
            with gzip.open(os.path.join(self.path,
                                        'rjob_20051006_q_%sba_%sinc.gz' %
                                        (ba, inci))) as f:
                data_pitsa_q = np.loadtxt(f)
            with gzip.open(os.path.join(self.path,
                                        'rjob_20051006_t_%sba_%sinc.gz' %
                                        (ba, inci))) as f:
                data_pitsa_t = np.loadtxt(f)
            with gzip.open(os.path.join(self.path,
                                        'rjob_20051006_l_%sba_%sinc.gz' %
                                        (ba, inci))) as f:
                data_pitsa_l = np.loadtxt(f)
            # Assert the output. Has to be to rather low accuracy due to
            # rounding error prone rotation and single precision value.
            self.assertTrue(
                np.allclose(data_l, data_pitsa_l, rtol=1E-3, atol=1E-5))
            self.assertTrue(
                np.allclose(data_q, data_pitsa_q, rtol=1E-3, atol=1E-5))
            self.assertTrue(
                np.allclose(data_t, data_pitsa_t, rtol=1E-3, atol=1E-5))
            self.assertTrue(
                np.allclose(data_z, data_back_z, rtol=1E-3, atol=1E-5))
            self.assertTrue(
                np.allclose(data_n, data_back_n, rtol=1E-3, atol=1E-5))
            self.assertTrue(
                np.allclose(data_e, data_back_e, rtol=1E-3, atol=1E-5))

    def test_rotate_ne_rt_ne(self):
        """
        Rotating there and back with the same back-azimuth should not change
        the data.
        """
        # load the data
        with gzip.open(os.path.join(self.path, 'rjob_20051006_n.gz')) as f:
            data_n = np.loadtxt(f)
        with gzip.open(os.path.join(self.path, 'rjob_20051006_e.gz')) as f:
            data_e = np.loadtxt(f)
        # Use double precision to get more accuracy for testing.
        data_n = np.require(data_n, np.float64)
        data_e = np.require(data_e, np.float64)
        ba = 33.3
        new_n, new_e = rotate_ne_rt(data_n, data_e, ba)
        new_n, new_e = rotate_rt_ne(new_n, new_e, ba)
        self.assertTrue(np.allclose(data_n, new_n, rtol=1E-7, atol=1E-12))
        self.assertTrue(np.allclose(data_e, new_e, rtol=1E-7, atol=1E-12))

    def test_rotate2zne_round_trip(self):
        """
        The rotate2zne() function has an inverse argument. Thus round
        tripping should work.
        """
        z = np.ones(10, dtype=np.float64)
        n = 2.0 * np.ones(10, dtype=np.float64)
        e = 3.0 * np.ones(10, dtype=np.float64)

        # Random values.
        dip_1, dip_2, dip_3 = 0.0, 30.0, 60.0
        azi_1, azi_2, azi_3 = 0.0, 170.0, 35.0

        a, b, c = rotate2zne(z, azi_1, dip_1, n, azi_2, dip_2, e, azi_3, dip_3)

        z_new, n_new, e_new = rotate2zne(a, azi_1, dip_1,
                                         b, azi_2, dip_2,
                                         c, azi_3, dip_3,
                                         inverse=True)

        self.assertTrue(np.allclose(z, z_new, rtol=1E-7, atol=1e-7))
        self.assertTrue(np.allclose(n, n_new, rtol=1E-7, atol=1e-7))
        self.assertTrue(np.allclose(e, e_new, rtol=1E-7, atol=1e-7))

    def test_rotate2zne_raise(self):
        """
        Check that rotate2zne() raises on unequal lengths of data.
        """
        z = np.ones(3, dtype=np.float64)
        n = np.ones(5, dtype=np.float64)
        e = np.ones(3, dtype=np.float64)

        # Random values.
        dip_1, dip_2, dip_3 = 0.0, 30.0, 60.0
        azi_1, azi_2, azi_3 = 0.0, 170.0, 35.0

        if PY2:
            testmethod = self.assertRaisesRegexp
        else:
            testmethod = self.assertRaisesRegex
        testmethod(
            ValueError, 'All three data arrays must be of same length.',
            rotate2zne, z, azi_1, dip_1, n, azi_2, dip_2, e, azi_3, dip_3)

    def test_base_vector_calculation_simple_cases(self):
        """
        Tests the _dip_azimuth2zne_base_vector() with some simple cases.
        """
        # Up and down.
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(-90, 0),
                                   [1.0, 0.0, 0.0], atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(90, 0),
                                   [-1.0, 0.0, 0.0], atol=1E-10)
        # North and South.
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 0),
                                   [0.0, 1.0, 0.0], atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 180),
                                   [0.0, -1.0, 0.0], atol=1E-10)
        # East and West.
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 90),
                                   [0.0, 0.0, 1.0], atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 270),
                                   [0.0, 0.0, -1.0], atol=1E-10)

        # Normalizing helper.
        def _n(v):
            return np.array(v) / np.linalg.norm(np.array(v))

        # 4 corners in the plain.
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 45),
                                   _n([0.0, 1.0, 1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 135),
                                   _n([0.0, -1.0, 1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 225),
                                   _n([0.0, -1.0, -1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(0, 315),
                                   _n([0.0, 1.0, -1.0]), atol=1E-10)

        # 4 corners in the top.
        dip = np.rad2deg(np.arctan2(1, np.sqrt(2)))
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(-dip, 45),
                                   _n([1.0, 1.0, 1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(-dip, 135),
                                   _n([1.0, -1.0, 1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(-dip, 225),
                                   _n([1.0, -1.0, -1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(-dip, 315),
                                   _n([1.0, 1.0, -1.0]), atol=1E-10)

        # And in the bottom
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(dip, 45),
                                   _n([-1.0, 1.0, 1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(dip, 135),
                                   _n([-1.0, -1.0, 1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(dip, 225),
                                   _n([-1.0, -1.0, -1.0]), atol=1E-10)
        np.testing.assert_allclose(_dip_azimuth2zne_base_vector(dip, 315),
                                   _n([-1.0, 1.0, -1.0]), atol=1E-10)

    def test_base_vector_from_azimuth_and_dip_calculation(self):
        """
        Tests the _dip_azimuth2zne_base_vector() method against a solution
        from the Wieland book.
        """
        dip = - (90.0 - np.rad2deg(np.arctan(np.sqrt(2.0))))

        v1 = _dip_azimuth2zne_base_vector(dip, -90.0)
        v2 = _dip_azimuth2zne_base_vector(dip, 30.0)
        v3 = _dip_azimuth2zne_base_vector(dip, 150.0)

        v1_ref = np.array([np.sqrt(2.0), 0.0, -2.0]) / np.sqrt(6.0)
        v2_ref = np.array([np.sqrt(2.0), np.sqrt(3.0), 1.0]) / np.sqrt(6.0)
        v3_ref = np.array([np.sqrt(2.0), -np.sqrt(3.0), 1.0]) / np.sqrt(6.0)

        self.assertTrue(np.allclose(v1, v1_ref, rtol=1E-7, atol=1E-7))
        self.assertTrue(np.allclose(v2, v2_ref, rtol=1E-7, atol=1E-7))
        self.assertTrue(np.allclose(v3, v3_ref, rtol=1E-7, atol=1E-7))

    def test_galperin_configuration(self):
        """
        Equal arrays on a Galperin configuration should result in only the
        vertical component remaining.
        """
        dip = - (90.0 - np.rad2deg(np.arctan(np.sqrt(2.0))))

        u = np.array([1.0, 0.0, 1.0])
        v = np.array([1.0, 1.0, -1.0])
        w = np.array([1.0, -1.0, -1.0])

        z, n, e = rotate2zne(
            u, -90, dip,
            v, 30, dip,
            w, 150, dip)

        fac = 1.0 / np.sqrt(6.0)

        z_ref = np.array([fac * 3.0 * np.sqrt(2.0), 0.0, -fac * np.sqrt(2.0)])
        n_ref = np.array([0.0, fac * 2.0 * np.sqrt(3.0), 0.0])
        e_ref = np.array([0.0, 0.0, -4.0 * fac])

        self.assertTrue(np.allclose(z, z_ref, rtol=1E-7, atol=1E-7))
        self.assertTrue(np.allclose(n, n_ref, rtol=1E-7, atol=1E-7))
        self.assertTrue(np.allclose(e, e_ref, rtol=1E-7, atol=1E-7))

    def test_rotate2zne_against_rotate_ne_rt(self):
        np.random.seed(123)
        z = np.random.random(10)
        n = np.random.random(10)
        e = np.random.random(10)

        for ba in [0.0, 14.325, 38.234, 78.1, 90.0, 136.3435, 265.4, 351.35]:
            r, t = rotate_ne_rt(n=n, e=e, ba=ba)

            # Unrotate with rotate2zne() - this should make sure the azimuth is
            # interpreted correctly.
            z_new, n_new, e_new = rotate2zne(z, 0, -90,
                                             r, ba + 180, 0,
                                             t, ba + 270, 0)
            np.testing.assert_allclose(z_new, z)
            np.testing.assert_allclose(n_new, n)
            np.testing.assert_allclose(e_new, e)

    def test_rotate2zne_against_ne_rt_picking_any_two_horizontal_comps(self):
        """
        This also tests non-orthogonal configurations to some degree.
        """
        np.random.seed(456)
        z = np.random.random(10)
        n = np.random.random(10)
        e = np.random.random(10)

        # Careful to not pick any coordinate axes.
        for ba in [14.325, 38.234, 78.1, 136.3435, 265.4, 351.35]:
            r, t = rotate_ne_rt(n=n, e=e, ba=ba)

            _r = [r, ba + 180, 0]
            _t = [t, ba + 270, 0]
            _n = [n, 0, 0]
            _e = [e, 90, 0]

            # Picking any two should be enough to reconstruct n and e.
            for a, b in itertools.permutations([_r, _t, _n, _e], 2):
                z_new, n_new, e_new = rotate2zne(z, 0, -90,
                                                 a[0], a[1], a[2],
                                                 b[0], b[1], b[2])
                np.testing.assert_allclose(z_new, z)
                np.testing.assert_allclose(n_new, n)
                np.testing.assert_allclose(e_new, e)

    def test_rotate2zne_against_lqt(self):
        np.random.seed(789)
        z = np.random.random(10)
        n = np.random.random(10)
        e = np.random.random(10)

        bas = [0.0, 14.325, 38.234, 78.1, 90.0, 136.3435, 265.4, 180.0,
               351.35, 360.0]
        incs = [0.0, 10.325, 32.23, 88.1, 90.0, 132.3435, 245.4, 180.0,
                341.35, 360.0]

        for ba, inc in itertools.product(bas, incs):
            l, q, t = rotate_zne_lqt(z=z, n=n, e=e, ba=ba, inc=inc)

            dip_l = (inc % 180.0) - 90.0
            if 180 <= inc < 360:
                dip_l *= -1.0

            dip_q = ((inc + 90) % 180) - 90
            if 0 < inc < 90 or 270 <= inc < 360:
                dip_q *= -1.0

            az_l = ba + 180.0
            az_q = ba

            # Azimuths flip depending on the incidence angle.
            if inc > 180:
                az_l += 180
            if 90 < inc <= 270:
                az_q += 180

            z_new, n_new, e_new = rotate2zne(l, az_l, dip_l,
                                             q, az_q, dip_q,
                                             t, ba + 270, 0)
            np.testing.assert_allclose(z_new, z)
            np.testing.assert_allclose(n_new, n)
            np.testing.assert_allclose(e_new, e)

    def test_rotate2zne_against_lqt_different_combinations(self):
        np.random.seed(101112)
        z = np.random.random(10)
        n = np.random.random(10)
        e = np.random.random(10)

        # Exclude coordinate axis.
        bas = [14.325, 38.234, 78.1, 136.3435, 265.4, 351.35]
        incs = [10.325, 32.23, 88.1, 132.3435, 245.4, 341.35]

        success_count = 0
        failure_count = 0

        for ba, inc in itertools.product(bas, incs):
            l, q, t = rotate_zne_lqt(z=z, n=n, e=e, ba=ba, inc=inc)

            dip_l = (inc % 180.0) - 90.0
            if 180 <= inc < 360:
                dip_l *= -1.0

            dip_q = ((inc + 90) % 180) - 90
            if 0 < inc < 90 or 270 <= inc < 360:
                dip_q *= -1.0

            az_l = ba + 180.0
            az_q = ba

            # Azimuths flip depending on the incidence angle.
            if inc > 180:
                az_l += 180
            if 90 < inc <= 270:
                az_q += 180

            _z = [z, 0, -90, "Z"]
            _n = [n, 0, 0, "N"]
            _e = [e, 90, 0, "E"]
            _l = [l, az_l, dip_l, "L"]
            _q = [q, az_q, dip_q, "Q"]
            _t = [t, ba + 270, 0, "T"]

            # Any three of them (except three horizontal ones) should be
            # able to reconstruct ZNE.
            for a, b, c in itertools.permutations([_l, _q, _t, _z, _n, _e], 3):

                # Three horizontal components are linearly dependent, as are
                # Z, Q, and L.
                if (a[2] == b[2] == c[2] == 0 or
                    set([_i[3] for _i in (a, b, c)]) == set(["Z", "Q", "L"])):
                    with self.assertRaises(ValueError) as err:
                        rotate2zne(a[0], a[1], a[2],
                                   b[0], b[1], b[2],
                                   c[0], c[1], c[2])
                    self.assertTrue(err.exception.args[0].startswith(
                        "The given directions are not linearly independent, "
                        "at least within numerical precision. Determinant of "
                        "the base change matrix:"))
                    failure_count += 1
                    continue

                z_new, n_new, e_new = rotate2zne(a[0], a[1], a[2],
                                                 b[0], b[1], b[2],
                                                 c[0], c[1], c[2])
                np.testing.assert_allclose(z_new, z)
                np.testing.assert_allclose(n_new, n)
                np.testing.assert_allclose(e_new, e)
                success_count += 1
        # Make sure it actually tested all combinations.
        self.assertEqual(success_count, 3888)
        # Also the linearly dependent variants.
        self.assertEqual(failure_count, 432)


def suite():
    return unittest.makeSuite(RotateTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
