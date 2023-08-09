#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The Rotate test suite.
"""
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
        np.random.seed(45645623)
        z = np.random.random(10)
        n = np.random.random(10)
        e = np.random.random(10)

        for _ in range(100):
            # The risk of producing linear dependent directions is very
            # small (the seed value should also prevent it across machines).
            dip_1, dip_2, dip_3 = np.random.random(3) * 180.0 - 90.0
            azi_1, azi_2, azi_3 = np.random.random(3) * 360.0

            a, b, c = rotate2zne(z, azi_1, dip_1,
                                 n, azi_2, dip_2,
                                 e, azi_3, dip_3)

            z_new, n_new, e_new = rotate2zne(a, azi_1, dip_1,
                                             b, azi_2, dip_2,
                                             c, azi_3, dip_3,
                                             inverse=True)

            np.testing.assert_allclose(z, z_new, rtol=1E-7, atol=1e-7)
            np.testing.assert_allclose(n, n_new, rtol=1E-7, atol=1e-7)
            np.testing.assert_allclose(e, e_new, rtol=1E-7, atol=1e-7)

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

        self.assertRaisesRegex(
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
                if a[2] == b[2] == c[2] == 0 or \
                        set([_i[3] for _i in (a, b, c)]) == \
                        set(["Z", "Q", "L"]):
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

    def test_with_real_data(self):
        # Filtered and downsampled test data with two co-located
        # seismometers on a step table. One (UVW) in Galperin configuration,
        # the other (XYZ) oriented 25 degree towards counter-clockwise from
        # North.
        #
        # Original data can be found here:
        # http://examples.obspy.org/step_table_galperin_and_xyz.mseed
        #
        # Picture of setup:
        # http://examples.obspy.org/step_table_galperin_and_xyz.jpg
        u = np.array([
            -887.77005805, 7126.9690531, 48436.17065483, 138585.24660557,
            220190.69362083, 179040.5715419, -21365.23030094,
            -253885.25529288, -344888.20815164, -259362.36082208,
            -117476.30748613, - 42988.81966958, -45995.43307308,
            -57130.87444412, -30545.75344533, 16298.87665025])
        v = np.array([
            2.33308511e+02, 8.16259596e+03, 5.11074487e+04, 1.48229541e+05,
            2.41322335e+05, 2.08201013e+05, 4.93732289e+03, -2.39867750e+05,
            -3.44167596e+05, -2.66558032e+05, -1.27714987e+05, -5.54712804e+04,
            -6.19973652e+04, -7.66740787e+04, -5.17925310e+04,
            -4.71673443e+03])
        w = np.array([
            1692.48532892, 9875.6136413, 53089.61423663, 149373.52749023,
            240009.30157128, 204362.69005767, 1212.47406863, -239380.57384624,
            -336783.01040666, -252884.65411222, -110766.44577398,
            -38182.18142102, -45729.92956198, -61691.87092415,
            -38434.81993441, 6224.73096858])
        x = np.array([
            -1844.85832046, -1778.44974024, -2145.6117388, -4325.27560474,
            -8278.1836905, -10842.53378841, -8565.12113951, -2024.53838011,
            4439.22322848, 7340.96354878, 7081.65449722, 6303.91640198,
            6549.98692684, 7223.59663617, 7133.72073748, 6068.56702479])
        y = np.array([
            -242.70568894, -458.3864756, -351.75925077, 2142.51669733,
            8287.98182002, 15822.24351111, 20151.78532927, 18511.90136103,
            12430.22438956, 5837.66044337, 1274.9580289, -1597.06115226,
            -4331.40686142, -7529.87533286, -10544.34374306, -12656.77586305])
        z = np.array([
            5.79050980e+02, 1.45190734e+04, 8.85582128e+04, 2.53690907e+05,
            4.08578800e+05, 3.45046937e+05, -8.15914926e+03, -4.26449298e+05,
            -5.97207861e+05, -4.53464470e+05, -2.07176498e+05, -7.94526512e+04,
            -8.95206215e+04, -1.14008287e+05, -7.05797830e+04, 1.01175730e+04])

        # Component orientations.
        u = (u, 90.0, -(90.0 - 54.7), "U")
        v = (v, 330.0, -(90.0 - 54.7), "V")
        w = (w, 210.0, -(90.0 - 54.7), "W")
        x = (x, 65.0, 0.0, "X")
        y = (y, 335.0, 0.0, "Y")
        z = (z, 0.0, -90.0, "Z")

        # Any three should result in the same ZNE.
        success_count = 0
        failure_count = 0
        for a, b, c in itertools.permutations([x, y, z, u], 3):
            # Except if "X" and "Y" are both part of it because they don't
            # really contain any data (vertical step table).
            if set(["X", "Y"]).issubset(set([a[-1], b[-1], c[-1]])):
                failure_count += 1
                continue

            z_new, _, _ = rotate2zne(
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])
            np.testing.assert_allclose(z_new, z[0], rtol=1E-5)
            success_count += 1

            # Sanity check that it fails for slightly different rotations.
            z_new, _, _ = rotate2zne(
                a[0], a[1] + 1.5, a[2] - 1.5,
                b[0], b[1] - 0.7, b[2] + 1.2,
                c[0], c[1] + 1.0, c[2] - 0.4)
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(z_new, z[0], rtol=1E-5)

        self.assertEqual(success_count, 12)
        self.assertEqual(failure_count, 12)
