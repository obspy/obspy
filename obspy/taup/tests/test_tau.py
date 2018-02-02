#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the high level obspy.taup.tau interface.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import collections
import inspect
import os
import unittest
import warnings
from collections import OrderedDict

import numpy as np

from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.taup import TauPyModel
from obspy.taup.tau import Arrivals
from obspy.taup.taup_create import build_taup_model
import obspy.geodetics.base as geodetics


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data", "TauP_test_data")

# checks every x. entry in _compare_against_ak135_tables_kennet - set to 1 in
# order to check the full table - a factor > 20 does not improve speed much
SPEEDUP_FACTOR = 20


class TauPyModelTestCase(unittest.TestCase):
    """
    Test suite for the obspy.taup.TauPy class.
    """

    def setUp(self):
        """setup method. Instantiates cache values to be used in
        P phase arrival calculations
        to test also `TauModel.load_from_depth_cache`"""
        self.caches = [OrderedDict(), False, None]

    def _read_taup_output(self, filename):
        """
        Helper method reading a stdout capture of TauP.
        """
        with open(os.path.join(DATA, filename), "rb") as fh:
            while True:
                line = fh.readline().strip()
                if line.startswith(b"-----"):
                    break

            output = np.genfromtxt(
                fh,
                usecols=[0, 1, 2, 3, 4, 5, 6, 7, 9],
                dtype=[(native_str('distance'), np.float_),
                       (native_str('depth'), np.float_),
                       (native_str('name'), (np.str_, 10)),
                       (native_str('time'), np.float_),
                       (native_str('ray_param_sec_degree'), np.float_),
                       (native_str('takeoff_angle'), np.float_),
                       (native_str('incident_angle'), np.float_),
                       (native_str('purist_distance'), np.float_),
                       (native_str('purist_name'), (np.str_, 10))])

        output = np.atleast_1d(output)
        return output

    def _compare_arrivals_with_file(self, arrivals, filename):
        """
        Helper method comparing arrivals against the phases stored in a file.
        """
        arrivals = sorted(arrivals, key=lambda x: (x.time, x.name))

        expected_arrivals = self._read_taup_output(filename)
        expected_arrivals = sorted(expected_arrivals,
                                   key=lambda x: (x["time"], x["name"]))

        for arr, expected_arr in zip(arrivals, expected_arrivals):
            self._assert_arrivals_equal(arr, expected_arr)

    def _assert_arrivals_equal(self, arr, expected_arr):
        """
        Helper method comparing an arrival object and a dictionary of expected
        arrivals.
        """
        # Zero travel time result in the other parameters being undefined.
        if arr.time == 0.0:
            return
        self.assertEqual(arr.distance, expected_arr["distance"])
        self.assertEqual(arr.source_depth, expected_arr["depth"])
        self.assertEqual(arr.name, expected_arr["name"])
        self.assertAlmostEqual(arr.time, expected_arr["time"], 2)
        self.assertAlmostEqual(arr.ray_param_sec_degree,
                               expected_arr["ray_param_sec_degree"], 3)
        self.assertAlmostEqual(arr.takeoff_angle,
                               expected_arr["takeoff_angle"], 2)
        self.assertAlmostEqual(arr.incident_angle,
                               expected_arr["incident_angle"], 2)
        self.assertAlmostEqual(arr.purist_distance,
                               expected_arr["purist_distance"], 2)
        self.assertEqual(arr.purist_name, expected_arr["purist_name"])

    def test_p_iasp91_manual(self):
        """
        Manual test for P phase in IASP91.
        """
        m = TauPyModel(model="iasp91")
        arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                      distance_in_degree=35.0,
                                      phase_list=["P"])
        self.assertEqual(len(arrivals), 1)
        p_arrival = arrivals[0]

        self.assertEqual(p_arrival.name, "P")
        self.assertAlmostEqual(p_arrival.time, 412.43, 2)
        self.assertAlmostEqual(p_arrival.ray_param_sec_degree, 8.613, 3)
        self.assertAlmostEqual(p_arrival.takeoff_angle, 26.74, 2)
        self.assertAlmostEqual(p_arrival.incident_angle, 26.70, 2)
        self.assertAlmostEqual(p_arrival.purist_distance, 35.00, 2)
        self.assertEqual(p_arrival.purist_name, "P")

    @unittest.skipIf(not geodetics.HAS_GEOGRAPHICLIB,
                     'Module geographiclib is not installed')
    def test_p_iasp91_geo_manual(self):
        """
        Manual test for P phase in IASP91 given geographical input.

        This version of the test is used when geographiclib is installed
        """
        m = TauPyModel(model="iasp91")
        arrivals = m.get_travel_times_geo(source_depth_in_km=10.0,
                                          source_latitude_in_deg=20.0,
                                          source_longitude_in_deg=33.0,
                                          receiver_latitude_in_deg=55.0,
                                          receiver_longitude_in_deg=33.0,
                                          phase_list=["P"])
        self.assertEqual(len(arrivals), 1)
        p_arrival = arrivals[0]

        self.assertEqual(p_arrival.name, "P")
        self.assertAlmostEqual(p_arrival.time, 412.43, 2)
        self.assertAlmostEqual(p_arrival.ray_param_sec_degree, 8.613, 3)
        self.assertAlmostEqual(p_arrival.takeoff_angle, 26.74, 2)
        self.assertAlmostEqual(p_arrival.incident_angle, 26.70, 2)
        self.assertAlmostEqual(p_arrival.purist_distance, 35.00, 2)
        self.assertEqual(p_arrival.purist_name, "P")

    def test_p_iasp91_geo_fallback_manual(self):
        """
        Manual test for P phase in IASP91 given geographical input.

        This version of the test checks that things still work when
        geographiclib is not installed.
        """
        has_geographiclib_real = geodetics.HAS_GEOGRAPHICLIB
        geodetics.HAS_GEOGRAPHICLIB = False
        m = TauPyModel(model="iasp91")
        arrivals = m.get_travel_times_geo(source_depth_in_km=10.0,
                                          source_latitude_in_deg=20.0,
                                          source_longitude_in_deg=33.0,
                                          receiver_latitude_in_deg=55.0,
                                          receiver_longitude_in_deg=33.0,
                                          phase_list=["P"])
        geodetics.HAS_GEOGRAPHICLIB = has_geographiclib_real
        self.assertEqual(len(arrivals), 1)
        p_arrival = arrivals[0]

        self.assertEqual(p_arrival.name, "P")
        self.assertAlmostEqual(p_arrival.time, 412.43, 2)
        self.assertAlmostEqual(p_arrival.ray_param_sec_degree, 8.613, 3)
        self.assertAlmostEqual(p_arrival.takeoff_angle, 26.74, 2)
        self.assertAlmostEqual(p_arrival.incident_angle, 26.70, 2)
        self.assertAlmostEqual(p_arrival.purist_distance, 35.00, 2)
        self.assertEqual(p_arrival.purist_name, "P")

    def test_p_iasp91(self):
        """
        Test P phase arrival against TauP output in model IASP91
        with different cache values to test `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="iasp91", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["P"])
            self._compare_arrivals_with_file(arrivals,
                                             "taup_time_-h_10_-ph_P_-deg_35")

    def test_p_ak135(self):
        """
        Test P phase arrival against TauP output in model AK135
        with different cache values to test `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="ak135", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["P"])
            self._compare_arrivals_with_file(
                arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")

    def test_p_ak135f_no_mud(self):
        """
        Test P phase arrival against TauP output in model ak135f_no_mud
        with different cache values to test `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="ak135f_no_mud", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["P"])
            self._compare_arrivals_with_file(
                arrivals, "taup_time_-h_10_-ph_P_-deg_35_-mod_ak135f_no_mud")

    def test_p_jb(self):
        """
        Test P phase arrival against TauP output in model jb
        with different cache values to test `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="jb", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["P"])
            self._compare_arrivals_with_file(
                arrivals, "taup_time_-h_10_-ph_P_-deg_35_-mod_jb")

    def test_p_pwdk(self):
        """
        Test P phase arrival against TauP output in model pwdk
        with different cache values to test `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="pwdk", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["P"])
            self._compare_arrivals_with_file(
                arrivals, "taup_time_-h_10_-ph_P_-deg_35_-mod_pwdk")

    def test_iasp91(self):
        """
        Test travel times for lots of phases against output from TauP in model
        IASP91 with different cache values to test
        `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="iasp91", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["ttall"])
            self._compare_arrivals_with_file(
                arrivals, "taup_time_-h_10_-ph_ttall_-deg_35")

    def test_ak135(self):
        """
        Test travel times for lots of phases against output from TauP in model
        AK135 with different cache values to test
        `TauModel.load_from_depth_cache`
        """
        for cache in self.caches:
            m = TauPyModel(model="ak135", cache=cache)
            arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                          distance_in_degree=35.0,
                                          phase_list=["ttall"])
            self._compare_arrivals_with_file(
                arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")

    def test_pierce_p_iasp91(self):
        """
        Test single pierce point against output from TauP.
        """
        m = TauPyModel(model="iasp91")
        arrivals = m.get_pierce_points(source_depth_in_km=10.0,
                                       distance_in_degree=35.0,
                                       phase_list=["P"])
        self.assertEqual(len(arrivals), 1)
        p_arr = arrivals[0]

        # Open test file.
        filename = os.path.join(DATA, "taup_pierce_-h_10_-ph_P_-deg_35")

        expected = np.genfromtxt(filename, skip_header=1)

        np.testing.assert_almost_equal(expected[:, 0],
                                       np.degrees(p_arr.pierce['dist']), 2)
        np.testing.assert_almost_equal(expected[:, 1],
                                       p_arr.pierce['depth'], 1)
        np.testing.assert_almost_equal(expected[:, 2],
                                       p_arr.pierce['time'], 1)

    @unittest.skipIf(not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
                     'test needs geographiclib >= 1.34')
    def test_pierce_p_iasp91_geo(self):
        """
        Test single pierce point against output from TauP using geo data.

        This version of the test is used when geographiclib is installed
        """
        m = TauPyModel(model="iasp91")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arrivals = m.get_pierce_points_geo(
                source_depth_in_km=10.0,
                source_latitude_in_deg=-45.0,
                source_longitude_in_deg=-50.0,
                receiver_latitude_in_deg=-80.0,
                receiver_longitude_in_deg=-50.0, phase_list=["P"])
        self.assertEqual(len(arrivals), 1)
        p_arr = arrivals[0]

        # Open test file.
        filename = os.path.join(DATA,
                                "taup_pierce_-mod_isp91_ph_P_-h_10_-evt_" +
                                "-45_-50_-sta_-80_-50")

        expected = np.genfromtxt(filename, skip_header=1)

        np.testing.assert_almost_equal(expected[:, 0],
                                       np.degrees(p_arr.pierce['dist']), 2)
        np.testing.assert_almost_equal(expected[:, 1],
                                       p_arr.pierce['depth'], 1)
        np.testing.assert_almost_equal(expected[:, 2],
                                       p_arr.pierce['time'], 1)
        if geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34:
            np.testing.assert_almost_equal(expected[:, 3],
                                           p_arr.pierce['lat'], 1)
            np.testing.assert_almost_equal(expected[:, 4],
                                           p_arr.pierce['lon'], 1)

    def test_pierce_p_iasp91_fallback_geo(self):
        """
        Test single pierce point against output from TauP using geo data.

        This version of the test checks that things still work when
        geographiclib is not installed.
        """
        has_geographiclib_real = geodetics.HAS_GEOGRAPHICLIB
        geodetics.HAS_GEOGRAPHICLIB = False
        m = TauPyModel(model="iasp91")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arrivals = m.get_pierce_points_geo(source_depth_in_km=10.0,
                                               source_latitude_in_deg=-45.0,
                                               source_longitude_in_deg=-50.0,
                                               receiver_latitude_in_deg=-80.0,
                                               receiver_longitude_in_deg=-50.0,
                                               phase_list=["P"])
            geodetics.HAS_GEOGRAPHICLIB = has_geographiclib_real
            self.assertTrue(issubclass(w[-1].category, UserWarning))

        self.assertEqual(len(arrivals), 1)
        p_arr = arrivals[0]

        # Open test file.
        filename = os.path.join(DATA, "taup_pierce_-h_10_-ph_P_-deg_35")

        expected = np.genfromtxt(filename, skip_header=1)

        np.testing.assert_almost_equal(expected[:, 0],
                                       np.degrees(p_arr.pierce['dist']), 2)
        np.testing.assert_almost_equal(expected[:, 1],
                                       p_arr.pierce['depth'], 1)
        np.testing.assert_almost_equal(expected[:, 2],
                                       p_arr.pierce['time'], 1)
        # NB: we do not check pierce['lat'] and pierce['lon'] here, as these
        # are not calculated when geographiclib is not installed. We check
        # that they are not present.
        with self.assertRaises(ValueError):
            p_arr.pierce["lat"]
        with self.assertRaises(ValueError):
            p_arr.pierce["lon"]

    def test_vs_java_iasp91(self):
        """
        Tests the traveltime calculation against the output from TauP in the
        file 'java_tauptime_testoutput'.

        Essentially tests all kinds of depths and epicentral distances in the
        model iasp91.

        Test data generated with:

          $ phases="P,p,PcP,PcS,Pdiff,PKIKKIKP,PKIKKIKS,PKIKP,PKiKP,PKIKPPKIKP"
          $ phases="${phases},PKIKS,PKKP,PKKS,PKP,PKPPKP,PP,pP,pPdiff,pPKIKP"
          $ phases="${phases},pPKiKP,pPKP,PS,pS,pSKIKS,pSKS,S,s,ScP,ScS,Sdiff"
          $ phases="${phases},SKIKKIKP,SKIKKIKS,SKIKP,SKiKP,SKIKS,SKIKSSKIKS"
          $ phases="${phases},SKKP,SKKS,SKS,SKSSKS,SP,sP,sPdiff,sPKIKP,sPKiKP"
          $ phases="${phases},sPKP,SS,sS,sSdiff,sSKIKS,sSKS"

          $ for dist in 0 45 90 180 160; do
              for depth in 0 100 1000 2889; do
                ./taup_time -mod iasp91 -ph ${phases} -h ${depth} -deg ${dist}
              done
            done
        """
        m = TauPyModel(model="iasp91")
        filename = os.path.join(DATA, "java_tauptime_testoutput")

        expected = collections.defaultdict(list)
        all_phases = []

        with open(filename, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("----"):
                    continue
                line = line.replace("=", "")
                line = line.split()
                try:
                    float(line[0])
                except Exception:
                    continue
                expected[(float(line[0]), float(line[1]))].append({
                    "distance": float(line[0]),
                    "depth": float(line[1]),
                    "name": line[2],
                    "time": float(line[3]),
                    "ray_param_sec_degree": float(line[4]),
                    "takeoff_angle": float(line[5]),
                    "incident_angle": float(line[6]),
                    "purist_distance": float(line[7]),
                    "purist_name": line[8]})
                all_phases.append(line[2])

        all_phases = sorted(set(all_phases))

        for key, value in expected.items():
            expected[key] = sorted(value,
                                   key=lambda x: (x["time"],
                                                  x["ray_param_sec_degree"],
                                                  x["name"]))
        actual_phases = []
        for dist, depth in expected.keys():
            tt = m.get_travel_times(source_depth_in_km=depth,
                                    distance_in_degree=dist,
                                    phase_list=all_phases)
            tt = sorted(tt, key=lambda x: (
                round(x.time, 2),
                round(x.ray_param_sec_degree, 3),
                x.name))
            expected_arrivals = expected[(dist, depth)]

            for actual_arrival, expected_arrival in zip(tt, expected_arrivals):
                self._assert_arrivals_equal(actual_arrival, expected_arrival)
            for arr in tt:
                actual_phases.append(arr.name)
        actual_phases = sorted(set(actual_phases))
        self.assertEqual(actual_phases, all_phases)

    def test_pierce_all_phases(self):
        """
        Tests pierce points against those calculated in TauP.
        """
        filename = os.path.join(DATA, "java_taup_pierce_h10_deg35_ttall")
        expected = collections.defaultdict(list)
        with open(filename, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    current_phase = line[1:].strip().split()[0]
                    continue
                dist, depth, time = list(map(float, line.split()))
                expected[current_phase].append((dist, depth, time))
        expected_phases = sorted(set(expected.keys()))

        m = TauPyModel(model="iasp91")
        arrivals = m.get_pierce_points(source_depth_in_km=10.0,
                                       distance_in_degree=35.0,
                                       phase_list=["ttall"])

        # Make sure the same stuff is available.
        arrival_phases = sorted(set([_i.name for _i in arrivals]))
        self.assertEqual(expected_phases, arrival_phases)

        actual = collections.defaultdict(list)
        for arr in arrivals:
            for p in arr.pierce:
                actual[arr.name].append((
                    round(np.degrees(p['dist']), 2),
                    round(p['depth'], 1),
                    round(p['time'], 1)))

        self.assertEqual(sorted(actual.keys()), sorted(expected.keys()))

        for key in actual.keys():
            actual_values = sorted(actual[key])
            expected_values = sorted(expected[key])
            self.assertEqual(actual_values, expected_values)

    def test_single_path_iasp91(self):
        """
        Test the raypath for a single phase.
        """
        filename = os.path.join(DATA,
                                "taup_path_-o_stdout_-h_10_-ph_P_-deg_35")
        expected = np.genfromtxt(filename, comments='>')

        m = TauPyModel(model="iasp91")
        arrivals = m.get_ray_paths(source_depth_in_km=10.0,
                                   distance_in_degree=35.0, phase_list=["P"])
        self.assertEqual(len(arrivals), 1)

        # Interpolate both paths to 100 samples and make sure they are
        # approximately equal.
        sample_points = np.linspace(0, 35, 100)

        interpolated_expected = np.interp(
            sample_points,
            expected[:, 0],
            expected[:, 1])

        interpolated_actual = np.interp(
            sample_points,
            np.round(np.degrees(arrivals[0].path['dist']), 2),
            np.round(6371 - arrivals[0].path['depth'], 2))

        self.assertTrue(np.allclose(interpolated_actual,
                                    interpolated_expected, rtol=1E-4, atol=0))

    @unittest.skipIf(not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34,
                     'test needs geographiclib >= 1.34')
    def test_single_path_geo_iasp91(self):
        """
        Test the raypath for a single phase given geographical input.

        This tests the case when geographiclib is installed.
        """
        filename = os.path.join(DATA,
                                "taup_path_-mod_iasp91_-o_stdout_-h_10_" +
                                "-ph_P_-sta_-45_-60_evt_-80_-60")
        expected = np.genfromtxt(filename, comments='>')

        m = TauPyModel(model="iasp91")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arrivals = m.get_ray_paths_geo(
                source_depth_in_km=10.0,
                source_latitude_in_deg=-80.0,
                source_longitude_in_deg=-60.0,
                receiver_latitude_in_deg=-45.0,
                receiver_longitude_in_deg=-60.0, phase_list=["P"])
        self.assertEqual(len(arrivals), 1)

        # Interpolate both paths to 100 samples and make sure they are
        # approximately equal.
        sample_points = np.linspace(0, 35, 100)

        interpolated_expected_depth = np.interp(
            sample_points,
            expected[:, 0],
            expected[:, 1])
        interpolated_expected_lat = np.interp(
            sample_points,
            expected[:, 0],
            expected[:, 2])
        interpolated_expected_lon = np.interp(
            sample_points,
            expected[:, 0],
            expected[:, 3])

        interpolated_actual_depth = np.interp(
            sample_points,
            np.round(np.degrees(arrivals[0].path['dist']), 2),
            np.round(6371 - arrivals[0].path['depth'], 2))
        np.testing.assert_allclose(interpolated_actual_depth,
                                   interpolated_expected_depth,
                                   rtol=1E-4, atol=0)

        if geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34:
            interpolated_actual_lat = np.interp(
                sample_points,
                np.round(np.degrees(arrivals[0].path['dist']), 2),
                np.round(arrivals[0].path['lat'], 2))
            interpolated_actual_lon = np.interp(
                sample_points,
                np.round(np.degrees(arrivals[0].path['dist']), 2),
                np.round(arrivals[0].path['lon'], 2))
            np.testing.assert_allclose(interpolated_actual_lat,
                                       interpolated_expected_lat,
                                       rtol=1E-4, atol=0)
            np.testing.assert_allclose(interpolated_actual_lon,
                                       interpolated_expected_lon,
                                       rtol=1E-4, atol=0)

    def test_single_path_geo_fallback_iasp91(self):
        """
        Test the raypath for a single phase given geographical input.

        This version of the test checks that things still work when
        geographiclib is not installed.
        """
        has_geographiclib_real = geodetics.HAS_GEOGRAPHICLIB
        geodetics.HAS_GEOGRAPHICLIB = False
        filename = os.path.join(DATA,
                                "taup_path_-o_stdout_-h_10_-ph_P_-deg_35")
        expected = np.genfromtxt(filename, comments='>')

        m = TauPyModel(model="iasp91")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            arrivals = m.get_ray_paths_geo(source_depth_in_km=10.0,
                                           source_latitude_in_deg=-80.0,
                                           source_longitude_in_deg=-60.0,
                                           receiver_latitude_in_deg=-45.0,
                                           receiver_longitude_in_deg=-60.0,
                                           phase_list=["P"])
            geodetics.HAS_GEOGRAPHICLIB = has_geographiclib_real
            self.assertTrue(issubclass(w[-1].category, UserWarning))

        self.assertEqual(len(arrivals), 1)

        # Interpolate both paths to 100 samples and make sure they are
        # approximately equal.
        sample_points = np.linspace(0, 35, 100)

        interpolated_expected = np.interp(
            sample_points,
            expected[:, 0],
            expected[:, 1])

        interpolated_actual = np.interp(
            sample_points,
            np.round(np.degrees(arrivals[0].path['dist']), 2),
            np.round(6371 - arrivals[0].path['depth'], 2))

        np.testing.assert_allclose(interpolated_actual, interpolated_expected,
                                   rtol=1E-4, atol=0)

        # NB: we do not check path['lat'] and path['lon'] here, as these
        # are not calculated when geographiclib is not installed. We check
        # that they are not present.
        with self.assertRaises(ValueError):
            arrivals[0].path["lat"]
        with self.assertRaises(ValueError):
            arrivals[0].path["lon"]

    def test_single_path_ak135(self):
        """
        Test the raypath for a single phase. This time for model AK135.
        """
        filename = os.path.join(
            DATA, "taup_path_-o_stdout_-h_10_-ph_P_-deg_35_-mod_ak135")
        expected = np.genfromtxt(filename, comments='>')

        m = TauPyModel(model="ak135")
        arrivals = m.get_ray_paths(source_depth_in_km=10.0,
                                   distance_in_degree=35.0, phase_list=["P"])
        self.assertEqual(len(arrivals), 1)

        # Interpolate both paths to 100 samples and make sure they are
        # approximately equal.
        sample_points = np.linspace(0, 35, 100)

        interpolated_expected = np.interp(
            sample_points,
            expected[:, 0],
            expected[:, 1])

        interpolated_actual = np.interp(
            sample_points,
            np.round(np.degrees(arrivals[0].path['dist']), 2),
            np.round(6371 - arrivals[0].path['depth'], 2))

        self.assertTrue(np.allclose(interpolated_actual,
                                    interpolated_expected, rtol=1E-4, atol=0))

    def _read_ak135_test_files(self, filename):
        """
        Helper function parsing the AK135 test data from the original TauP
        test suite.
        """
        filename = os.path.join(DATA, filename)
        with open(filename, "rb") as fh:
            line = fh.readline()
            line = line.strip().split()
            depths = list(map(float, line))

            data = np.genfromtxt(fh)

        dist = data[:, 0]
        time = data[:, 1:-len(depths)]
        ray_param = data[:, -len(depths):]
        time = time[:, ::2] * 60.0 + time[:, 1::2]

        values = np.empty(np.size(ray_param),
                          dtype=[(native_str('depth'), np.float_),
                                 (native_str('dist'), np.float_),
                                 (native_str('ray_param'), np.float_),
                                 (native_str('time'), np.float_)])

        values['depth'] = np.tile(depths, len(dist))
        values['dist'] = np.repeat(dist, len(depths))
        values['ray_param'] = ray_param.flat
        values['time'] = time.flat

        return values

    def _compare_against_ak135_tables_kennet(self, filename, phases):
        """
        Helper function to compare against the AK135 traveltime tables of
        Kennet. This test is also done in the Java TauP version.
        """
        values = self._read_ak135_test_files(filename)
        m = TauPyModel(model="ak135")
        start = np.random.randint(SPEEDUP_FACTOR)
        for value in values[start::SPEEDUP_FACTOR]:
            # Parameters are not strictly defined for a non-existent travel
            # time.
            if value["time"] == 0.0:
                continue
            arrivals = m.get_ray_paths(
                source_depth_in_km=value["depth"],
                distance_in_degree=value["dist"],
                phase_list=phases)
            arrivals = sorted(arrivals, key=lambda x: x.time)
            arr = arrivals[0]
            # These are the same tolerances as in the Java tests suite.
            self.assertTrue(abs(arr.time - value["time"]) < 0.07)
            self.assertTrue(abs(arr.ray_param_sec_degree -
                                value["ray_param"]) < 0.11)

    def test_kennet_ak135_ttime_tables_p_deep(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_P_deep.txt", phases=["p", "Pdiff", "P"])

    def test_kennet_ak135_ttime_tables_p_shallow(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_P_shallow.txt", phases=["p", "Pdiff", "P"])

    def test_kennet_ak135_ttime_tables_pcp(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_PcP.txt", phases=["PcP"])

    def test_kennet_ak135_ttime_tables_pkikp(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_PKIKP.txt", phases=["PKIKP"])

    def test_kennet_ak135_ttime_tables_s_deep(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_S_deep.txt", phases=["s", "S", "Sdiff"])

    def test_kennet_ak135_ttime_tables_s_shallow(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_S_shallow.txt", phases=["s", "S", "Sdiff"])

    def test_kennet_ak135_ttime_tables_scp(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_ScP.txt", phases=["ScP"])

    def test_kennet_ak135_ttime_tables_scs(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_ScS.txt", phases=["ScS"])

    def test_java_pnps(self):
        """
        Test for Pn Ps waves bug, 53797854298c6ee9dcbf3398bbec3bdd12def964

        Reference output generated by::

          $ ./taup_time -mod iasp91 -ph P,S,PP,PPP,PPPP,S,SS,SS,SSSS,Sn,Pn \
              -h 15 -deg 10
        """
        m = TauPyModel(model="iasp91")
        ph = "P,S,PP,PPP,PPPP,S,SS,SS,SSSS,Sn,Pn".split(",")
        arrivals = m.get_travel_times(source_depth_in_km=15.0,
                                      distance_in_degree=10.0,
                                      phase_list=ph)
        self._compare_arrivals_with_file(arrivals,
                                         "java_tauptime_pnsn")

    def test_surface_wave_ttimes(self):
        """
        Tests the calculation of surface ttimes.

        Tested against a reference output from the Java TauP version.
        """
        for model, table in [("iasp91", "iasp91_surface_waves_table.txt"),
                             ("ak135", "ak135_surface_waves_table.txt")]:
            m = TauPyModel(model=model)
            filename = os.path.join(DATA, table)

            with open(filename, "rt") as fh:
                for line in fh:
                    _, distance, depth, phase, time, ray_param, _, _ \
                        = line.split()
                    distance, depth, time, ray_param = \
                        map(float, [distance, depth, time, ray_param])

                    arrivals = m.get_travel_times(
                        source_depth_in_km=depth, distance_in_degree=distance,
                        phase_list=[phase])

                    self.assertTrue(len(arrivals) > 0)

                    # Potentially multiple arrivals. Get the one closest in
                    # time and closest in ray parameter.
                    arrivals = sorted(
                        arrivals,
                        key=lambda x: (abs(x.time - time),
                                       abs(x.ray_param_sec_degree -
                                           ray_param)))
                    arrival = arrivals[0]
                    self.assertEqual(round(arrival.time, 2), round(time, 2))
                    self.assertEqual(round(arrival.ray_param_sec_degree, 2),
                                     round(ray_param, 2))

    def test_underside_reflections(self):
        """
        Tests the calculation of a couple of underside reflection phases.
        """
        m = TauPyModel(model="iasp91")
        # If an interface that is not in the model is used for a phase name,
        # it should snap to the next closest interface. This is reflected in
        # the purist name of the arrivals.
        arrivals = m.get_travel_times(
            source_depth_in_km=10.0, distance_in_degree=90.0,
            phase_list=["P", "PP", "P^410P", "P^660P", "P^300P", "P^400P",
                        "P^500P", "P^600P"])

        self._compare_arrivals_with_file(arrivals, "underside_reflections.txt")

    def test_buried_receiver(self):
        """
        Simple test for a buried receiver.
        """
        m = TauPyModel(model="iasp91")
        arrivals = m.get_travel_times(
            source_depth_in_km=10.0, distance_in_degree=90.0,
            receiver_depth_in_km=50,
            phase_list=["P", "PP", "S"])

        self._compare_arrivals_with_file(arrivals, "buried_receivers.txt")

    def test_different_models(self):
        """
        Open all included models and make sure that they can produce
        reasonable travel times.
        """
        models = ["1066a", "1066b", "ak135", "herrin", "iasp91", "prem",
                  "sp6", "jb", "pwdk", "ak135f_no_mud"]
        for model in models:
            m = TauPyModel(model=model)

            # Get a p phase.
            arrivals = m.get_travel_times(
                source_depth_in_km=10.0, distance_in_degree=50.0,
                phase_list=["P"])
            # AK135 travel time.
            expected = 534.4
            self.assertTrue(abs(arrivals[0].time - expected) < 5)

            # Get an s phase.
            arrivals = m.get_travel_times(
                source_depth_in_km=10.0, distance_in_degree=50.0,
                phase_list=["S"])
            # AK135 travel time.
            expected = 965.1
            # Some models do produce s-waves but they are very far from the
            # AK135 value.
            self.assertTrue(abs(arrivals[0].time - expected) < 50)

    def test_paths_for_crustal_phases(self):
        """
        Tests that Pn and PmP are correctly modelled and not mixed up.

        See #1392.
        """
        model = TauPyModel(model='iasp91')
        paths = model.get_ray_paths(source_depth_in_km=0,
                                    distance_in_degree=1,
                                    phase_list=['Pn', 'PmP'])
        self.assertEqual(len(paths), 2)

        self.assertEqual(paths[0].name, "PmP")
        self.assertAlmostEqual(paths[0].time, 21.273, 3)
        self.assertEqual(paths[1].name, "Pn")
        self.assertAlmostEqual(paths[1].time, 21.273, 3)

        self.assertAlmostEqual(paths[0].time, 21.273, 3)

        # Values of visually checked paths to guard against regressions.
        pmp_path = [
            [0.0, 0.0],
            [8.732364066174294e-07, 0.005402127286288305],
            [0.00020293803558129412, 1.2550644943312363],
            [0.000405124234951687, 2.5047268613761844],
            [0.0008098613580518535, 5.004051595465171],
            [0.0016207981804224497, 10.002701063643144],
            [0.00243369214394241, 15.001350531821117],
            [0.0032485517460744207, 20.0],
            [0.0034500021984838003, 20.9375],
            [0.0036515675727796315, 21.875],
            [0.004055043605178164, 23.75],
            [0.004863380445624696, 27.5],
            [0.006485622554890742, 35.0],
            [0.008803860898528547, 35.018603858353345],
            [0.011122099242166353, 35.0],
            [0.012744341351432398, 27.5],
            [0.01355267819187893, 23.75],
            [0.013956154224277463, 21.875],
            [0.014157719598573294, 20.9375],
            [0.014359170050982674, 20.0],
            [0.015174029653114684, 15.001350531821117],
            [0.015986923616634643, 10.002701063643144],
            [0.01679786043900524, 5.004051595465171],
            [0.017202597562105407, 2.5047268613761844],
            [0.0174047837614758, 1.2550644943312363],
            [0.017606848560650475, 0.005402127286288305],
            [0.017607721797057094, 0.0]]
        pn_path = [
            [0.0, 0.0],
            [8.732421799574388e-07, 0.005402127286288305],
            [0.00020293937754080365, 1.2550644943312363],
            [0.0004051269144571584, 2.5047268613761844],
            [0.0008098667167377564, 5.004051595465171],
            [0.0016208089138889542, 10.002701063643144],
            [0.002433708274208186, 15.001350531821117],
            [0.003248573295310095, 20.0],
            [0.0034500255976490177, 20.9375],
            [0.0036515928239481774, 21.875],
            [0.004055072566590625, 23.75],
            [0.004863416852584004, 27.5],
            [0.0064856739540738425, 35.0],
            [0.0064856739540738425, 35.0],
            [0.010967618565869454, 35.0],
            [0.010967618565869454, 35.0],
            [0.012589875667359293, 27.5],
            [0.013398219953352672, 23.75],
            [0.01380169969599512, 21.875],
            [0.01400326692229428, 20.9375],
            [0.014204719224633202, 20.0],
            [0.015019584245735112, 15.001350531821117],
            [0.015832483606054343, 10.002701063643144],
            [0.01664342580320554, 5.004051595465171],
            [0.017048165605486137, 2.5047268613761844],
            [0.017250353142402492, 1.2550644943312363],
            [0.017452419277763337, 0.005402127286288305],
            [0.017453292519943295, 0.0]]

        np.testing.assert_allclose([_i[0] for _i in pmp_path],
                                   paths[0].path["dist"])
        np.testing.assert_allclose([_i[1] for _i in pmp_path],
                                   paths[0].path["depth"])
        np.testing.assert_allclose([_i[0] for _i in pn_path],
                                   paths[1].path["dist"])
        np.testing.assert_allclose([_i[1] for _i in pn_path],
                                   paths[1].path["depth"])

    def test_arrivals_class(self):
        """
        Tests list operations on the Arrivals class.

        See #1518.
        """
        model = TauPyModel(model='iasp91')
        arrivals = model.get_ray_paths(source_depth_in_km=0,
                                       distance_in_degree=1,
                                       phase_list=['Pn', 'PmP'])
        self.assertEqual(len(arrivals), 2)
        # test copy
        self.assertTrue(isinstance(arrivals.copy(), Arrivals))
        # test sum
        self.assertTrue(isinstance(arrivals + arrivals, Arrivals))
        self.assertTrue(isinstance(arrivals + arrivals[0], Arrivals))
        # test multiplying
        self.assertTrue(isinstance(arrivals * 2, Arrivals))
        arrivals *= 3
        self.assertEqual(len(arrivals), 6)
        self.assertTrue(isinstance(arrivals, Arrivals))
        # test slicing
        self.assertTrue(isinstance(arrivals[2:5], Arrivals))
        # test appending
        arrivals.append(arrivals[0])
        self.assertEqual(len(arrivals), 7)
        self.assertTrue(isinstance(arrivals, Arrivals))
        # test assignment
        arrivals[0] = arrivals[-1]
        self.assertTrue(isinstance(arrivals, Arrivals))
        arrivals[2:5] = arrivals[1:4]
        self.assertTrue(isinstance(arrivals, Arrivals))
        # test assignment with wrong type
        with self.assertRaises(TypeError):
            arrivals[0] = 10.
        with self.assertRaises(TypeError):
            arrivals[2:5] = [0, 1, 2]
        with self.assertRaises(TypeError):
            arrivals.append(arrivals)
        # test add and mul with wrong type
        with self.assertRaises(TypeError):
            arrivals + [2, ]
        with self.assertRaises(TypeError):
            arrivals += [2, ]
        with self.assertRaises(TypeError):
            arrivals * [2, ]
        with self.assertRaises(TypeError):
            arrivals *= [2, ]

    def test_small_regional_model(self):
        """
        Tests a small regional model as this used to not work.
        """
        with TemporaryWorkingDirectory():
            folder = os.path.abspath(os.curdir)
            model_name = "regional_model"
            build_taup_model(
                filename=os.path.join(DATA, os.path.pardir,
                                      model_name + ".tvel"),
                output_folder=folder, verbose=False)
            m = TauPyModel(os.path.join(folder, model_name + ".npz"))
        arr = m.get_ray_paths(source_depth_in_km=18.0, distance_in_degree=1.0)
        self.assertEqual(len(arr), 9)
        for a, d in zip(arr, [("p", 18.143), ("Pn", 19.202), ("PcP", 19.884),
                              ("sP", 22.054), ("ScP", 23.029), ("PcS", 26.410),
                              ("s", 31.509), ("Sn", 33.395), ("ScS", 34.533)]):
            self.assertEqual(a.name, d[0])
            self.assertAlmostEqual(a.time, d[1], 3)


def suite():
    return unittest.makeSuite(TauPyModelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
