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

import numpy as np

from obspy.taup import TauPyModel


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data", "TauP_test_data")


class TauPyModelTestCase(unittest.TestCase):
    """
    Test suite for the obspy.taup.TauPy class.
    """
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
        self.assertAlmostEqual(p_arrival.ray_param_sec_degree, 8.612, 3)
        self.assertAlmostEqual(p_arrival.takeoff_angle, 26.74, 2)
        self.assertAlmostEqual(p_arrival.incident_angle, 26.69, 2)
        self.assertAlmostEqual(p_arrival.purist_distance, 35.00, 2)
        self.assertEqual(p_arrival.purist_name, "P")

    def test_p_iasp91(self):
        """
        Test P phase arrival against TauP output in in model IASP91.
        """
        m = TauPyModel(model="iasp91")
        arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                      distance_in_degree=35.0,
                                      phase_list=["P"])
        self._compare_arrivals_with_file(arrivals,
                                         "taup_time_-h_10_-ph_P_-deg_35")

    def test_p_ak135(self):
        """
        Test P phase arrival against TauP output in in model AK135.
        """
        m = TauPyModel(model="ak135")
        arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                      distance_in_degree=35.0,
                                      phase_list=["P"])
        self._compare_arrivals_with_file(
            arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")

    def test_iasp91(self):
        """
        Test travel times for lots of phases against output from TauP in model
        IASP91.
        """
        m = TauPyModel(model="iasp91")
        arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                      distance_in_degree=35.0,
                                      phase_list=["ttall"])
        self._compare_arrivals_with_file(arrivals,
                                         "taup_time_-h_10_-ph_ttall_-deg_35")

    def test_ak135(self):
        """
        Test travel times for lots of phases against output from TauP in model
        AK135.
        """
        m = TauPyModel(model="ak135")
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

    def test_vs_java_iasp91(self):
        """
        Tests the traveltime calculation against the output from TauP in the
        file 'java_tauptime_testoutput'.

        Essentially tests all kinds of depths and epicentral distances in the
        model iasp91.
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
                except:
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
        for value in values:
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

    def test_kennet_ak135_ttime_tables_P_deep(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_P_deep.txt", phases=["p", "Pdiff", "P"])

    def test_kennet_ak135_ttime_tables_P_shallow(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_P_shallow.txt", phases=["p", "Pdiff", "P"])

    def test_kennet_ak135_ttime_tables_PcP(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_PcP.txt", phases=["PcP"])

    def test_kennet_ak135_ttime_tables_PKIKP(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_PKIKP.txt", phases=["PKIKP"])

    def test_kennet_ak135_ttime_tables_S_deep(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_S_deep.txt", phases=["s", "S", "Sdiff"])

    def test_kennet_ak135_ttime_tables_S_shallow(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_S_shallow.txt", phases=["s", "S", "Sdiff"])

    def test_kennet_ak135_ttime_tables_ScP(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_ScP.txt", phases=["ScP"])

    def test_kennet_ak135_ttime_tables_ScS(self):
        self._compare_against_ak135_tables_kennet(
            "ak135_ScS.txt", phases=["ScS"])

    def test_javaPnPs(self):
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


def suite():
    return unittest.makeSuite(TauPyModelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
