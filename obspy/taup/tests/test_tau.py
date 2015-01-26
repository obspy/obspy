#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the high level obspy.taup.tau interface.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import inspect
import os
import unittest

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
        output = []
        with open(os.path.join(DATA, filename), "rt") as fh:
            while True:
                line = fh.readline().strip()
                if line.startswith("-----"):
                    break
            while True:
                line = fh.readline().strip()
                if not line:
                    break
                line = line.replace("=", "").strip()
                if not line:
                    continue
                line = line.split()
                output.append({
                    "distance": float(line[0]),
                    "depth": float(line[1]),
                    "name": line[2],
                    "time": float(line[3]),
                    "ray_param_sec_degree": float(line[4]),
                    "takeoff_angle": float(line[5]),
                    "incident_angle": float(line[6]),
                    "purist_distance": float(line[7]),
                    "purist_name": line[8]})
        return output

    def _compare_arrivals_with_file(self, arrivals, filename):
        """
        Helper method comparing arrivals against the phases stored in a file.
        """
        arrivals = sorted(arrivals, key=lambda x: x.time)
        _expected_arrivals_unsorted = self._read_taup_output(filename)
        expected_arrivals = sorted(_expected_arrivals_unsorted,
                                   key=lambda x: x["time"])
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
        self.assertEqual(arr.get_modulo_dist_deg(), expected_arr["distance"])
        self.assertEqual(arr.source_depth, expected_arr["depth"])
        self.assertEqual(arr.name, expected_arr["name"])
        self.assertEqual(round(arr.time, 2), round(expected_arr["time"], 2))
        self.assertEqual(round(arr.ray_param_sec_degree, 3),
                         round(expected_arr["ray_param_sec_degree"], 3))
        self.assertEqual(round(arr.takeoff_angle, 2),
                         round(expected_arr["takeoff_angle"], 2))
        self.assertEqual(round(arr.incident_angle, 2),
                         round(expected_arr["incident_angle"], 2))
        self.assertEqual(round(arr.purist_distance, 2),
                         round(expected_arr["purist_distance"], 2))
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
        self.assertEqual(round(p_arrival.time, 2), 412.43)
        self.assertEqual(round(p_arrival.ray_param_sec_degree, 3), 8.612)
        self.assertEqual(round(p_arrival.takeoff_angle, 2), 26.74)
        self.assertEqual(round(p_arrival.incident_angle, 2), 26.69)
        self.assertEqual(round(p_arrival.purist_distance, 2), 35.00)
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

        expected = []
        with open(filename, "rt") as fh:
            fh.readline()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                expected.append(list(map(float, line.split())))

        actual = []
        for pierce in p_arr.pierce:
            actual.append([round(pierce.get_dist_deg(), 2),
                           round(pierce.depth, 1),
                           round(pierce.time, 1)])

        self.assertEqual(expected, actual)

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
                    current_phase = line.replace(">", "").strip().split()[0]
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
                    round(p.get_dist_deg(), 2),
                    round(p.depth, 1),
                    round(p.time, 1)))

        self.assertEqual(sorted(actual.keys()), sorted(expected.keys()))

        for key in actual.keys():
            actual_values = sorted(actual[key])
            expected_values = sorted(expected[key])
            self.assertEqual(actual_values, expected_values)


def suite():
    return unittest.makeSuite(TauPyModelTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
