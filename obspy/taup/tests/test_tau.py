#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the high level obspy.taup.tau interface.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os

from obspy.taup import TauPyModel

# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data", "TauP_test_data")


def _read_taup_output(filename):
    output = []
    with open(os.path.join(DATA, filename), "rt") as fh:
        while True:
            line = fh.readline().strip()
            if line.startswith("-----"):
                break
        for line in fh:
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


def _compare_arrivals_with_file(arrivals, filename):
    arrivals = sorted(arrivals, key=lambda x: x.time)
    expected_arrivals = sorted(_read_taup_output(filename),
                               key=lambda x: x["time"])

    for arr, expected_arr in zip(arrivals, expected_arrivals):
        assert arr.get_modulo_dist_deg() == expected_arr["distance"]
        assert arr.source_depth == expected_arr["depth"]
        assert arr.name == expected_arr["name"]
        assert round(arr.time, 2) == round(expected_arr["time"], 2)
        assert round(arr.ray_param_sec_degree, 3) == \
            round(expected_arr["ray_param_sec_degree"], 3)
        assert round(arr.takeoff_angle, 2) == \
            round(expected_arr["takeoff_angle"], 2)
        assert round(arr.incident_angle, 2) == \
            round(expected_arr["incident_angle"], 2)
        assert round(arr.purist_distance, 2) == \
            round(expected_arr["purist_distance"], 2)
        assert arr.purist_name == expected_arr["purist_name"]


def test_p_iasp91_manual():
    m = TauPyModel(model="iasp91")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0, phase_list=["P"])
    assert len(arrivals) == 1
    p_arrival = arrivals[0]

    assert p_arrival.name == "P"
    assert round(p_arrival.time, 2) == 412.43
    assert round(p_arrival.ray_param_sec_degree, 3) == 8.612
    assert round(p_arrival.takeoff_angle, 2) == 26.74
    assert round(p_arrival.incident_angle, 2) == 26.69
    assert round(p_arrival.purist_distance, 2) == 35.00
    assert p_arrival.purist_name == "P"


def test_p_iasp91():
    m = TauPyModel(model="iasp91")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0, phase_list=["P"])
    _compare_arrivals_with_file(arrivals, "taup_time_-h_10_-ph_P_-deg_35")


def test_p_ak135():
    m = TauPyModel(model="ak135")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0, phase_list=["P"])
    _compare_arrivals_with_file(
        arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")


def test_iasp91():
    m = TauPyModel(model="iasp91")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0,
                                  phase_list=["ttall"])
    _compare_arrivals_with_file(arrivals, "taup_time_-h_10_-ph_ttall_-deg_35")


def test_ak135():
    m = TauPyModel(model="ak135")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0,
                                  phase_list=["ttall"])
    _compare_arrivals_with_file(
        arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")


def test_pierce_p_iasp91():
    m = TauPyModel(model="iasp91")
    arrivals = m.get_pierce_points(source_depth_in_km=10.0,
                                   distance_in_degree=35.0, phase_list=["P"])
    assert len(arrivals) == 1
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

    assert expected == actual
