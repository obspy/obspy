#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Manual test to just check if the tau interface runs without upsets.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *

import inspect
import os

from taupy import tau

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
                "rayParam_sec_degree": float(line[4]),
                "takeoffAngle": float(line[5]),
                "incidentAngle": float(line[6]),
                "purist_distance": float(line[7]),
                "puristName": line[8]})
    return output


def _compare_arrivals_with_file(arrivals, filename):
    arrivals = sorted(arrivals, key=lambda x: x.time)
    expected_arrivals = sorted(_read_taup_output(filename),
                               key=lambda x: x["time"])

    for arr, expected_arr in zip(arrivals, expected_arrivals):
        assert arr.getDistDeg() == expected_arr["distance"]
        assert arr.sourceDepth == expected_arr["depth"]
        assert arr.name == expected_arr["name"]
        assert round(arr.time, 2) == round(expected_arr["time"], 2)
        assert round(arr.rayParam_sec_degree, 3) == \
               round(expected_arr["rayParam_sec_degree"], 3)
        assert round(arr.takeoffAngle, 2) == \
               round(expected_arr["takeoffAngle"], 2)
        assert round(arr.incidentAngle, 2) == \
               round(expected_arr["incidentAngle"], 2)
        assert round(arr.purist_distance, 2) == \
               round(expected_arr["purist_distance"], 2)
        assert arr.puristName == expected_arr["puristName"]


def test_p_iasp91_manual():
    m = tau.TauPyModel(model="iasp91")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0, phase_list=["P"])
    assert len(arrivals) == 1
    p_arrival = arrivals[0]

    assert p_arrival.name == "P"
    assert round(p_arrival.time, 2) == 412.43
    assert round(p_arrival.rayParam_sec_degree, 3) == 8.612
    assert round(p_arrival.takeoffAngle, 2) == 26.74
    assert round(p_arrival.incidentAngle, 2) == 26.69
    assert round(p_arrival.purist_distance, 2) == 35.00
    assert p_arrival.puristName == "P"


def test_p_iasp91():
    m = tau.TauPyModel(model="iasp91")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0, phase_list=["P"])
    _compare_arrivals_with_file(arrivals, "taup_time_-h_10_-ph_P_-deg_35")


def test_p_ak135():
    m = tau.TauPyModel(model="ak135")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0, phase_list=["P"])
    _compare_arrivals_with_file(
        arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")

def test_iasp91():
    m = tau.TauPyModel(model="iasp91")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0,
                                  phase_list=["ttall"])
    _compare_arrivals_with_file(arrivals, "taup_time_-h_10_-ph_ttall_-deg_35")


def test_ak135():
    m = tau.TauPyModel(model="ak135")
    arrivals = m.get_travel_times(source_depth_in_km=10.0,
                                  distance_in_degree=35.0,
                                  phase_list=["ttall"])
    _compare_arrivals_with_file(
        arrivals, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")
