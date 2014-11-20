#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file tests the high-level interface of TauPy against the original TauPy
output.
"""
import inspect
import os

from taupy.tau import TauPyModel

# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data", "TauP_test_data")


def parse_taup_time_output(filename):
    with open(filename, "rt") as fh:
        data_started = False
        arrivals = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("-------"):
                data_started = True
                continue
            if data_started is False:
                continue
            l = [_i.strip() for _i in line.split() if _i != "="]
            arrivals.append({
                "distance": float(l[0]),
                "depth": float(l[1]),
                "phase_name": str(l[2]),
                "time": float(l[3]),
                "ray_param": float(l[4]),
                "takeoff": float(l[5]),
                "incident": float(l[6]),
                "purist_distance": float(l[7]),
                "purist_name": str(l[8]),
            })
    return arrivals

def compare_arrivals_with_taup_time_output(arrivals, filename):
    """
    """
    filename = os.path.join(DATA, filename)
    expected_arrivals = parse_taup_time_output(filename)

    arrivals = [
        {
            "distance": _i.getModuloDistDeg(),
            "depth": _i.sourceDepth,
            "phase_name": _i.phase.name,
            "time": _i.time,
            "ray_param": _i.rayParam_sec_degree,
            "takeoff": _i.takeoffAngle,
            "incident": _i.incidentAngle,
            "purist_distance": _i.getDistDeg(),
            "purist_name": _i.puristName
        } for _i in arrivals]

    # Sort both by time.
    expected_arrivals = sorted(expected_arrivals, key=lambda x: x["time"])
    arrivals = sorted(arrivals, key=lambda x: x["time"])

    assert len(expected_arrivals) == len(arrivals)

    for e_arr, arr in zip(expected_arrivals, arrivals):
        assert sorted(e_arr.keys()) == sorted(arr.keys())
        for key, value in e_arr.items():
            if isinstance(value, float):
                # Estimate the precision in the taup output.
                v = str(value)
                prec = len(v) - v.find(".") - 1
                assert value == round(arr[key], prec)
            else:
                assert value == arr[key]


def test_all_phases_iasp91_35_deg_distance():
    """
    Tests a run at 35 degree distance.
    """
    model = TauPyModel("iasp91")
    tts = model.get_travel_time(source_depth_in_km=10.0,
                                distance_in_degree=35.0)
    compare_arrivals_with_taup_time_output(
        tts, "taup_time_-h_10_-ph_ttall_-deg_35")
