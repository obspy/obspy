#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file tests the TauP_Time utility  against the original TauPy using
both the high-level tau interface of TauPy and the java-like old script-based
interface.
"""
import inspect
import os
import unittest
import sys
import subprocess

from taupy.tau import TauPyModel
from taupy.TauP_Time import TauP_Time

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
    Tests tauptime at 35 degree distance, phases ttall.
    """
    model = TauPyModel("iasp91")
    tts = model.get_travel_time(source_depth_in_km=10.0,
                                distance_in_degree=35.0)
    compare_arrivals_with_taup_time_output(
        tts, "taup_time_-h_10_-ph_ttall_-deg_35")


class TestTauPTime(unittest.TestCase):
    # For some reason this test throws nosetests off if not in the unittest
    # framwork like the test above...?
    def test_all_phases_ak135_35_deg_distance(self):
        """
        Tests tauptime at 35 degree distance for the ak135 model, phases ttall.
        """
        model = TauPyModel("ak135")
        tts = model.get_travel_time(source_depth_in_km=10.0,
                                    distance_in_degree=35.0)
        compare_arrivals_with_taup_time_output(
            tts, "taup_time_-h_10_-ph_ttall_-deg_35_-mod_ak135")

    def test_range(self):
        """
        Check taup_time output for a range of inputs against the Java output.
        """
        if not os.path.isfile("data/java_tauptime_testoutput"):
            subprocess.call("./generate_tauptime_output.sh", shell=True)
        stdout = sys.stdout
        with open('data/taup_time_test_output', 'wt') as sys.stdout:
            for degree in [0, 45, 90, 180, 360, 560]:
                for depth in [0, 100, 1000, 2889]:
                    tauptime = TauP_Time(degrees=degree, depth=depth,
                                         modelName="iasp91",
                                         phaseList=["ttall"])
                    tauptime.run(printOutput=True)
        sys.stdout = stdout
        # Using ttall need to sort; or lines with same arrival times are in
        # different order. With explicit names of all the phases might not be
        # a problem.
        subprocess.check_call("./compare_tauptime_outputs.sh", shell=True)
        # Use this if lines are in same order:
        #subprocess.check_call("diff -wB data/java_tauptime_testoutput "
        #                     "taup_time_test_output", shell=True)
        os.remove("data/taup_time_test_output")

    def test_degree_distance_from_coords(self):
        tt = TauP_Time(depth=143.2, phaseList=["ttall"],
                       coordinate_list=[13, 14, 50, 200])
        tt.run()
        self.assertEqual(tt.degrees, 116.77958601543997)

if __name__ == '__main__':
    unittest.main(buffer=True)