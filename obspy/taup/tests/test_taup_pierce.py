#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file tests the TauP_Pierce utility against the original TauPy
output for the TauP_Pierce utility using the high-level interface of TauPy.
Unlike the test for TauP_Time, it doesn't test a range of input depths and
degrees, that would be difficult to implement. It can probably be assumed that
if TauP_Time works well for a range of inputs, TauP_Pierce would too.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import unittest

from obspy.taup import TauPyModel

# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data", "TauP_test_data")


def parse_taup_pierce_output(filename):
    with open(filename, "r") as fh:
        arrivals = []
        pp_dist = []
        pp_depth = []
        pp_time = []
        new_phase = False
        phase_name = None
        i = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                l = [_i.strip() for _i in line.split()]
                if phase_name is not None:
                    new_phase = True
                phase_name = str(l[1])
                time = float(l[3])
                arrivals.append({
                    "phase_name": phase_name,
                    "time": time})
            else:
                l = [_i.strip() for _i in line.split()]
                pp_dist.append(l[0])
                pp_depth.append(l[1])
                pp_time.append(l[2])

            if new_phase is True:
                arrivals[i].update({
                    "pp_distance": pp_dist,
                    "pp_depth": pp_depth,
                    "pp_time": pp_time,
                })
                new_phase = False
                pp_dist = []
                pp_depth = []
                pp_time = []
                i += 1
        arrivals[i].update(
            dict(pp_distance=pp_dist, pp_depth=pp_depth, pp_time=pp_time))

    # A very specific hack for testing the ttall file. For SOME weird
    # reason the loop above iterates over all lines in the file TWICE!
    if len(arrivals) == 66:
        arrivals = arrivals[:33]
    return arrivals


def compare_arrivals_with_taup_pierce_output(arrivals, filename):
    """
    """
    filename = os.path.join(DATA, filename)
    expected_arrivals = parse_taup_pierce_output(filename)

    arrivals = [
        {
            "phase_name": _i.name,
            "time": _i.time,
            "pp_distance": [_p.get_dist_deg() for _p in _i.pierce],
            "pp_depth": [_p.depth for _p in _i.pierce],
            "pp_time": [_p.time for _p in _i.pierce]
        } for _i in arrivals]

    # Sort both by phase_name and time.
    expected_arrivals = sorted(expected_arrivals,
                               key=lambda x: x["phase_name"])
    expected_arrivals = sorted(expected_arrivals,
                               key=lambda x: x["time"])
    arrivals = sorted(arrivals, key=lambda x: x["phase_name"])
    arrivals = sorted(arrivals, key=lambda x: x["time"])

    assert len(expected_arrivals) == len(arrivals)

    for e_arr, arr in zip(expected_arrivals, arrivals):
        assert sorted(e_arr.keys()) == sorted(arr.keys())
        for key, values in e_arr.items():
            if isinstance(values, list):
                for j, value in enumerate(values):
                    if isinstance(value, float):
                        # Estimate the precision in the taup output.
                        v = str(value)
                        prec = len(v) - v.find(".") - 1
                        assert value == round(arr[key][j], prec)
            elif isinstance(values, float):
                        # Estimate the precision in the taup output.
                        v = str(values)
                        prec = len(v) - v.find(".") - 1
                        try:
                            assert values == round(arr[key], prec)
                        except AssertionError:
                            print(values, arr[key], key)
                            raise AssertionError
            else:
                assert values == arr[key]


class TestTauPPierce(unittest.TestCase):
    def test_P_phase_iasp91_35_deg_distance(self):
        """
        Tests tauppierce at 35 degree distance for phase P with iasp91.
        """
        model = TauPyModel("iasp91")
        tts = model.get_pierce_points(source_depth_in_km=10.0,
                                      distance_in_degree=35.0,
                                      phase_list=["P"])
        compare_arrivals_with_taup_pierce_output(
            tts, "taup_pierce_-h_10_-ph_P_-deg_35")

    def test_ttall_phase_iasp91_35_deg_distance(self):
        """
        Tests tauppierce at 35 degree distance for phases ttall with iasp91.
        """
        model = TauPyModel("iasp91")
        tts = model.get_pierce_points(source_depth_in_km=10.0,
                                      distance_in_degree=35.0,
                                      phase_list=["ttall"])
        compare_arrivals_with_taup_pierce_output(
            tts, "java_taup_pierce_h10_deg35_ttall")


if __name__ == '__main__':
    unittest.main(buffer=True)
