#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This tests TauP_Path.
"""
import sys
import subprocess
import os
from unittest import TestCase
from taupy.TauP_Path import TauP_Path


class TestTauPPath(TestCase):
    def test_script_output_h10_deg35_iasp91(self):
        tp = TauP_Path(degrees=35, depth=10, modelName="iasp91",
                       phaseList=["P"])
        stdout = sys.stdout
        with open('data/tmp_tauppath_test_output', 'wt') as sys.stdout:
            tp.run(printOutput=True)
        sys.stdout = stdout
        subprocess.check_call("diff -wB data/TauP_test_data/taup_path_-o_"
                              "stdout_-h_10_-ph_P_-deg_35 "
                              "data/tmp_tauppath_test_output", shell=True)
        os.remove("data/tmp_tauppath_test_output")

    def test_script_output_h10_deg35_ak135(self):
        tp = TauP_Path(degrees=35, depth=10, modelName="ak135",
                       phaseList=["P"])
        stdout = sys.stdout
        with open('data/tmp_tauppath_test_output', 'wt') as sys.stdout:
            tp.run(printOutput=True)
        sys.stdout = stdout
        subprocess.check_call("diff -wB data/TauP_test_data/taup_path_-o_"
                              "stdout_-h_10_-ph_P_-deg_35_-mod_ak135 "
                              "data/tmp_tauppath_test_output", shell=True)
        os.remove("data/tmp_tauppath_test_output")