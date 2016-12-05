#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import unittest

from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.taup import TauPyModel
from obspy.taup.tau_model import TauModel
from obspy.taup.seismic_phase import SeismicPhase
from obspy.taup.taup_create import build_taup_model


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


class TauPySeismicPhaseTestCase(unittest.TestCase):
    """
    Test suite for the SeismicPhase class.
    """
    def setUp(self):
        self.depth = 119
        self.tau_model = TauModel.from_file('iasp91').depth_correct(self.depth)

    def test_shoot_existing_ray_param(self):
        self.do_shoot_existing_ray_param_for_phase('P')
        self.do_shoot_existing_ray_param_for_phase('S')
        self.do_shoot_existing_ray_param_for_phase('p')
        self.do_shoot_existing_ray_param_for_phase('s')
        self.do_shoot_existing_ray_param_for_phase('PP')
        self.do_shoot_existing_ray_param_for_phase('SS')
        self.do_shoot_existing_ray_param_for_phase('PcP')
        self.do_shoot_existing_ray_param_for_phase('ScS')
        self.do_shoot_existing_ray_param_for_phase('PKP')
        self.do_shoot_existing_ray_param_for_phase('SKS')
        self.do_shoot_existing_ray_param_for_phase('PKIKP')
        self.do_shoot_existing_ray_param_for_phase('SKIKS')

    def do_shoot_existing_ray_param_for_phase(self, phase_name):
        phase = SeismicPhase(phase_name, self.tau_model)
        for i, ray_param in enumerate(phase.ray_param):
            max_rp_arrival = phase.shoot_ray(-1, ray_param)
            self.assertAlmostEqual(phase.dist[i], max_rp_arrival.purist_dist,
                                   delta=0.0001)
            self.assertAlmostEqual(phase.time[i], max_rp_arrival.time,
                                   delta=0.0001)

    def test_shoot_middle_ray_param(self):
        phase = SeismicPhase('P', self.tau_model)
        for i in range(phase.ray_param.shape[0] - 1):
            rp = (phase.ray_param[i] + phase.ray_param[i + 1]) / 2
            time_tol = abs(phase.time[i] - phase.time[i + 1])
            max_rp_arrival = phase.shoot_ray(-1, rp)
            self.assertAlmostEqual(phase.dist[i], max_rp_arrival.purist_dist,
                                   delta=0.1)
            self.assertAlmostEqual(phase.time[i], max_rp_arrival.time,
                                   delta=time_tol)
            self.assertAlmostEqual(phase.dist[i + 1],
                                   max_rp_arrival.purist_dist, delta=0.1)
            self.assertAlmostEqual(phase.time[i + 1], max_rp_arrival.time,
                                   delta=time_tol)

    def test_many_identically_named_phases(self):
        """
        Regression test to make sure obspy.taup works with models that
        produce many identically names seismic phases.
        """
        with TemporaryWorkingDirectory():
            folder = os.path.abspath(os.curdir)
            model_name = "smooth_geodynamic_model"
            build_taup_model(
                filename=os.path.join(DATA, model_name + ".tvel"),
                output_folder=folder, verbose=False)
            m = TauPyModel(os.path.join(folder, model_name + ".npz"))
        arr = m.get_ray_paths(172.8000, 46.762440693494824, ["SS"])
        self.assertGreater(len(arr), 10)


def suite():
    return unittest.makeSuite(TauPySeismicPhaseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
