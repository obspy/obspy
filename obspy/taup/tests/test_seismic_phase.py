#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
import inspect
import os

import pytest

from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.taup import TauPyModel
from obspy.taup.tau_model import TauModel
from obspy.taup.seismic_phase import SeismicPhase
from obspy.taup.taup_create import build_taup_model
from obspy.taup.helper_classes import TauModelError

# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


class TestTauPySeismicPhase:
    """
    Test suite for the SeismicPhase class.
    """
    depth = 119

    @pytest.fixture(scope='class')
    def tau_model(self):
        """Return the tau model for testing."""
        return TauModel.from_file('iasp91').depth_correct(self.depth)

    def test_shoot_existing_ray_param(self, tau_model):
        self.do_shoot_existing_ray_param_for_phase('P', tau_model)
        self.do_shoot_existing_ray_param_for_phase('S', tau_model)
        self.do_shoot_existing_ray_param_for_phase('p', tau_model)
        self.do_shoot_existing_ray_param_for_phase('p', tau_model)
        self.do_shoot_existing_ray_param_for_phase('s', tau_model)
        self.do_shoot_existing_ray_param_for_phase('PP', tau_model)
        self.do_shoot_existing_ray_param_for_phase('SS', tau_model)
        self.do_shoot_existing_ray_param_for_phase('PcP', tau_model)
        self.do_shoot_existing_ray_param_for_phase('ScS', tau_model)
        self.do_shoot_existing_ray_param_for_phase('PKP', tau_model)
        self.do_shoot_existing_ray_param_for_phase('SKS', tau_model)
        self.do_shoot_existing_ray_param_for_phase('PKIKP', tau_model)
        self.do_shoot_existing_ray_param_for_phase('SKIKS', tau_model)

    def do_shoot_existing_ray_param_for_phase(self, phase_name, tau_model):
        phase = SeismicPhase(phase_name, tau_model)
        for i, ray_param in enumerate(phase.ray_param):
            max_rp_arrival = phase.shoot_ray(-1, ray_param)
            assert abs(phase.dist[i]-max_rp_arrival.purist_dist) < 0.0001
            assert abs(phase.time[i]-max_rp_arrival.time) < 0.0001

    def test_shoot_middle_ray_param(self, tau_model):
        phase = SeismicPhase('P', tau_model)
        for i in range(phase.ray_param.shape[0] - 1):
            rp = (phase.ray_param[i] + phase.ray_param[i + 1]) / 2
            time_tol = abs(phase.time[i] - phase.time[i + 1])
            max_rp_arrival = phase.shoot_ray(-1, rp)
            assert abs(phase.dist[i]-max_rp_arrival.purist_dist) < 0.1
            assert abs(phase.time[i]-max_rp_arrival.time) < time_tol
            assert abs(phase.dist[i + 1]-max_rp_arrival.purist_dist) < 0.1
            assert abs(phase.time[i + 1]-max_rp_arrival.time) < time_tol

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
        assert len(arr) > 10

    def test_diffracted_phases(self):
        """
        Test of exotic diffracted phases.
        """
        model = TauPyModel('iasp91')
        phs = ["SedPdiffKP", "PdiffPdiff", "PedPdiffKKP",
               "PdiffKKPdiff", "PPdiff", "SKdiffP"]
        dists = [155.0, 210.0, 310.0, 300.0, 220.0, 200.0]
        times = [1464.97, 1697.88, 2052.42, 2008.03, 1742.27, 1431.53]

        for ph, dist, time in zip(phs, dists, times):
            phase = SeismicPhase(ph, model.model)
            arrivals = phase.calc_time(dist)
            arrival = arrivals[0]
            phase.calc_pierce_from_arrival(arrival)
            phase.calc_path_from_arrival(arrival)

            tol = 1e-2
            assert abs(arrival.time - time) < tol
            assert abs(arrival.pierce["time"][-1] - time) < tol
            assert abs(arrival.path["time"][-1] - time) < tol

    def test_phase_names(self, tau_model):
        """
        Simple check to see if illegal phase names are caught.
        """
        legal_phase_names = ["ScS", "ScSScS"]
        illegal_phase_names = ["ScScS", "PKIKPKIKP", "PKIKIKP"]

        for name in legal_phase_names:
            SeismicPhase(name, tau_model)

        for name in illegal_phase_names:
            with pytest.raises(TauModelError):
                SeismicPhase(name, tau_model)
