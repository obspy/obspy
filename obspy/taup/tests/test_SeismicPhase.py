#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests the SeismicPhase class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.taup.tau_model import TauModel
from obspy.taup.seismic_phase import SeismicPhase


class TauPySeismicPhaseTestCase(unittest.TestCase):
    """
    Test suite for the SeismicPhase class.
    """
    def setUp(self):
        self.depth = 119
        self.tMod = TauModel.from_file('iasp91').depth_correct(self.depth)

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
        phase = SeismicPhase(phase_name, self.tMod)
        for i, ray_param in enumerate(phase.ray_param):
            maxRPArrival = phase.shoot_ray(-1, ray_param)
            self.assertAlmostEqual(phase.dist[i], maxRPArrival.purist_dist,
                                   delta=0.0001)
            self.assertAlmostEqual(phase.time[i], maxRPArrival.time,
                                   delta=0.0001)

    def test_shoot_middle_ray_param(self):
        phase = SeismicPhase('P', self.tMod)
        for i in range(phase.ray_param.shape[0] - 1):
            rp = (phase.ray_param[i] + phase.ray_param[i + 1]) / 2
            timeTol = abs(phase.time[i] - phase.time[i + 1])
            maxRPArrival = phase.shoot_ray(-1, rp)
            self.assertAlmostEqual(phase.dist[i], maxRPArrival.purist_dist,
                                   delta=0.1)
            self.assertAlmostEqual(phase.time[i], maxRPArrival.time,
                                   delta=timeTol)
            self.assertAlmostEqual(phase.dist[i + 1], maxRPArrival.purist_dist,
                                   delta=0.1)
            self.assertAlmostEqual(phase.time[i + 1], maxRPArrival.time,
                                   delta=timeTol)


def suite():
    return unittest.makeSuite(TauPySeismicPhaseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
