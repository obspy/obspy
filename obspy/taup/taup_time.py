#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Travel time calculations.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .helper_classes import TauModelError
from .seismic_phase import SeismicPhase
from .utils import parse_phase_list


class TauP_Time(object):
    """
    Calculate travel times for different branches using linear interpolation
    between known slowness samples.
    """
    def __init__(self, model, phase_list, depth, degrees):
        self.depth = depth
        self.degrees = degrees
        self.arrivals = []
        self.phases = []
        # Names of phases to be used, e.g. PKIKP
        self.phase_names = parse_phase_list(phase_list)

        # A standard and a depth corrected model. Both are needed.
        self.model = model
        self.depth_corrected_model = self.model

    def run(self):
        """
        Do all the calculations and print the output if told to. The resulting
        arrival times will be in self.arrivals.
        """
        self.depth_correct(self.depth)
        self.calculate(self.degrees)

    def depth_correct(self, depth):
        """
        Corrects the TauModel for the given source depth (if not already
        corrected).
        """
        if self.depth_corrected_model is None or \
                self.depth_corrected_model.source_depth != depth:
            self.depth_corrected_model = self.model.depth_correct(depth)
            self.arrivals = []
            self.recalc_phases()
        self.source_depth = depth

    def recalc_phases(self):
        """
        Recalculates the given phases using a possibly new or changed tau
        model.
        """
        new_phases = []
        for temp_phase_name in self.phase_names:
            for phase_num, seismic_phase in enumerate(self.phases):
                if seismic_phase.name == temp_phase_name:
                    self.phases.pop(phase_num)
                    if (seismic_phase.source_depth == self.depth and
                            seismic_phase.tMod == self.depth_corrected_model):
                        # OK so copy to new_phases:
                        new_phases.append(seismic_phase)
                        break
            # Executed, if break is NOT called.
            else:
                # Didn't find it precomputed, so recalculate:
                try:
                    seismic_phase = SeismicPhase(temp_phase_name,
                                                 self.depth_corrected_model)
                    new_phases.append(seismic_phase)
                except TauModelError:
                    print("Error with this phase, skipping it: " +
                          str(temp_phase_name))
            self.phases = new_phases

    def calculate(self, degrees):
        """
        Calculate the arrival times.
        """
        self.depth_correct(self.source_depth)
        # Called before, but depth_correct might have changed the phases.
        self.recalc_phases()
        self.calc_time(degrees)

    def calc_time(self, degrees):
        """
        Calls the calc_time method of SeismicPhase to calculate arrival
        times for every phase, each sorted by time.
        """
        self.degrees = degrees
        self.arrivals = []
        for phase in self.phases:
            self.arrivals += phase.calc_time(degrees)
        # Sort them.
        self.arrivals = sorted(self.arrivals,
                               key=lambda arrivals: arrivals.time)
