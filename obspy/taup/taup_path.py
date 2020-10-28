# -*- coding: utf-8 -*-
"""
Ray path calculations.
"""
from .taup_pierce import TauPPierce


class TauPPath(TauPPierce):
    """
    Calculate the ray paths for each phase, using TauPPierce and TauPTime.
    """
    def calculate(self, degrees):
        """
        Call all the necessary calculations to obtain the ray paths.
        """
        self.depth_correct(self.source_depth, self.receiver_depth)
        self.recalc_phases()
        self.arrivals = []
        self.calculate_path(degrees)

    def calculate_path(self, degrees):
        """
        Calculates the ray paths for phases at the given distance by
        calling the calculate_path method of the SeismicPhase class. The
        results are then in self.arrivals.
        """
        self.degrees = degrees
        for phase in self.phases:
            self.arrivals += phase.calc_path(degrees)
