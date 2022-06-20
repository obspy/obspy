# -*- coding: utf-8 -*-
"""
Ray path calculations.
"""
from .taup_pierce import TauPPierce
from . import _DEFAULT_VALUES


class TauPPath(TauPPierce):
    """
    Calculate the ray paths for each phase, using TauPPierce and TauPTime.
    """
    def calculate(self, degrees,
                  ray_param_tol=_DEFAULT_VALUES["default_path_ray_param_tol"]):
        """
        Call all the necessary calculations to obtain the ray paths.
        """
        self.depth_correct(self.source_depth, self.receiver_depth)
        self.recalc_phases()
        self.arrivals = []
        self.calculate_path(degrees, ray_param_tol)

    def calculate_path(self, degrees,
                       ray_param_tol=_DEFAULT_VALUES[
                           "default_path_ray_param_tol"]
                       ):
        """
        Calculates the ray paths for phases at the given distance by
        calling the calculate_path method of the SeismicPhase class. The
        results are then in self.arrivals.
        """
        self.degrees = degrees
        for phase in self.phases:
            self.arrivals += phase.calc_path(degrees, ray_param_tol)
