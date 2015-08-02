#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ray path calculations.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .taup_pierce import TauP_Pierce


class TauP_Path(TauP_Pierce):
    """
    Calculate the ray paths for each phase, using TauP_Pierce and TauP_Time.
    """
    def calculate(self, degrees):
        """
        Call all the necessary calculations to obtain the ray paths.
        """
        self.depth_correct(self.source_depth, self.receiver_depth)
        self.recalc_phases()
        self.arrivals = []
        self.calcPath(degrees)

    def calcPath(self, degrees):
        """
        Calculates the ray paths for phases at the given distance by
        calling the calcPath method of the SeismicPhase class. The results
        are then in self.arrivals.
        """
        self.degrees = degrees
        for phase in self.phases:
            self.arrivals += phase.calc_path(degrees)
