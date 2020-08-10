# -*- coding: utf-8 -*-
"""
Pierce point calculations.
"""
from .taup_time import TauPTime


class TauPPierce(TauPTime):
    """
    The methods here allow using TauPTime to calculate the pierce points
    relating to the different arrivals.
    """
    def __init__(self, model, phase_list, depth, degrees, receiver_depth=0.0):
        super(TauPPierce, self).__init__(
            model=model, phase_list=phase_list, depth=depth, degrees=degrees,
            receiver_depth=receiver_depth)
        self.only_turn_points = False
        self.only_rev_points = False
        self.only_under_points = False
        self.only_add_points = False
        self.add_depth = []

    def depth_correct(self, depth, receiver_depth=None):
        """
        Override TauPTime.depth_correct so that the pierce points may be
        added.
        """
        orig_tau_model = self.model
        must_recalc = False
        # First check if depth_corrected_model is correct as it is. Check to
        # make sure source depth is the same, and then check to make sure
        # each add_depth is in the model.
        if self.depth_corrected_model.source_depth == depth:
            if self.add_depth:
                branch_depths = self.depth_corrected_model.getBranchDepths()
                for add_depth in self.add_depth:
                    for branch_depth in branch_depths:
                        if add_depth == branch_depth:
                            # Found it, so break and go to the next add_depth.
                            break
                        # Didn't find the depth as a branch, so must
                        # recalculate.
                        must_recalc = True
                    if must_recalc:
                        break
        else:
            # The depth isn't event the same, so must recalculate
            must_recalc = True
        if not must_recalc:
            # Won't actually do anything much since depth_corrected_model !=
            #  None.
            TauPTime.depth_correct(self, depth, receiver_depth)
        else:
            self.depth_corrected_model = None
            if self.add_depth is not None:
                for add_depth in self.add_depth:
                    self.model = self.model.split_branch(add_depth)
            TauPTime.depth_correct(self, depth, receiver_depth)
            self.model = orig_tau_model

    def calculate(self, degrees):
        """
        Call all the necessary calculations to obtain the pierce points.
        """
        self.depth_correct(self.source_depth, self.receiver_depth)
        self.recalc_phases()
        self.arrivals = []
        self.calculate_pierce(degrees)

    def calculate_pierce(self, degrees):
        """
        Calculates the pierce points for phases at the given distance by
        calling the calculate_pierce method of the SeismicPhase class.
        The results are then in self.arrivals.
        """
        for phase in self.phases:
            self.arrivals += phase.calc_pierce(degrees)
