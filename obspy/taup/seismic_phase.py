#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Objects and functions dealing with seismic phases.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import raise_from

from itertools import count
import math
import re

import numpy as np

from obspy.core.util.obspy_types import Enum

from .helper_classes import (Arrival, SlownessModelError, TauModelError,
                             TimeDist)

from .c_wrappers import clibtau


REFINE_DIST_RADIAN_TOL = 0.0049 * math.pi / 180


_ACTIONS = Enum([
    # Used by add_to_branch when the path turns within a segment. We assume
    # that no ray will turn downward so turning implies turning from downward
    # to upward, ie U.
    "turn",
    # Used by add_to_branch when the path reflects off the top of the end of
    # a segment, ie ^.
    "reflect_underside",
    # Used by add_to_branch when the path reflects off the bottom of the end
    # of a segment, ie v.
    "reflect_topside",
    # Used by add_to_branch when the path transmits up through the end of a
    # segment.
    "transup",
    # Used by add_to_branch when the path transmits down through the end of a
    # segment.
    "transdown"
])


class SeismicPhase(object):
    """
    Stores and transforms seismic phase names to and from their
    corresponding sequence of branches. Will maybe contain "expert" mode
    wherein paths may start in the core. Principal use is to calculate leg
    contributions for scattered phases. Nomenclature: "K" - downgoing wave
    from source in core; "k" - upgoing wave from source in core.
    """
    def __init__(self, name, tau_model, receiver_depth=0.0):
        # The phase name, e.g. PKiKP.
        self.name = name
        # The receiver depth within the TauModel that was used to generate this
        # phase. Normally this is 0.0 for a surface station, but can be
        # different for borehole or scattering calculations.
        self.receiver_depth = receiver_depth
        # TauModel to generate phase for.
        self.tau_model = tau_model

        # The source depth within the TauModel that was used to generate
        # this phase.
        self.source_depth = self.tau_model.source_depth

        # List containing strings for each leg.
        self.legs = leg_puller(name)

        # Name with depths corrected to be actual discontinuities in the model.
        self.purist_name = self.create_purist_name(tau_model)

        # Settings for this instance. Should eventually be configurable.
        self._settings = {
            # The maximum degrees that a Pn or Sn can refract along the moho.
            # Note this is not the total distance, only the segment along the
            # moho.
            "max_refraction_in_radians": np.radians(20.0),
            # The maximum degrees that a Pdiff or Sdiff can diffract along the
            # CMB. Note this is not the total distance, only the segment along
            # the CMB.
            "max_diffraction_in_radians": np.radians(60.0),
            # The maximum number of refinements to make to an Arrival.
            "max_recursion": 5
        }

        # Enables phases originating in core.
        self.expert = False
        # Minimum/maximum ray parameters that exist for this phase.
        self.min_ray_param = None
        self.max_ray_param = None
        # Index within TauModel.ray_param that corresponds to max_ray_param.
        # Note that max_ray_param_index < min_ray_param_index as ray parameter
        # decreases with increasing index.
        self.max_ray_param_index = -1
        # Index within TauModel.ray_param that corresponds to min_ray_param.
        # Note that max_ray_param_index < min_ray_param_index as ray parameter
        # decreases with increasing index.
        self.min_ray_param_index = -1
        # Temporary branch numbers determining where to start adding to the
        # branch sequence.
        self.current_branch = None
        # Array of distances corresponding to the ray parameters stored in
        # ray_param.
        self.dist = None
        # Array of times corresponding to the ray parameters stored in
        # ray_param.
        self.time = None
        # Array of possible ray parameters for this phase.
        self.ray_param = None
        # The minimum distance that this phase can be theoretically observed.
        self.min_distance = 0.0
        # The maximum distance that this phase can be theoretically observed.
        self.max_distance = 1e300
        # List (could make array!) of branch numbers for the given phase.
        # Note that this depends upon both the planet model and the source
        # depth.
        self.branch_seq = []
        # True if the current leg of the phase is down going. This allows a
        # check to make sure the path is correct.
        # Used in addToBranch() and parseName().
        self.down_going = []
        # ArrayList of wave types corresponding to each leg of the phase.
        self.wave_type = []

        self.parse_name(tau_model)
        self.sum_branches(tau_model)

    def create_purist_name(self, tau_model):
        current_leg = self.legs[0]
        # Deal with surface wave veocities first, since they are a special
        # case.
        if len(self.legs) == 2 and current_leg.endswith("kmps"):
            purist_name = self.name
            return purist_name
        purist_name = ""
        # Only loop to penultimate element as last leg is always "END".
        for current_leg in self.legs[:-1]:
            # Find out if the next leg represents a phase conversion or
            # reflection depth.
            if current_leg[0] in "v^":
                disconBranch = closest_branch_to_depth(tau_model,
                                                       current_leg[1:])
                legDepth = tau_model.tauBranches[0, disconBranch].topDepth
                purist_name += current_leg[0]
                purist_name += str(int(round(legDepth)))
            else:
                try:
                    float(current_leg)
                except ValueError:
                    # If current_leg is just a string:
                    purist_name += current_leg
                else:
                    # If it is indeed a number:
                    disconBranch = closest_branch_to_depth(tau_model,
                                                           current_leg)
                    legDepth = tau_model.tauBranches[0, disconBranch].topDepth
                    purist_name += str(legDepth)
        return purist_name

    def parse_name(self, tau_model):
        """
        Construct a branch sequence from the given phase name and tau model.
        """
        current_leg = self.legs[0]
        next_leg = current_leg

        # Deal with surface wave velocities first, since they are a special
        # case.
        if len(self.legs) == 2 and current_leg.endswith("kmps"):
            return

        # Make a check for J legs if the model doesn't allow J:
        if "J" in self.name and not tau_model.sMod.allowInnerCoreS:
            raise TauModelError("J phases are not created for this model: {}"
                                .format(self.name))

        # Set currWave to be the wave type for this leg, P or S
        if current_leg in ("p", "K", "k", "I") or current_leg[0] == "P":
            is_p_wave = True
            is_p_wave_previous = is_p_wave
        elif current_leg in ("s", "J") or current_leg[0] == "S":
            is_p_wave = False
            is_p_wave_previous = is_p_wave
        else:
            raise TauModelError('Unknown starting phase: ' + current_leg)

        # First, decide whether the ray is upgoing or downgoing from the
        # source. If it is up going then the first branch number would be
        # model.sourceBranch-1 and downgoing would be model.sourceBranch.
        upgoing_rec_branch = tau_model.find_branch(self.receiver_depth)
        downgoing_rec_branch = upgoing_rec_branch - 1  # One branch shallower.
        if current_leg[0] in "sS":
            # Exclude S sources in fluids.
            sdep = tau_model.source_depth
            if tau_model.cmb_depth < sdep < tau_model.iocb_depth:
                self.max_ray_param, self.min_ray_param = -1, -1
                return

        # Set self.max_ray_param to be a horizontal ray leaving the source and
        # self.min_ray_param to be a vertical (p=0) ray.
        if current_leg[0] in "PS" or (self.expert and current_leg[0] in "KIJ"):
            # Downgoing from source.
            self.current_branch = tau_model.sourceBranch
            # Treat initial downgoing as if it were an underside reflection.
            endAction = _ACTIONS["reflect_underside"]
            try:
                sLayerNum = tau_model.sMod.layer_number_below(
                    tau_model.source_depth, is_p_wave_previous)
                layer = tau_model.sMod.getSlownessLayer(
                    sLayerNum, is_p_wave_previous)
                self.max_ray_param = layer['topP']
            except SlownessModelError as e:
                raise_from(RuntimeError('Please contact the developers. This '
                                        'error should not occur.'), e)
            self.max_ray_param = tau_model.get_tau_branch(
                tau_model.sourceBranch, is_p_wave).max_ray_param
        elif current_leg in ("p", "s") or (
                self.expert and current_leg[0] == "k"):
            # Upgoing from source: treat initial downgoing as if it were a
            # topside reflection.
            endAction = _ACTIONS["reflect_topside"]
            try:
                sLayerNum = tau_model.sMod.layer_number_above(
                    tau_model.source_depth, is_p_wave_previous)
                layer = tau_model.sMod.getSlownessLayer(
                    sLayerNum, is_p_wave_previous)
                self.max_ray_param = layer['botP']
            except SlownessModelError as e:
                raise_from(RuntimeError('Please contact the developers. This '
                                        'error should not occur.'), e)
            if tau_model.sourceBranch != 0:
                self.current_branch = tau_model.sourceBranch - 1
            else:
                # p and s for zero source depth are only at zero distance
                # and then can be called P or S.
                self.max_ray_param = -1
                self.min_ray_param = -1
                return
        else:
            raise TauModelError(
                'First phase not recognised {}: Must be one of P, Pg, Pn, '
                'Pdiff, p, Ped or the S equivalents.'.format(current_leg))
        if self.receiver_depth != 0:
            if self.legs[-2] in ('Ped', 'Sed'):
                # Downgoing at receiver
                self.max_ray_param = min(
                    tau_model.get_tau_branch(
                        downgoing_rec_branch, is_p_wave).minTurnRayParam,
                    self.max_ray_param)
            else:
                # upgoing at receiver
                self.max_ray_param = min(
                    tau_model.get_tau_branch(upgoing_rec_branch,
                                             is_p_wave).minTurnRayParam,
                    self.max_ray_param)

        self.min_ray_param = 0

        isLegDepth, isNextLegDepth = False, False

        # Now loop over all the phase legs and construct the proper branch
        # sequence.
        current_leg = "START"  # So the prevLeg isn't wrong on the first pass.
        for legNum in range(len(self.legs) - 1):
            prevLeg = current_leg
            current_leg = next_leg
            next_leg = self.legs[legNum + 1]
            isLegDepth = isNextLegDepth

            # Find out if the next leg represents a phase conversion depth.
            try:
                nextLegDepth = float(next_leg)
                isNextLegDepth = True
            except ValueError:
                nextLegDepth = -1
                isNextLegDepth = False

            # Set currWave to be the wave type for this leg, "P" or "S".
            is_p_wave_previous = is_p_wave
            if current_leg in ("p", "k", "I") or current_leg[0] == "P":
                is_p_wave = True
            elif current_leg in ("s", "J") or current_leg[0] == "S":
                is_p_wave = False
            elif current_leg == "K":
                # Here we want to use whatever is_p_wave was on the last leg
                # so do nothing. This makes sure we use the correct
                # max_ray_param from the correct TauBranch within the outer
                # core. In other words K has a high slowness zone if it
                # entered the outer core as a mantle P wave, but doesn't if
                # it entered as a mantle S wave. It shouldn't matter for
                # inner core to outer core type legs.
                pass

            # Check to see if there has been a phase conversion.
            if len(self.branch_seq) > 0 and is_p_wave_previous != is_p_wave:
                self.phase_conversion(tau_model, self.branch_seq[-1],
                                      endAction, is_p_wave_previous)

            if current_leg in ('Ped', 'Sed'):
                if next_leg == "END":
                    if receiverDepth > 0:
                        endAction = REFLECT_TOPSIDE
                        self.add_to_branch(tau_model, self.current_branch,
                                           downgoing_rec_branch, is_p_wave,
                                           endAction)
                    else:
                        # This should be impossible except for 0 dist 0 source
                        # depth which can be called p or P.
                        self.max_ray_param = -1
                        self.min_ray_param = -1
                        return
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            # Deal with p and s case.
            elif current_leg in ("p", "s", "k"):
                if next_leg[0] == "v":
                    raise TauModelError(
                        "p and s must always be upgoing and cannot come "
                        "immediately before a top-sided reflection.")
                elif next_leg.startswith("^"):
                    disconBranch = closest_branch_to_depth(
                        tau_model, next_leg[1:])
                    if self.current_branch >= disconBranch:
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, disconBranch,
                            is_p_wave, endAction)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "current_branch > disconBranch".format(
                                current_leg, next_leg))
                elif next_leg == "m" and \
                        self.current_branch >= tau_model.mohoBranch:
                    endAction = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.mohoBranch,
                        is_p_wave, endAction)
                elif next_leg[0] in ("P", "S") or next_leg in ("K", "END"):
                    if next_leg == 'END':
                        disconBranch = upgoing_rec_branch
                    elif next_leg == 'K':
                        disconBranch = tau_model.cmbBranch
                    else:
                        disconBranch = 0
                    if current_leg == 'k' and next_leg != 'K':
                        endAction = _ACTIONS["transup"]
                    else:
                        endAction = _ACTIONS["reflect_underside"]
                    self.add_to_branch(
                        tau_model, self.current_branch, disconBranch,
                        is_p_wave, endAction)
                elif isNextLegDepth:
                    disconBranch = closest_branch_to_depth(tau_model, next_leg)
                    endAction = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, disconBranch,
                        is_p_wave, endAction)
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            # Now deal with P and S case.
            elif current_leg in ("P", "S"):
                if next_leg in ("P", "S", "Pn", "Sn", "END"):
                    if endAction == _ACTIONS["transdown"] or \
                            endAction == _ACTIONS["reflect_underside"]:
                        # Was downgoing, so must first turn in mantle.
                        endAction = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.cmbBranch - 1, is_p_wave,
                                           endAction)
                    if next_leg == 'END':
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           upgoing_rec_branch, is_p_wave,
                                           endAction)
                    else:
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, 0, is_p_wave,
                            endAction)
                elif next_leg[0] == "v":
                    disconBranch = closest_branch_to_depth(tau_model,
                                                           next_leg[1:])
                    if self.current_branch <= disconBranch - 1:
                        endAction = _ACTIONS["reflect_topside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           disconBranch - 1, is_p_wave,
                                           endAction)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "current_branch > disconBranch".format(current_leg,
                                                                   next_leg))
                elif next_leg[0] == "^":
                    disconBranch = closest_branch_to_depth(tau_model,
                                                           next_leg[1:])
                    if prevLeg == "K":
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           disconBranch, is_p_wave, endAction)
                    elif prevLeg[0] == "^" or prevLeg in ("P", "S", "p", "s",
                                                          "START"):
                        endAction = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.cmbBranch - 1, is_p_wave,
                                           endAction)
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, disconBranch,
                            is_p_wave, endAction)
                    elif ((prevLeg[0] == "v" and
                            disconBranch < closest_branch_to_depth(
                                tau_model, prevLeg[1:]) or
                           (prevLeg == "m" and
                               disconBranch < tau_model.mohoBranch) or
                           (prevLeg == "c" and
                               disconBranch < tau_model.cmbBranch))):
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, disconBranch,
                            is_p_wave, endAction)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "current_branch > disconBranch".format(current_leg,
                                                                   next_leg))
                elif next_leg == "c":
                    endAction = _ACTIONS["reflect_topside"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.cmbBranch - 1, is_p_wave, endAction)
                elif next_leg == "K":
                    endAction = _ACTIONS["transdown"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.cmbBranch - 1, is_p_wave, endAction)
                elif next_leg == "m" or (isNextLegDepth and
                                         nextLegDepth < tau_model.cmb_depth):
                    # Treat the Moho in the same way as 410 type
                    # discontinuities.
                    disconBranch = closest_branch_to_depth(tau_model, next_leg)
                    if endAction == _ACTIONS["turn"] \
                            or endAction == _ACTIONS["reflect_topside"] \
                            or endAction == _ACTIONS["transup"]:
                        # Upgoing section
                        if disconBranch > self.current_branch:
                            # Check the discontinuity below the current
                            # branch when the ray should be upgoing
                            raise TauModelError(
                                "Phase not recognised: {} followed by {} when "
                                "current_branch > disconBranch".format(
                                    current_leg, next_leg))
                        endAction = _ACTIONS["transup"]
                        self.add_to_branch(
                            tau_model, self.current_branch, disconBranch,
                            is_p_wave, endAction)
                    else:
                        # Downgoing section, must look at leg after next to
                        # determine whether to convert on the downgoing or
                        # upgoing part of the path.
                        nextnextLeg = self.legs[legNum + 2]
                        if nextnextLeg == "p" or nextnextLeg == "s":
                            # Convert on upgoing section
                            endAction = _ACTIONS["turn"]
                            self.add_to_branch(
                                tau_model, self.current_branch,
                                tau_model.cmbBranch - 1, is_p_wave, endAction)
                            endAction = _ACTIONS["transup"]
                            self.add_to_branch(tau_model, self.current_branch,
                                               disconBranch, is_p_wave,
                                               endAction)
                        elif nextnextLeg == "P" or nextnextLeg == "S":
                            if disconBranch > self.current_branch:
                                # discon is below current loc
                                endAction = _ACTIONS["transdown"]
                                self.add_to_branch(
                                    tau_model, self.current_branch,
                                    disconBranch - 1, is_p_wave, endAction)
                            else:
                                # Discontinuity is above current location,
                                # but we have a downgoing ray, so this is an
                                # illegal ray for this source depth.
                                self.max_ray_param = -1
                                return
                        else:
                            raise TauModelError(
                                "Phase not recognized: {} followed by {} "
                                "followed by {}".format(current_leg, next_leg,
                                                        nextnextLeg))
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            elif current_leg[0] in "PS":
                if current_leg == "Pdiff" or current_leg == "Sdiff":
                    # In the diffracted case we trick addtoBranch into
                    # thinking we are turning, but then make max_ray_param
                    # equal to min_ray_param, which is the deepest turning ray.
                    if (self.max_ray_param >= tau_model.get_tau_branch(
                            tau_model.cmbBranch - 1,
                            is_p_wave).minTurnRayParam >= self.min_ray_param):
                        endAction = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.cmbBranch - 1, is_p_wave,
                                           endAction)
                        self.max_ray_param = self.min_ray_param
                        if next_leg == "END":
                            endAction = _ACTIONS["reflect_underside"]
                            self.add_to_branch(tau_model, self.current_branch,
                                               upgoing_rec_branch, is_p_wave,
                                               endAction)
                        elif next_leg[0] in "PS":
                            endAction = _ACTIONS["reflect_underside"]
                            self.add_to_branch(
                                tau_model, self.current_branch, 0, is_p_wave,
                                endAction)
                    else:
                        # Can't have head wave as ray param is not within
                        # range.
                        self.max_ray_param = -1
                        return
                elif current_leg in ("Pg", "Sg", "Pn", "Sn"):
                    if self.current_branch >= tau_model.mohoBranch:
                        # Pg, Pn, Sg and Sn must be above the moho and so is
                        # not valid for rays coming upwards from below,
                        # possibly due to the source depth. Setting
                        # max_ray_param = -1 effectively disallows this phase.
                        self.max_ray_param = -1
                        return
                    if current_leg in ("Pg", "Sg"):
                        endAction = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.mohoBranch - 1, is_p_wave,
                                           endAction)
                        endAction = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           upgoing_rec_branch, is_p_wave,
                                           endAction)
                    elif current_leg in ("Pn", "Sn"):
                        # In the diffracted case we trick addtoBranch into
                        # thinking we are turning below the Moho, but then
                        # make the min_ray_param equal to max_ray_param,
                        # which is the head wave ray.
                        if (self.max_ray_param >= tau_model.get_tau_branch(
                                tau_model.mohoBranch,
                                is_p_wave).max_ray_param >=
                                self.min_ray_param):
                            endAction = _ACTIONS["turn"]
                            self.add_to_branch(
                                tau_model, self.current_branch,
                                tau_model.mohoBranch, is_p_wave, endAction)
                            endAction = _ACTIONS["transup"]
                            self.add_to_branch(
                                tau_model, self.current_branch,
                                tau_model.mohoBranch, is_p_wave, endAction)
                            self.min_ray_param = self.max_ray_param
                            if next_leg == "END":
                                endAction = _ACTIONS["reflect_underside"]
                                self.add_to_branch(
                                    tau_model, self.current_branch,
                                    upgoing_rec_branch, is_p_wave, endAction)
                            elif next_leg[0] in "PS":
                                endAction = _ACTIONS["reflect_underside"]
                                self.add_to_branch(
                                    tau_model, self.current_branch, 0,
                                    is_p_wave, endAction)
                        else:
                            # Can't have head wave as ray param is not
                            # within range.
                            self.max_ray_param = -1
                            return
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            elif current_leg == "K":
                if next_leg in ("P", "S"):
                    if prevLeg in ("P", "S", "K", "k", "START"):
                        endAction = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.iocbBranch - 1, is_p_wave,
                                           endAction)
                    endAction = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.cmbBranch,
                        is_p_wave, endAction)
                elif next_leg == "K":
                    if prevLeg in ("P", "S", "K"):
                        endAction = _ACTIONS["turn"]
                        self.add_to_branch(
                            tau_model, self.current_branch,
                            tau_model.iocbBranch - 1, is_p_wave, endAction)
                    endAction = _ACTIONS["reflect_underside"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.cmbBranch,
                        is_p_wave, endAction)
                elif next_leg in ("I", "J"):
                    endAction = _ACTIONS["transdown"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.iocbBranch - 1, is_p_wave, endAction)
                elif next_leg == "i":
                    endAction = _ACTIONS["reflect_topside"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.iocbBranch - 1, is_p_wave, endAction)
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            elif current_leg in ("I", "J"):
                endAction = _ACTIONS["turn"]
                self.add_to_branch(
                    tau_model, self.current_branch,
                    tau_model.tauBranches.shape[1] - 1, is_p_wave, endAction)
                if next_leg in ("I", "J"):
                    endAction = _ACTIONS["reflect_underside"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.iocbBranch,
                        is_p_wave, endAction)
                elif next_leg == "K":
                    endAction = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.iocbBranch,
                        is_p_wave, endAction)

            elif current_leg in ("m", "c", "i") or current_leg[0] == "^":
                pass

            elif current_leg[0] == "v":
                b = closest_branch_to_depth(tau_model, current_leg[1:])
                if b == 0:
                    raise TauModelError(
                        "Phase not recognized: {} looks like a top side "
                        "reflection at the free surface.".format(current_leg))

            elif isLegDepth:
                # Check for phase like P0s, but could also be P2s if first
                # discontinuity is deeper.
                b = closest_branch_to_depth(tau_model, current_leg)
                if b == 0 and next_leg in ("p", "s"):
                    raise TauModelError(
                        "Phase not recognized: {} followed by {} looks like "
                        "an upgoing wave from the free surface as closest "
                        "discontinuity to {} is zero depth.".format(
                            current_leg, next_leg, current_leg))

            else:
                raise TauModelError(
                    "Phase not recognized: {} followed by {}".format(
                        current_leg, next_leg))

        if self.max_ray_param != -1:
            if (endAction == _ACTIONS["reflect_underside"] and
                    downgoing_rec_branch == self.branch_seq[-1]):
                # Last action was upgoing, so last branch should be
                # upgoing_rec_branch
                self.min_ray_param = -1
                self.max_ray_param = -1
            elif (endAction == _ACTIONS["reflect_topside"] and
                    upgoing_rec_branch == self.branch_seq[-1]):
                # Last action was downgoing, so last branch should be
                # downgoing_rec_branch
                self.min_ray_param = -1
                self.max_ray_param = -1

    def phase_conversion(self, tau_model, fromBranch, endAction, isPtoS):
        """
        Change max_ray_param and min_ray_param where there is a phase
        conversion.

        For instance, SKP needs to change the max_ray_param because there are
        SKS ray parameters that cannot propagate from the CMB into the mantle
        as a P wave.
        """
        if endAction == _ACTIONS["turn"]:
            # Can't phase convert for just a turn point
            raise TauModelError("Bad endAction: phase conversion is not "
                                "allowed at turn points.")
        elif endAction == _ACTIONS["reflect_underside"]:
            self.max_ray_param = \
                min(self.max_ray_param,
                    tau_model.get_tau_branch(fromBranch, isPtoS).max_ray_param,
                    tau_model.get_tau_branch(fromBranch,
                                             not isPtoS).max_ray_param)
        elif endAction == _ACTIONS["reflect_topside"]:
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(fromBranch, isPtoS).minTurnRayParam,
                tau_model.get_tau_branch(fromBranch, not isPtoS).minTurnRayParam)
        elif endAction == _ACTIONS["transup"]:
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(fromBranch, isPtoS).max_ray_param,
                tau_model.get_tau_branch(fromBranch - 1,
                                         not isPtoS).minTurnRayParam)
        elif endAction == _ACTIONS["transdown"]:
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(fromBranch, isPtoS).min_ray_param,
                tau_model.get_tau_branch(fromBranch + 1,
                                         not isPtoS).max_ray_param)
        else:
            raise TauModelError("Illegal endAction = {}".format(endAction))

    def add_to_branch(self, tau_model, startBranch, endBranch, isPWave,
                      endAction):
        """
        Add branch numbers to branch_seq.

        Branches from startBranch to endBranch, inclusive, are added in order.
        Also, current_branch is set correctly based on the value of endAction.
        endAction can be one of transup, transdown, reflect_underside,
        reflect_topside, or turn.
        """
        if endBranch < 0 or endBranch > tau_model.tauBranches.shape[1]:
            raise ValueError('End branch outside range: %d' % (endBranch, ))

        if endAction == _ACTIONS["turn"]:
            endOffset = 0
            isDownGoing = True
            self.min_ray_param = max(
                self.min_ray_param,
                tau_model.get_tau_branch(endBranch, isPWave).minTurnRayParam)
        elif endAction == _ACTIONS["reflect_underside"]:
            endOffset = 0
            isDownGoing = False
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(endBranch, isPWave).max_ray_param)
        elif endAction == _ACTIONS["reflect_topside"]:
            endOffset = 0
            isDownGoing = True
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(endBranch, isPWave).minTurnRayParam)
        elif endAction == _ACTIONS["transup"]:
            endOffset = -1
            isDownGoing = False
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(endBranch, isPWave).max_ray_param)
        elif endAction == _ACTIONS["transdown"]:
            endOffset = 1
            isDownGoing = True
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(endBranch, isPWave).min_ray_param)
        else:
            raise TauModelError("Illegal endAction: {}".format(endAction))

        if isDownGoing:
            if startBranch > endBranch:
                # Can't be downgoing as we are already below.
                self.min_ray_param = -1
                self.max_ray_param = -1
            else:
                # Must be downgoing, so increment i.
                for i in range(startBranch, endBranch + 1):
                    self.branch_seq.append(i)
                    self.down_going.append(isDownGoing)
                    self.wave_type.append(isPWave)
        else:
            if startBranch < endBranch:
                # Can't be upgoing as we are already above.
                self.min_ray_param = -1
                self.max_ray_param = -1
            else:
                # Upgoing, so decrement i.
                for i in range(startBranch, endBranch - 1, -1):
                    self.branch_seq.append(i)
                    self.down_going.append(isDownGoing)
                    self.wave_type.append(isPWave)
        self.current_branch = endBranch + endOffset

    def sum_branches(self, tau_model):
        """Sum the appropriate branches for this phase."""
        # Special case for surface waves.
        if self.name.endswith("kmps"):
            self.dist = np.zeros(2)
            self.time = np.zeros(2)
            self.ray_param = np.empty(2)

            self.ray_param[0] = \
                tau_model.radius_of_planet / float(self.name[:-4])

            self.dist[1] = 2 * math.pi
            self.time[1] = \
                2 * math.pi * tau_model.radius_of_planet / \
                float(self.name[:-4])
            self.ray_param[1] = self.ray_param[0]

            self.min_distance = 0
            self.max_distance = 2 * math.pi
            self.down_going.append(True)
            return

        if self.max_ray_param < 0 or self.min_ray_param > self.max_ray_param:
            # Phase has no arrivals, possibly due to source depth.
            self.ray_param = np.empty(0)
            self.min_ray_param = -1
            self.max_ray_param = -1
            self.dist = np.empty(0)
            self.time = np.empty(0)
            self.max_distance = -1
            return

        # Find the ray parameter index that corresponds to the min_ray_param
        # and max_ray_param.
        index = np.where(tau_model.ray_params >= self.min_ray_param)[0]
        if len(index):
            self.min_ray_param_index = index[-1]
        index = np.where(tau_model.ray_params >= self.max_ray_param)[0]
        if len(index):
            self.max_ray_param_index = index[-1]
        if self.max_ray_param_index == 0 \
                and self.min_ray_param_index == len(tau_model.ray_params) - 1:
            # All ray parameters are valid so just copy:
            self.ray_param = tau_model.ray_param.copy()
        elif self.max_ray_param_index == self.min_ray_param_index:
            # if "Sdiff" in self.name or "Pdiff" in self.name:
            # self.ray_param = [self.min_ray_param, self.min_ray_param]
            # elif "Pn" in self.name or "Sn" in self.name:
            # self.ray_param = [self.min_ray_param, self.min_ray_param]
            if self.name.endswith("kmps"):
                self.ray_param = np.array([0, self.max_ray_param])
            else:
                self.ray_param = np.array([self.min_ray_param,
                                           self.min_ray_param])
        else:
            # Only a subset of the ray parameters is valid so use these.
            self.ray_param = \
                tau_model.ray_params[self.max_ray_param_index:
                                     self.min_ray_param_index + 1].copy()

        self.dist = np.zeros(shape=self.ray_param.shape)
        self.time = np.zeros(shape=self.ray_param.shape)

        # Counter for passes through each branch. 0 is P and 1 is S.
        timesBranches = self.calc_branch_mult(tau_model)

        # Sum the branches with the appropriate multiplier.
        size = self.min_ray_param_index - self.max_ray_param_index + 1
        index = slice(self.max_ray_param_index, self.min_ray_param_index + 1)
        for i in range(tau_model.tauBranches.shape[1]):
            tb = timesBranches[0, i]
            tbs = timesBranches[1, i]
            taub = tau_model.tauBranches[0, i]
            taubs = tau_model.tauBranches[1, i]

            if tb != 0:
                self.dist[:size] += tb * taub.dist[index]
                self.time[:size] += tb * taub.time[index]
            if tbs != 0:
                self.dist[:size] += tbs * taubs.dist[index]
                self.time[:size] += tbs * taubs.time[index]

        if "Sdiff" in self.name or "Pdiff" in self.name:
            if tau_model.sMod.depthInHighSlowness(tau_model.cmb_depth - 1e-10,
                                                  self.min_ray_param,
                                                  self.name[0] == "P"):
                # No diffraction if there is a high slowness zone at the CMB.
                self.min_ray_param = -1
                self.max_ray_param = -1
                self.max_distance = -1
                self.time = np.empty(0)
                self.dist = np.empty(0)
                self.ray_param = np.empty(0)
                return
            else:
                self.dist[1] = self.dist[0] + \
                    self._settings["max_diffraction_in_radians"]
                self.time[1] = (
                    self.time[0] +
                    self._settings["max_diffraction_in_radians"] *
                    self.min_ray_param)

        elif "Pn" in self.name or "Sn" in self.name:
            self.dist[1] = self.dist[0] + \
                self._settings["max_refraction_in_radians"]
            self.time[1] = (
                self.time[0] + self._settings["max_refraction_in_radians"] *
                self.min_ray_param)

        elif self.max_ray_param_index == self.min_ray_param_index:
            self.dist[1] = self.dist[0]
            self.time[1] = self.time[0]

        self.min_distance = np.min(self.dist)
        self.max_distance = np.max(self.dist)

        # Now check to see if our ray parameter range includes any ray
        # parameters that are associated with high slowness zones. If so,
        # then we will need to insert a "shadow zone" into our time and
        # distance arrays. It is represented by a repeated ray parameter.
        for isPwave in [True, False]:
            hsz = tau_model.sMod.highSlownessLayerDepthsP \
                if isPwave \
                else tau_model.sMod.highSlownessLayerDepthsS
            indexOffset = 0
            for hszi in hsz:
                if self.max_ray_param > hszi.ray_param > self.min_ray_param:
                    # There is a high slowness zone within our ray parameter
                    # range so might need to add a shadow zone. Need to
                    # check if the current wave type is part of the phase at
                    # this depth/ray parameter.
                    branchNum = tau_model.find_branch(hszi.topDepth)
                    foundOverlap = False
                    for legNum in range(len(self.branch_seq)):
                        # Check for downgoing legs that cross the high
                        # slowness zone with the same wave type.
                        if (self.branch_seq[legNum] == branchNum and
                                self.wave_type[legNum] == isPwave and
                                self.down_going[legNum] is True and
                                self.branch_seq[legNum - 1] ==
                                branchNum - 1 and
                                self.wave_type[legNum - 1] == isPwave and
                                self.down_going[legNum - 1] is True):
                            foundOverlap = True
                            break
                    if foundOverlap:
                        hszIndex = np.where(self.ray_param == hszi.ray_param)
                        hszIndex = hszIndex[0][0]

                        newlen = self.ray_param.shape[0] + 1
                        new_ray_params = np.empty(newlen)
                        newdist = np.empty(newlen)
                        newtime = np.empty(newlen)

                        new_ray_params[:hszIndex] = self.ray_param[:hszIndex]
                        newdist[:hszIndex] = self.dist[:hszIndex]
                        newtime[:hszIndex] = self.time[:hszIndex]

                        # Sum the branches with an appropriate multiplier.
                        new_ray_params[hszIndex] = hszi.ray_param
                        newdist[hszIndex] = 0
                        newtime[hszIndex] = 0
                        for tb, tbs, taub, taubs in zip(
                                timesBranches[0], timesBranches[1],
                                tau_model.tauBranches[0],
                                tau_model.tauBranches[1]):
                            if tb != 0 and taub.topDepth < hszi.topDepth:
                                newdist[hszIndex] += tb * taub.dist[
                                    self.max_ray_param_index + hszIndex -
                                    indexOffset]
                                newtime[hszIndex] += tb * taub.time[
                                    self.max_ray_param_index + hszIndex -
                                    indexOffset]
                            if tbs != 0 and taubs.topDepth < hszi.topDepth:
                                newdist[hszIndex] += tbs * taubs.dist[
                                    self.max_ray_param_index + hszIndex -
                                    indexOffset]
                                newtime[hszIndex] += tbs * taubs.time[
                                    self.max_ray_param_index + hszIndex -
                                    indexOffset]

                        newdist[hszIndex + 1:] = self.dist[hszIndex:]
                        newtime[hszIndex + 1:] = self.time[hszIndex:]
                        new_ray_params[hszIndex + 1:] = \
                            self.ray_param[hszIndex:]

                        indexOffset += 1
                        self.dist = newdist
                        self.time = newtime
                        self.ray_param = new_ray_params

    def calc_branch_mult(self, tau_model):
        """
        Calculate how many times the phase passes through a branch, up or down.

        With this result, we can just multiply instead of doing the ray calc
        for each time.
        """
        # Initialise the counter for each branch to 0. 0 is P and 1 is S.
        timesBranches = np.zeros((2, tau_model.tauBranches.shape[1]))
        # Count how many times each branch appears in the path.
        # wave_type is at least as long as branch_seq
        for wt, bs in zip(self.wave_type, self.branch_seq):
            if wt:
                timesBranches[0][bs] += 1
            else:
                timesBranches[1][bs] += 1
        return timesBranches

    def calc_time(self, degrees):
        """
        Calculate arrival times for this phase, sorted by time.
        """
        # 10 should be enough. This is only for one phase.
        r_dist = np.empty(10, dtype=np.float64)
        r_ray_num = np.empty(10, dtype=np.int32)

        # This saves around 17% runtime when calculating arrival times which
        # is probably the major use case.
        phase_count = clibtau.seismic_phase_calc_time_inner_loop(
            float(degrees),
            self.max_distance,
            self.dist,
            self.ray_param,
            r_dist,
            r_ray_num,
            len(self.dist)
        )

        arrivals = []
        for _i in range(phase_count):
            arrivals.append(self.refine_arrival(
                degrees, r_ray_num[_i], r_dist[_i], REFINE_DIST_RADIAN_TOL,
                self._settings["max_recursion"]))
        return arrivals

    def calc_pierce(self, degrees):
        """
        Calculate pierce points for this phase.

        First calculates arrivals, then the "pierce points" corresponding to
        the stored arrivals. The pierce points are stored within each arrival
        object.
        """
        arrivals = self.calc_time(degrees)
        for arrival in arrivals:
            self.calc_pierce_from_arrival(arrival)
        return arrivals

    def calc_pierce_from_arrival(self, currArrival):
        """
        Calculate the pierce points for a particular arrival.

        The returned arrival is the same as the input argument but now has the
        pierce points filled in.
        """
        # Find the ray parameter index that corresponds to the arrival ray
        # parameter in the TauModel, ie it is between rayNum and rayNum+1,
        # We know that it must be <model.ray_param.length-1 since the last
        # ray parameter sample is 0 in a spherical model.
        rayNum = 0
        for i, rp in enumerate(self.tau_model.ray_params[:-1]):
            if rp >= currArrival.ray_param:
                rayNum = i
            else:
                break

        # Here we use ray parameter and dist info stored within the
        # SeismicPhase so we can use currArrival.ray_param_index, which
        # may not correspond to rayNum (for model.ray_param).
        ray_param_a = self.ray_param[currArrival.ray_param_index]
        ray_param_b = self.ray_param[currArrival.ray_param_index + 1]
        distA = self.dist[currArrival.ray_param_index]
        distB = self.dist[currArrival.ray_param_index + 1]
        distRatio = (currArrival.purist_dist - distA) / (distB - distA)
        distRayParam = distRatio * (ray_param_b - ray_param_a) + ray_param_a

        # + 2 for first point and kmps, if it exists.
        pierce = np.empty(len(self.branch_seq) + 2, dtype=TimeDist)
        # First pierce point is always 0 distance at the source depth.
        pierce[0] = (distRayParam, 0, 0, self.tau_model.source_depth)
        index = 1
        branchDist = 0
        branchTime = 0

        # Loop from 0 but already done 0 [I just copy the comments, sorry!],
        # so the pierce point when the ray leaves branch i is stored in i + 1.
        # Use linear interpolation between rays that we know.
        assert len(self.branch_seq) == len(self.wave_type) == \
            len(self.down_going)
        for branchNum, isPWave, isDownGoing in zip(self.branch_seq,
                                                   self.wave_type,
                                                   self.down_going):
            # Save the turning depths for the ray parameter for both P and
            # S waves. This way we get the depth correct for any rays that
            # turn within a layer. We have to do this on a per branch basis
            # because of converted phases, e.g. SKS.
            tauBranch = self.tau_model.get_tau_branch(branchNum, isPWave)
            if distRayParam > tauBranch.max_ray_param:
                turnDepth = tauBranch.topDepth
            elif distRayParam <= tauBranch.min_ray_param:
                turnDepth = tauBranch.botDepth
            else:
                if (isPWave or self.tau_model.sMod.depthInFluid((
                        tauBranch.topDepth + tauBranch.botDepth) / 2)):
                    turnDepth = self.tau_model.sMod.findDepth_from_depths(
                        distRayParam,
                        tauBranch.topDepth,
                        tauBranch.botDepth,
                        True)
                else:
                    turnDepth = self.tau_model.sMod.findDepth_from_depths(
                        distRayParam,
                        tauBranch.topDepth,
                        tauBranch.botDepth,
                        isPWave)

            if any(x in self.name for x in ["Pdiff", "Pn", "Sdiff", "Sn"]):
                # Head waves and diffracted waves are a special case.
                distA = tauBranch.dist[rayNum]
                timeA = tauBranch.time[rayNum]
                distB, timeB = distA, timeA
            else:
                distA = tauBranch.dist[rayNum]
                timeA = tauBranch.time[rayNum]
                distB = tauBranch.dist[rayNum + 1]
                timeB = tauBranch.time[rayNum + 1]

            branchDist += distRatio * (distB - distA) + distA
            prevBranchTime = np.array(branchTime, copy=True)
            branchTime += distRatio * (timeB - timeA) + timeA
            if isDownGoing:
                branchDepth = min(tauBranch.botDepth, turnDepth)
            else:
                branchDepth = min(tauBranch.topDepth, turnDepth)

            # Make sure ray actually propagates in this branch; leave a little
            # room for numerical chatter.
            if abs(prevBranchTime - branchTime) > 1e-10:
                pierce[index] = (distRayParam, branchTime, branchDist,
                                 branchDepth)
                index += 1

        if any(x in self.name for x in ["Pdiff", "Pn", "Sdiff", "Sn"]):
            pierce, index = self.handle_special_waves(currArrival,
                                                      pierce,
                                                      index)
        elif "kmps" in self.name:
            pierce[index] = (distRayParam, currArrival.time,
                             currArrival.purist_dist, 0)
            index += 1

        currArrival.pierce = pierce[:index]
        # The arrival is modified in place and must (?) thus be returned.
        return currArrival

    def calc_path(self, degrees):
        """
        Calculate the paths this phase takes through the planet model.

        Only calls :meth:`calc_path_from_arrival`.
        """
        arrivals = self.calc_time(degrees)
        for arrival in arrivals:
            self.calc_path_from_arrival(arrival)
        return arrivals

    def calc_path_from_arrival(self, currArrival):
        """
        Calculate the paths this phase takes through the planet model.
        """
        # Find the ray parameter index that corresponds to the arrival ray
        # parameter in the TauModel, i.e. it is between rayNum and rayNum + 1.
        tempTimeDist = (currArrival.ray_param, 0, 0,
                        self.tau_model.source_depth)
        # pathList is a list of lists.
        pathList = [tempTimeDist]
        for i, branchNum, isPWave, isDownGoing in zip(count(), self.branch_seq,
                                                      self.wave_type,
                                                      self.down_going):
            br = self.tau_model.get_tau_branch(branchNum, isPWave)
            tempTimeDist = br.path(currArrival.ray_param, isDownGoing,
                                   self.tau_model.sMod)
            if len(tempTimeDist):
                pathList.extend(tempTimeDist)
                if np.any(tempTimeDist['dist'] < 0):
                    raise RuntimeError("Path is backtracking, "
                                       "this is impossible.")

            # Special case for head and diffracted waves:
            if(branchNum == self.tau_model.cmbBranch - 1 and
               i < len(self.branch_seq) - 1 and
               self.branch_seq[i + 1] == self.tau_model.cmbBranch - 1 and
               ("Pdiff" in self.name or "Sdiff" in self.name)):
                dist_diff = currArrival.purist_dist - self.dist[0]
                diffTD = (
                    currArrival.ray_param,
                    dist_diff * currArrival.ray_param,
                    dist_diff,
                    self.tau_model.cmb_depth)
                pathList.append(diffTD)

            elif(branchNum == self.tau_model.mohoBranch - 1 and
                 i < len(self.branch_seq) - 1 and
                 self.branch_seq[i + 1] == self.tau_model.mohoBranch - 1 and
                 ("Pn" in self.name or "Sn" in self.name)):
                # Can't have both Pn and Sn in a wave, so one of these is 0.
                numFound = max(self.name.count("Pn"), self.name.count("Sn"))
                dist_head = (currArrival.purist_dist - self.dist[0]) / numFound
                headTD = (
                    currArrival.ray_param,
                    dist_head * currArrival.ray_param,
                    dist_head,
                    self.tau_model.moho_depth)
                pathList.append(headTD)

        if "kmps" in self.name:
            # kmps phases have no branches, so need to end them at the arrival
            # distance.
            headTD = (
                currArrival.ray_param,
                currArrival.purist_dist * currArrival.ray_param,
                currArrival.purist_dist,
                0)
            pathList.append(headTD)

        currArrival.path = np.array(pathList, dtype=TimeDist)
        np.cumsum(currArrival.path['time'], out=currArrival.path['time'])
        np.cumsum(currArrival.path['dist'], out=currArrival.path['dist'])

        return currArrival

    def handle_special_waves(self, currArrival, pierce, index):
        """
        Handle head or diffracted waves.

        It is assumed that a phase can be a diffracted wave or a head wave, but
        not both. Nor can it be a head wave or diffracted wave for both P and
        S.
        """
        for ps in ["Pn", "Sn", "Pdiff", "Sdiff"]:
            if ps in self.name:
                phaseSeg = ps
                break
        else:
            raise TauModelError("No head/diff segment in" + str(self.name))

        if phaseSeg in ["Pn", "Sn"]:
            headDepth = self.tau_model.moho_depth
        else:
            headDepth = self.tau_model.cmb_depth

        numFound = self.name.count(phaseSeg)
        refractDist = currArrival.purist_dist - self.dist[0]
        refractTime = refractDist * currArrival.ray_param

        # This is a little weird as we are not checking where we are in
        # the phase name, but simply if the depth matches. This likely
        # works in most cases, but may not for head/diffracted waves that
        # undergo a phase change, if that type of phase can even exist.
        mask = pierce['depth'][:index] == headDepth
        adjust = np.cumsum(mask)
        pierce['time'][:index] += adjust * refractTime / numFound
        pierce['dist'][:index] += adjust * refractDist / numFound

        head_index = np.where(mask)[0]
        if len(head_index):
            head_index += 1
            td = pierce[head_index]
            pierce = np.insert(pierce, head_index, td)
            index += len(head_index)

        return pierce, index

    def refine_arrival(self, degrees, ray_index, dist_radian, tolerance,
                       recursion_limit):
        left = Arrival(self, degrees, self.time[ray_index],
                       self.dist[ray_index], self.ray_param[ray_index],
                       ray_index, self.name, self.purist_name,
                       self.source_depth, self.receiver_depth)
        right = Arrival(self, degrees, self.time[ray_index + 1],
                        self.dist[ray_index + 1],
                        self.ray_param[ray_index + 1],
                        # Use ray_index since dist is between ray_index and
                        # (ray_index + 1).
                        ray_index, self.name, self.purist_name,
                        self.source_depth, self.receiver_depth)
        return self._refine_arrival(degrees, left, right, dist_radian,
                                    tolerance, recursion_limit)

    def _refine_arrival(self, degrees, left_estimate, right_estimate,
                        search_dist, tolerance, recursion_limit):
        new_estimate = self.linear_interp_arrival(degrees, search_dist,
                                                  left_estimate,
                                                  right_estimate)
        if (recursion_limit <= 0 or self.name.endswith('kmps') or
                any(phase in self.name
                    for phase in ['Pdiff', 'Sdiff', 'Pn', 'Sn'])):
            # can't shoot/refine for non-body waves
            return new_estimate

        try:
            shoot = self.shoot_ray(degrees, new_estimate.ray_param)
            if ((left_estimate.purist_dist - search_dist) *
                    (search_dist - shoot.purist_dist)) > 0:
                # search between left and shoot
                if abs(shoot.purist_dist -
                       new_estimate.purist_dist) < tolerance:
                    return self.linear_interp_arrival(degrees, search_dist,
                                                      left_estimate, shoot)
                else:
                    return self._refine_arrival(degrees, left_estimate, shoot,
                                                search_dist, tolerance,
                                                recursion_limit - 1)
            else:
                # search between shoot and right
                if abs(shoot.purist_dist -
                       new_estimate.purist_dist) < tolerance:
                    return self.linear_interp_arrival(degrees, search_dist,
                                                      shoot, right_estimate)
                else:
                    return self._refine_arrival(degrees, shoot, right_estimate,
                                                search_dist, tolerance,
                                                recursion_limit - 1)
        except (IndexError, LookupError, SlownessModelError) as e:
            raise_from(RuntimeError('Please contact the developers. This '
                                    'error should not occur.'), e)

    def shoot_ray(self, degrees, ray_param):
        if (any(phase in self.name
                for phase in ['Pdiff', 'Sdiff', 'Pn', 'Sn']) or
                self.name.endswith('kmps')):
            raise SlownessModelError('Unable to shoot ray in non-body waves')

        if ray_param < self.min_ray_param or self.max_ray_param < ray_param:
            msg = 'Ray param %f is outside range for this phase: min=%f max=%f'
            raise SlownessModelError(msg % (ray_param, self.min_ray_param,
                                            self.max_ray_param))

        # looks like a body wave and ray param can propagate
        for ray_param_index in range(len(self.ray_param) - 1):
            if self.ray_param[ray_param_index + 1] < ray_param:
                break

        tau_model = self.tau_model
        sMod = tau_model.sMod

        # counter for passes through each branch. 0 is P and 1 is S.
        timesBranches = self.calc_branch_mult(tau_model)
        time = np.zeros(1)
        dist = np.zeros(1)
        ray_param = np.array([ray_param])

        # Sum the branches with the appropriate multiplier.
        for j in range(tau_model.tauBranches.shape[1]):
            if timesBranches[0, j] != 0:
                br = tau_model.get_tau_branch(j, sMod.PWAVE)
                top_layer = sMod.layer_number_below(br.topDepth, sMod.PWAVE)
                bot_layer = sMod.layer_number_above(br.botDepth, sMod.PWAVE)
                td = br.calc_time_dist(sMod, top_layer, bot_layer, ray_param,
                                       allow_turn_in_layer=True)

                time += timesBranches[0, j] * td['time']
                dist += timesBranches[0, j] * td['dist']

            if timesBranches[1, j] != 0:
                br = tau_model.get_tau_branch(j, sMod.SWAVE)
                top_layer = sMod.layer_number_below(br.topDepth, sMod.SWAVE)
                bot_layer = sMod.layer_number_above(br.botDepth, sMod.SWAVE)
                td = br.calc_time_dist(sMod, top_layer, bot_layer, ray_param,
                                       allow_turn_in_layer=True)

                time += timesBranches[1, j] * td['time']
                dist += timesBranches[1, j] * td['dist']

        return Arrival(self, degrees, time[0], dist[0], ray_param[0],
                       ray_param_index, self.name, self.purist_name,
                       self.source_depth, self.receiver_depth)

    def linear_interp_arrival(self, degrees, search_dist, left, right):
        if left.ray_param_index == 0 and search_dist == self.dist[0]:
            # degenerate case
            return Arrival(self, degrees, self.time[0], search_dist,
                           self.ray_param[0], 0, self.name, self.purist_name,
                           self.source_depth, self.receiver_depth, 0, 0)

        if left.purist_dist == search_dist:
            return left

        arrival_time = ((search_dist - left.purist_dist) /
                        (right.purist_dist - left.purist_dist) *
                        (right.time - left.time)) + left.time
        if math.isnan(arrival_time):
            msg = ('Time is NaN, search=%f leftDist=%f leftTime=%f '
                   'rightDist=%f rightTime=%f')
            raise RuntimeError(msg % (search_dist, left.purist_dist, left.time,
                                      right.purist_dist, right.time))

        ray_param = ((search_dist - right.purist_dist) /
                     (left.purist_dist - right.purist_dist) *
                     (left.ray_param - right.ray_param)) + right.ray_param
        return Arrival(self, degrees, arrival_time, search_dist, ray_param,
                       left.ray_param_index, self.name, self.purist_name,
                       self.source_depth, self.receiver_depth)

    def calc_ray_param_for_takeoff(self, takeoff_degree):
        vMod = self.tau_model.sMod.vMod
        try:
            if self.down_going[0]:
                takeoff_velocity = vMod.evaluate_below(self.source_depth,
                                                       self.name[0])
            else:
                takeoff_velocity = vMod.evaluate_above(self.source_depth,
                                                       self.name[0])
        except (IndexError, LookupError) as e:
            raise_from(RuntimeError('Please contact the developers. This '
                                    'error should not occur.'), e)

        return ((self.tau_model.radius_of_planet - self.source_depth) *
                math.sin(np.radians(takeoff_degree)) / takeoff_velocity)

    def calc_takeoff_angle(self, ray_param):
        if self.name.endswith('kmps'):
            return 0

        vMod = self.tau_model.sMod.vMod
        try:
            if self.down_going[0]:
                takeoff_velocity = vMod.evaluate_below(self.source_depth,
                                                       self.name[0])
            else:
                takeoff_velocity = vMod.evaluate_above(self.source_depth,
                                                       self.name[0])
        except (IndexError, LookupError) as e:
            raise_from(RuntimeError('Please contact the developers. This '
                                    'error should not occur.'), e)

        takeoff_angle = np.degrees(math.asin(np.clip(
            takeoff_velocity * ray_param /
            (self.tau_model.radius_of_planet - self.source_depth), -1.0, 1.0)))
        if not self.down_going[0]:
            # upgoing, so angle is in 90-180 range
            takeoff_angle = 180 - takeoff_angle

        return takeoff_angle

    def calc_incident_angle(self, ray_param):
        if self.name.endswith('kmps'):
            return 0

        vMod = self.tau_model.sMod.vMod
        # Very last item is "END", assume first char is P or S
        lastLeg = self.legs[-2][0]
        try:
            if self.down_going[-1]:
                incident_velocity = vMod.evaluate_above(self.receiver_depth,
                                                        lastLeg)
            else:
                incident_velocity = vMod.evaluate_below(self.receiver_depth,
                                                        lastLeg)
        except (IndexError, LookupError) as e:
            raise_from(RuntimeError('Please contact the developers. This '
                                    'error should not occur.'), e)

        incident_angle = np.degrees(math.asin(np.clip(
            incident_velocity * ray_param /
            (self.tau_model.radius_of_planet - self.receiver_depth),
            -1.0, 1.0)))
        if self.down_going[-1]:
            incident_angle = 180 - incident_angle

        return incident_angle

    @classmethod
    def get_earliest_arrival(cls, relPhases, degrees):
        raise NotImplementedError("baaa")


def closest_branch_to_depth(tau_model, depthString):
    """
    Find the closest discontinuity to the given depth that can have
    reflections and phase transformations.
    """
    if depthString == "m":
        return tau_model.mohoBranch
    elif depthString == "c":
        return tau_model.cmbBranch
    elif depthString == "i":
        return tau_model.iocbBranch
    # Non-standard boundary, given by a number: must look for it.
    disconBranch = -1
    disconMax = 1e300
    disconDepth = float(depthString)
    for i, tBranch in enumerate(tau_model.tauBranches[0]):
        if (abs(disconDepth - tBranch.topDepth) < disconMax and not
                any(ndc == tBranch.topDepth
                    for ndc in tau_model.noDisconDepths)):
            disconBranch = i
            disconMax = abs(disconDepth - tBranch.topDepth)
    return disconBranch


def self_tokenizer(scanner, token):
    return token


def wrong_phase(scanner, token):
    raise ValueError("Invalid phase name: %s cannot be followed by %s in %s" %
                     (token[0], token[1], token))


tokenizer = re.Scanner([
    # Surface wave phase velocity "phases"
    (r"\.?\d+\.?\d*kmps", self_tokenizer),
    # Composite legs.
    (r"Pn|Sn|Pg|Sg|Pb|Sb|Pdiff|Sdiff|Ped|Sed", self_tokenizer),
    # Reflections.
    (r"([\^v])([mci]|\.?\d+\.?\d*)", self_tokenizer),
    # Invalid phases.
    (r"[PS][ps]", wrong_phase),
    # Single legs.
    (r"[KkIiJmcPpSs]", self_tokenizer),
    # Single numerical value
    (r"\.?\d+\.?\d*", self_tokenizer)
])


def leg_puller(name):
    """
    Tokenize a phase name into legs.

    For example, ``PcS`` becomes ``'P' + 'c' + 'S'`` while ``p^410P`` would
    become ``'p' + '^410' + 'P'``. Once a phase name has been broken into
    tokens, we can begin to construct the sequence of branches to which it
    corresponds. Only minor error checking is done at this point, for instance
    ``PIP`` generates an exception but ``^410`` doesn't. It also appends
    ``"END"`` as the last leg.
    """
    results, remainder = tokenizer.scan(name)
    if remainder:
        raise ValueError("Invalid phase name: %s could not be parsed in %s"
                         % (str(remainder), name))
    return results + ["END"]
