# -*- coding: utf-8 -*-
"""
Objects and functions dealing with seismic phases.
"""
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
                discon_branch = closest_branch_to_depth(tau_model,
                                                        current_leg[1:])
                leg_depth = tau_model.tau_branches[0, discon_branch].top_depth
                purist_name += current_leg[0]
                purist_name += str(int(round(leg_depth)))
            else:
                try:
                    float(current_leg)
                except ValueError:
                    # If current_leg is just a string:
                    purist_name += current_leg
                else:
                    # If it is indeed a number:
                    discon_branch = closest_branch_to_depth(tau_model,
                                                            current_leg)
                    leg_depth = \
                        tau_model.tau_branches[0, discon_branch].top_depth
                    purist_name += str(leg_depth)
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
        if "J" in self.name and not tau_model.s_mod.allow_inner_core_s:
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
        # model.source_branch-1 and downgoing would be model.source_branch.
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
            self.current_branch = tau_model.source_branch
            # Treat initial downgoing as if it were an underside reflection.
            end_action = _ACTIONS["reflect_underside"]
            try:
                s_layer_num = tau_model.s_mod.layer_number_below(
                    tau_model.source_depth, is_p_wave_previous)
                layer = tau_model.s_mod.get_slowness_layer(
                    s_layer_num, is_p_wave_previous)
                self.max_ray_param = layer['top_p']
            except SlownessModelError as e:
                msg = ('Please contact the developers. This error should not '
                       'occur.')
                raise RuntimeError(msg) from e
            self.max_ray_param = tau_model.get_tau_branch(
                tau_model.source_branch, is_p_wave).max_ray_param
        elif current_leg in ("p", "s") or (
                self.expert and current_leg[0] == "k"):
            # Upgoing from source: treat initial downgoing as if it were a
            # topside reflection.
            end_action = _ACTIONS["reflect_topside"]
            try:
                s_layer_num = tau_model.s_mod.layer_number_above(
                    tau_model.source_depth, is_p_wave_previous)
                layer = tau_model.s_mod.get_slowness_layer(
                    s_layer_num, is_p_wave_previous)
                self.max_ray_param = layer['bot_p']
            except SlownessModelError as e:
                msg = ('Please contact the developers. This error should not '
                       'occur.')
                raise RuntimeError(msg) from e
            if tau_model.source_branch != 0:
                self.current_branch = tau_model.source_branch - 1
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
                        downgoing_rec_branch, is_p_wave).min_turn_ray_param,
                    self.max_ray_param)
            else:
                # upgoing at receiver
                self.max_ray_param = min(
                    tau_model.get_tau_branch(upgoing_rec_branch,
                                             is_p_wave).min_turn_ray_param,
                    self.max_ray_param)

        self.min_ray_param = 0

        is_leg_depth, is_next_leg_depth = False, False

        # Now loop over all the phase legs and construct the proper branch
        # sequence.
        current_leg = "START"  # So the prev_leg isn't wrong on the first pass.
        for leg_num in range(len(self.legs) - 1):
            prev_leg = current_leg
            current_leg = next_leg
            next_leg = self.legs[leg_num + 1]
            is_leg_depth = is_next_leg_depth

            # Find out if the next leg represents a phase conversion depth.
            try:
                next_leg_depth = float(next_leg)
                is_next_leg_depth = True
            except ValueError:
                next_leg_depth = -1
                is_next_leg_depth = False

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
                                      end_action, is_p_wave_previous)

            if current_leg in ('Ped', 'Sed'):
                if next_leg == "END":
                    if self.receiver_depth > 0:
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           downgoing_rec_branch, is_p_wave,
                                           end_action)
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
                    discon_branch = closest_branch_to_depth(
                        tau_model, next_leg[1:])
                    if self.current_branch >= discon_branch:
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, discon_branch,
                            is_p_wave, end_action)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "current_branch > discon_branch".format(
                                current_leg, next_leg))
                elif next_leg == "m" and \
                        self.current_branch >= tau_model.moho_branch:
                    end_action = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.moho_branch,
                        is_p_wave, end_action)
                elif next_leg[0] in ("P", "S") or next_leg in ("K", "END"):
                    if next_leg == 'END':
                        discon_branch = upgoing_rec_branch
                    elif next_leg == 'K':
                        discon_branch = tau_model.cmb_branch
                    else:
                        discon_branch = 0
                    if current_leg == 'k' and next_leg != 'K':
                        end_action = _ACTIONS["transup"]
                    else:
                        end_action = _ACTIONS["reflect_underside"]
                    self.add_to_branch(
                        tau_model, self.current_branch, discon_branch,
                        is_p_wave, end_action)
                elif is_next_leg_depth:
                    discon_branch = closest_branch_to_depth(tau_model,
                                                            next_leg)
                    end_action = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, discon_branch,
                        is_p_wave, end_action)
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            # Now deal with P and S case.
            elif current_leg in ("P", "S"):
                if next_leg in ("P", "S", "Pn", "Sn", "END"):
                    if end_action == _ACTIONS["transdown"] or \
                            end_action == _ACTIONS["reflect_underside"]:
                        # Was downgoing, so must first turn in mantle.
                        end_action = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.cmb_branch - 1, is_p_wave,
                                           end_action)
                    if next_leg == 'END':
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           upgoing_rec_branch, is_p_wave,
                                           end_action)
                    else:
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, 0, is_p_wave,
                            end_action)
                elif next_leg[0] == "v":
                    discon_branch = closest_branch_to_depth(tau_model,
                                                            next_leg[1:])
                    if self.current_branch <= discon_branch - 1:
                        end_action = _ACTIONS["reflect_topside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           discon_branch - 1, is_p_wave,
                                           end_action)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "current_branch > discon_branch".format(
                                current_leg, next_leg))
                elif next_leg[0] == "^":
                    discon_branch = closest_branch_to_depth(tau_model,
                                                            next_leg[1:])
                    if prev_leg == "K":
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           discon_branch, is_p_wave,
                                           end_action)
                    elif prev_leg[0] == "^" or prev_leg in ("P", "S", "p", "s",
                                                            "START"):
                        end_action = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.cmb_branch - 1, is_p_wave,
                                           end_action)
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, discon_branch,
                            is_p_wave, end_action)
                    elif ((prev_leg[0] == "v" and
                            discon_branch < closest_branch_to_depth(
                                tau_model, prev_leg[1:]) or
                           (prev_leg == "m" and
                               discon_branch < tau_model.moho_branch) or
                           (prev_leg == "c" and
                               discon_branch < tau_model.cmb_branch))):
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(
                            tau_model, self.current_branch, discon_branch,
                            is_p_wave, end_action)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "current_branch > discon_branch".format(
                                current_leg, next_leg))
                elif next_leg == "c":
                    end_action = _ACTIONS["reflect_topside"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.cmb_branch - 1, is_p_wave, end_action)
                elif next_leg == "K":
                    end_action = _ACTIONS["transdown"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.cmb_branch - 1, is_p_wave, end_action)
                elif next_leg == "m" or (is_next_leg_depth and
                                         next_leg_depth < tau_model.cmb_depth):
                    # Treat the Moho in the same way as 410 type
                    # discontinuities.
                    discon_branch = closest_branch_to_depth(tau_model,
                                                            next_leg)
                    if end_action == _ACTIONS["turn"] \
                            or end_action == _ACTIONS["reflect_topside"] \
                            or end_action == _ACTIONS["transup"]:
                        # Upgoing section
                        if discon_branch > self.current_branch:
                            # Check the discontinuity below the current
                            # branch when the ray should be upgoing
                            raise TauModelError(
                                "Phase not recognised: {} followed by {} when "
                                "current_branch > discon_branch".format(
                                    current_leg, next_leg))
                        end_action = _ACTIONS["transup"]
                        self.add_to_branch(
                            tau_model, self.current_branch, discon_branch,
                            is_p_wave, end_action)
                    else:
                        # Downgoing section, must look at leg after next to
                        # determine whether to convert on the downgoing or
                        # upgoing part of the path.
                        next_next_leg = self.legs[leg_num + 2]
                        if next_next_leg == "p" or next_next_leg == "s":
                            # Convert on upgoing section
                            end_action = _ACTIONS["turn"]
                            self.add_to_branch(
                                tau_model, self.current_branch,
                                tau_model.cmb_branch - 1, is_p_wave,
                                end_action)
                            end_action = _ACTIONS["transup"]
                            self.add_to_branch(tau_model, self.current_branch,
                                               discon_branch, is_p_wave,
                                               end_action)
                        elif next_next_leg == "P" or next_next_leg == "S":
                            if discon_branch > self.current_branch:
                                # discon is below current loc
                                end_action = _ACTIONS["transdown"]
                                self.add_to_branch(
                                    tau_model, self.current_branch,
                                    discon_branch - 1, is_p_wave, end_action)
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
                                                        next_next_leg))
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
                            tau_model.cmb_branch - 1,
                            is_p_wave).min_turn_ray_param >=
                            self.min_ray_param):
                        end_action = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.cmb_branch - 1, is_p_wave,
                                           end_action)
                        self.max_ray_param = self.min_ray_param
                        if next_leg == "END":
                            end_action = _ACTIONS["reflect_underside"]
                            self.add_to_branch(tau_model, self.current_branch,
                                               upgoing_rec_branch, is_p_wave,
                                               end_action)
                        elif next_leg[0] in "PS":
                            end_action = _ACTIONS["reflect_underside"]
                            self.add_to_branch(
                                tau_model, self.current_branch, 0, is_p_wave,
                                end_action)
                    else:
                        # Can't have head wave as ray param is not within
                        # range.
                        self.max_ray_param = -1
                        return
                elif current_leg in ("Pg", "Sg", "Pn", "Sn"):
                    if self.current_branch >= tau_model.moho_branch:
                        # Pg, Pn, Sg and Sn must be above the moho and so is
                        # not valid for rays coming upwards from below,
                        # possibly due to the source depth. Setting
                        # max_ray_param = -1 effectively disallows this phase.
                        self.max_ray_param = -1
                        return
                    if current_leg in ("Pg", "Sg"):
                        end_action = _ACTIONS["turn"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           tau_model.moho_branch - 1,
                                           is_p_wave, end_action)
                        end_action = _ACTIONS["reflect_underside"]
                        self.add_to_branch(tau_model, self.current_branch,
                                           upgoing_rec_branch, is_p_wave,
                                           end_action)
                    elif current_leg in ("Pn", "Sn"):
                        # In the diffracted case we trick addtoBranch into
                        # thinking we are turning below the Moho, but then
                        # make the min_ray_param equal to max_ray_param,
                        # which is the head wave ray.
                        if (self.max_ray_param >= tau_model.get_tau_branch(
                                tau_model.moho_branch,
                                is_p_wave).max_ray_param >=
                                self.min_ray_param):
                            end_action = _ACTIONS["turn"]
                            self.add_to_branch(
                                tau_model, self.current_branch,
                                tau_model.moho_branch, is_p_wave, end_action)
                            end_action = _ACTIONS["transup"]
                            self.add_to_branch(
                                tau_model, self.current_branch,
                                tau_model.moho_branch, is_p_wave, end_action)
                            self.min_ray_param = self.max_ray_param
                            if next_leg == "END":
                                end_action = _ACTIONS["reflect_underside"]
                                self.add_to_branch(
                                    tau_model, self.current_branch,
                                    upgoing_rec_branch, is_p_wave, end_action)
                            elif next_leg[0] in "PS":
                                end_action = _ACTIONS["reflect_underside"]
                                self.add_to_branch(
                                    tau_model, self.current_branch, 0,
                                    is_p_wave, end_action)
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
                    if prev_leg in ("P", "S", "K", "k", "START"):
                        end_action = _ACTIONS["turn"]
                        self.add_to_branch(
                            tau_model, self.current_branch,
                            tau_model.iocb_branch - 1, is_p_wave, end_action)
                    end_action = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.cmb_branch,
                        is_p_wave, end_action)
                elif next_leg == "K":
                    if prev_leg in ("P", "S", "K"):
                        end_action = _ACTIONS["turn"]
                        self.add_to_branch(
                            tau_model, self.current_branch,
                            tau_model.iocb_branch - 1, is_p_wave, end_action)
                    end_action = _ACTIONS["reflect_underside"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.cmb_branch,
                        is_p_wave, end_action)
                elif next_leg in ("I", "J"):
                    end_action = _ACTIONS["transdown"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.iocb_branch - 1, is_p_wave, end_action)
                elif next_leg == "i":
                    end_action = _ACTIONS["reflect_topside"]
                    self.add_to_branch(
                        tau_model, self.current_branch,
                        tau_model.iocb_branch - 1, is_p_wave, end_action)
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            current_leg, next_leg))

            elif current_leg in ("I", "J"):
                end_action = _ACTIONS["turn"]
                self.add_to_branch(
                    tau_model, self.current_branch,
                    tau_model.tau_branches.shape[1] - 1, is_p_wave, end_action)
                if next_leg in ("I", "J"):
                    end_action = _ACTIONS["reflect_underside"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.iocb_branch,
                        is_p_wave, end_action)
                elif next_leg == "K":
                    end_action = _ACTIONS["transup"]
                    self.add_to_branch(
                        tau_model, self.current_branch, tau_model.iocb_branch,
                        is_p_wave, end_action)

            elif current_leg in ("m", "c", "i") or current_leg[0] == "^":
                pass

            elif current_leg[0] == "v":
                b = closest_branch_to_depth(tau_model, current_leg[1:])
                if b == 0:
                    raise TauModelError(
                        "Phase not recognized: {} looks like a top side "
                        "reflection at the free surface.".format(current_leg))

            elif is_leg_depth:
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
            if (end_action == _ACTIONS["reflect_underside"] and
                    downgoing_rec_branch == self.branch_seq[-1]):
                # Last action was upgoing, so last branch should be
                # upgoing_rec_branch
                self.min_ray_param = -1
                self.max_ray_param = -1
            elif (end_action == _ACTIONS["reflect_topside"] and
                    upgoing_rec_branch == self.branch_seq[-1]):
                # Last action was downgoing, so last branch should be
                # downgoing_rec_branch
                self.min_ray_param = -1
                self.max_ray_param = -1

    def phase_conversion(self, tau_model, from_branch, end_action, is_p_to_s):
        """
        Change max_ray_param and min_ray_param where there is a phase
        conversion.

        For instance, SKP needs to change the max_ray_param because there are
        SKS ray parameters that cannot propagate from the CMB into the mantle
        as a P wave.
        """
        if end_action == _ACTIONS["turn"]:
            # Can't phase convert for just a turn point
            raise TauModelError("Bad end_action: phase conversion is not "
                                "allowed at turn points.")
        elif end_action == _ACTIONS["reflect_underside"]:
            self.max_ray_param = \
                min(self.max_ray_param,
                    tau_model.get_tau_branch(from_branch,
                                             is_p_to_s).max_ray_param,
                    tau_model.get_tau_branch(from_branch,
                                             not is_p_to_s).max_ray_param)
        elif end_action == _ACTIONS["reflect_topside"]:
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(from_branch,
                                         is_p_to_s).min_turn_ray_param,
                tau_model.get_tau_branch(from_branch,
                                         not is_p_to_s).min_turn_ray_param)
        elif end_action == _ACTIONS["transup"]:
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(from_branch, is_p_to_s).max_ray_param,
                tau_model.get_tau_branch(from_branch - 1,
                                         not is_p_to_s).min_turn_ray_param)
        elif end_action == _ACTIONS["transdown"]:
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(from_branch, is_p_to_s).min_ray_param,
                tau_model.get_tau_branch(from_branch + 1,
                                         not is_p_to_s).max_ray_param)
        else:
            raise TauModelError("Illegal end_action = {}".format(end_action))

    def add_to_branch(self, tau_model, start_branch, end_branch, is_p_wave,
                      end_action):
        """
        Add branch numbers to branch_seq.

        Branches from start_branch to end_branch, inclusive, are added in
        order. Also, current_branch is set correctly based on the value of
        end_action. end_action can be one of transup, transdown,
        reflect_underside, reflect_topside, or turn.
        """
        if end_branch < 0 or end_branch > tau_model.tau_branches.shape[1]:
            raise ValueError('End branch outside range: %d' % (end_branch, ))

        if end_action == _ACTIONS["turn"]:
            end_offset = 0
            is_down_going = True
            self.min_ray_param = max(
                self.min_ray_param,
                tau_model.get_tau_branch(end_branch,
                                         is_p_wave).min_turn_ray_param)
        elif end_action == _ACTIONS["reflect_underside"]:
            end_offset = 0
            is_down_going = False
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(end_branch, is_p_wave).max_ray_param)
        elif end_action == _ACTIONS["reflect_topside"]:
            end_offset = 0
            is_down_going = True
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(end_branch,
                                         is_p_wave).min_turn_ray_param)
        elif end_action == _ACTIONS["transup"]:
            end_offset = -1
            is_down_going = False
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(end_branch, is_p_wave).max_ray_param)
        elif end_action == _ACTIONS["transdown"]:
            end_offset = 1
            is_down_going = True
            self.max_ray_param = min(
                self.max_ray_param,
                tau_model.get_tau_branch(end_branch, is_p_wave).min_ray_param)
        else:
            raise TauModelError("Illegal end_action: {}".format(end_action))

        if is_down_going:
            if start_branch > end_branch:
                # Can't be downgoing as we are already below.
                self.min_ray_param = -1
                self.max_ray_param = -1
            else:
                # Must be downgoing, so increment i.
                for i in range(start_branch, end_branch + 1):
                    self.branch_seq.append(i)
                    self.down_going.append(is_down_going)
                    self.wave_type.append(is_p_wave)
        else:
            if start_branch < end_branch:
                # Can't be upgoing as we are already above.
                self.min_ray_param = -1
                self.max_ray_param = -1
            else:
                # Upgoing, so decrement i.
                for i in range(start_branch, end_branch - 1, -1):
                    self.branch_seq.append(i)
                    self.down_going.append(is_down_going)
                    self.wave_type.append(is_p_wave)
        self.current_branch = end_branch + end_offset

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
        times_branches = self.calc_branch_mult(tau_model)

        # Sum the branches with the appropriate multiplier.
        size = self.min_ray_param_index - self.max_ray_param_index + 1
        index = slice(self.max_ray_param_index, self.min_ray_param_index + 1)
        for i in range(tau_model.tau_branches.shape[1]):
            tb = times_branches[0, i]
            tbs = times_branches[1, i]
            taub = tau_model.tau_branches[0, i]
            taubs = tau_model.tau_branches[1, i]

            if tb != 0:
                self.dist[:size] += tb * taub.dist[index]
                self.time[:size] += tb * taub.time[index]
            if tbs != 0:
                self.dist[:size] += tbs * taubs.dist[index]
                self.time[:size] += tbs * taubs.time[index]

        if "Sdiff" in self.name or "Pdiff" in self.name:
            if tau_model.s_mod.depth_in_high_slowness(
                    tau_model.cmb_depth - 1e-10, self.min_ray_param,
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
        for is_pwave in [True, False]:
            hsz = tau_model.s_mod.high_slowness_layer_depths_p \
                if is_pwave \
                else tau_model.s_mod.high_slowness_layer_depths_s
            index_offset = 0
            for hszi in hsz:
                if self.max_ray_param > hszi.ray_param > self.min_ray_param:
                    # There is a high slowness zone within our ray parameter
                    # range so might need to add a shadow zone. Need to
                    # check if the current wave type is part of the phase at
                    # this depth/ray parameter.
                    branch_num = tau_model.find_branch(hszi.top_depth)
                    found_overlap = False
                    for leg_num in range(len(self.branch_seq)):
                        # Check for downgoing legs that cross the high
                        # slowness zone with the same wave type.
                        if (self.branch_seq[leg_num] == branch_num and
                                self.wave_type[leg_num] == is_pwave and
                                self.down_going[leg_num] is True and
                                self.branch_seq[leg_num - 1] ==
                                branch_num - 1 and
                                self.wave_type[leg_num - 1] == is_pwave and
                                self.down_going[leg_num - 1] is True):
                            found_overlap = True
                            break
                    if found_overlap:
                        hsz_index = np.where(self.ray_param == hszi.ray_param)
                        hsz_index = hsz_index[0][0]

                        newlen = self.ray_param.shape[0] + 1
                        new_ray_params = np.empty(newlen)
                        newdist = np.empty(newlen)
                        newtime = np.empty(newlen)

                        new_ray_params[:hsz_index] = self.ray_param[:hsz_index]
                        newdist[:hsz_index] = self.dist[:hsz_index]
                        newtime[:hsz_index] = self.time[:hsz_index]

                        # Sum the branches with an appropriate multiplier.
                        new_ray_params[hsz_index] = hszi.ray_param
                        newdist[hsz_index] = 0
                        newtime[hsz_index] = 0
                        for tb, tbs, taub, taubs in zip(
                                times_branches[0], times_branches[1],
                                tau_model.tau_branches[0],
                                tau_model.tau_branches[1]):
                            if tb != 0 and taub.top_depth < hszi.top_depth:
                                newdist[hsz_index] += tb * taub.dist[
                                    self.max_ray_param_index + hsz_index -
                                    index_offset]
                                newtime[hsz_index] += tb * taub.time[
                                    self.max_ray_param_index + hsz_index -
                                    index_offset]
                            if tbs != 0 and taubs.top_depth < hszi.top_depth:
                                newdist[hsz_index] += tbs * taubs.dist[
                                    self.max_ray_param_index + hsz_index -
                                    index_offset]
                                newtime[hsz_index] += tbs * taubs.time[
                                    self.max_ray_param_index + hsz_index -
                                    index_offset]

                        newdist[hsz_index + 1:] = self.dist[hsz_index:]
                        newtime[hsz_index + 1:] = self.time[hsz_index:]
                        new_ray_params[hsz_index + 1:] = \
                            self.ray_param[hsz_index:]

                        index_offset += 1
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
        times_branches = np.zeros((2, tau_model.tau_branches.shape[1]))
        # Count how many times each branch appears in the path.
        # wave_type is at least as long as branch_seq
        for wt, bs in zip(self.wave_type, self.branch_seq):
            if wt:
                times_branches[0][bs] += 1
            else:
                times_branches[1][bs] += 1
        return times_branches

    def calc_time(self, degrees):
        """
        Calculate arrival times for this phase, sorted by time.
        """
        # 100 should finally be enough...this in only for one phase after
        # all..
        r_dist = np.empty(100, dtype=np.float64)
        r_ray_num = np.empty(100, dtype=np.int32)

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

    def calc_pierce_from_arrival(self, curr_arrival):
        """
        Calculate the pierce points for a particular arrival.

        The returned arrival is the same as the input argument but now has the
        pierce points filled in.
        """
        # Find the ray parameter index that corresponds to the arrival ray
        # parameter in the TauModel, ie it is between ray_num and ray_num+1,
        # We know that it must be <model.ray_param.length-1 since the last
        # ray parameter sample is 0 in a spherical model.
        ray_num = 0
        for i, rp in enumerate(self.tau_model.ray_params[:-1]):
            if rp >= curr_arrival.ray_param:
                ray_num = i
            else:
                break

        # Here we use ray parameter and dist info stored within the
        # SeismicPhase so we can use curr_arrival.ray_param_index, which
        # may not correspond to ray_num (for model.ray_param).
        ray_param_a = self.ray_param[curr_arrival.ray_param_index]
        ray_param_b = self.ray_param[curr_arrival.ray_param_index + 1]
        dist_a = self.dist[curr_arrival.ray_param_index]
        dist_b = self.dist[curr_arrival.ray_param_index + 1]
        dist_ratio = (curr_arrival.purist_dist - dist_a) / (dist_b - dist_a)
        dist_ray_param = dist_ratio * (ray_param_b - ray_param_a) + ray_param_a

        # + 2 for first point and kmps, if it exists.
        pierce = np.empty(len(self.branch_seq) + 2, dtype=TimeDist)
        # First pierce point is always 0 distance at the source depth.
        pierce[0] = (dist_ray_param, 0, 0, self.tau_model.source_depth)
        index = 1
        branch_dist = 0
        branch_time = 0

        # Loop from 0 but already done 0 [I just copy the comments, sorry!],
        # so the pierce point when the ray leaves branch i is stored in i + 1.
        # Use linear interpolation between rays that we know.
        assert len(self.branch_seq) == len(self.wave_type) == \
            len(self.down_going)
        for branch_num, is_p_wave, is_down_going in zip(
                self.branch_seq, self.wave_type, self.down_going):
            # Save the turning depths for the ray parameter for both P and
            # S waves. This way we get the depth correct for any rays that
            # turn within a layer. We have to do this on a per branch basis
            # because of converted phases, e.g. SKS.
            tau_branch = self.tau_model.get_tau_branch(branch_num, is_p_wave)
            if dist_ray_param > tau_branch.max_ray_param:
                turn_depth = tau_branch.top_depth
            elif dist_ray_param <= tau_branch.min_ray_param:
                turn_depth = tau_branch.bot_depth
            else:
                if (is_p_wave or self.tau_model.s_mod.depth_in_fluid((
                        tau_branch.top_depth + tau_branch.bot_depth) / 2)):
                    turn_depth = self.tau_model.s_mod.find_depth_from_depths(
                        dist_ray_param,
                        tau_branch.top_depth,
                        tau_branch.bot_depth,
                        True)
                else:
                    turn_depth = self.tau_model.s_mod.find_depth_from_depths(
                        dist_ray_param,
                        tau_branch.top_depth,
                        tau_branch.bot_depth,
                        is_p_wave)

            if any(x in self.name for x in ["Pdiff", "Pn", "Sdiff", "Sn"]):
                # Head waves and diffracted waves are a special case.
                dist_a = tau_branch.dist[ray_num]
                time_a = tau_branch.time[ray_num]
                dist_b, time_b = dist_a, time_a
            else:
                dist_a = tau_branch.dist[ray_num]
                time_a = tau_branch.time[ray_num]
                dist_b = tau_branch.dist[ray_num + 1]
                time_b = tau_branch.time[ray_num + 1]

            branch_dist += dist_ratio * (dist_b - dist_a) + dist_a
            prev_branch_time = np.array(branch_time, copy=True)
            branch_time += dist_ratio * (time_b - time_a) + time_a
            if is_down_going:
                branch_depth = min(tau_branch.bot_depth, turn_depth)
            else:
                branch_depth = min(tau_branch.top_depth, turn_depth)

            # Make sure ray actually propagates in this branch; leave a little
            # room for numerical chatter.
            if abs(prev_branch_time - branch_time) > 1e-10:
                pierce[index] = (dist_ray_param, branch_time, branch_dist,
                                 branch_depth)
                index += 1

        if any(x in self.name for x in ["Pdiff", "Pn", "Sdiff", "Sn"]):
            pierce, index = self.handle_special_waves(curr_arrival,
                                                      pierce,
                                                      index)
        elif "kmps" in self.name:
            pierce[index] = (dist_ray_param, curr_arrival.time,
                             curr_arrival.purist_dist, 0)
            index += 1

        curr_arrival.pierce = pierce[:index]
        # The arrival is modified in place and must (?) thus be returned.
        return curr_arrival

    def calc_path(self, degrees):
        """
        Calculate the paths this phase takes through the planet model.

        Only calls :meth:`calc_path_from_arrival`.
        """
        arrivals = self.calc_time(degrees)
        for arrival in arrivals:
            self.calc_path_from_arrival(arrival)
        return arrivals

    def calc_path_from_arrival(self, curr_arrival):
        """
        Calculate the paths this phase takes through the planet model.
        """
        # Find the ray parameter index that corresponds to the arrival ray
        # parameter in the TauModel, i.e. it is between ray_num and
        # ray_num + 1.
        temp_time_dist = (curr_arrival.ray_param, 0, 0,
                          self.tau_model.source_depth)
        # path_list is a list of lists.
        path_list = [temp_time_dist]
        for i, branch_num, is_p_wave, is_down_going in zip(
                count(), self.branch_seq, self.wave_type, self.down_going):
            br = self.tau_model.get_tau_branch(branch_num, is_p_wave)
            temp_time_dist = br.path(curr_arrival.ray_param, is_down_going,
                                     self.tau_model.s_mod)
            if len(temp_time_dist):
                path_list.extend(temp_time_dist)
                if np.any(temp_time_dist['dist'] < 0):
                    raise RuntimeError("Path is backtracking, "
                                       "this is impossible.")

            # Special case for head and diffracted waves:
            if(branch_num == self.tau_model.cmb_branch - 1 and
               i < len(self.branch_seq) - 1 and
               self.branch_seq[i + 1] == self.tau_model.cmb_branch - 1 and
               ("Pdiff" in self.name or "Sdiff" in self.name)):
                dist_diff = curr_arrival.purist_dist - self.dist[0]
                diff_td = (
                    curr_arrival.ray_param,
                    dist_diff * curr_arrival.ray_param,
                    dist_diff,
                    self.tau_model.cmb_depth)
                path_list.append(diff_td)

            elif(branch_num == self.tau_model.moho_branch and
                 i < len(self.branch_seq) - 1 and
                 self.branch_seq[i + 1] == self.tau_model.moho_branch and
                 ("Pn" in self.name or "Sn" in self.name)):
                # Can't have both Pn and Sn in a wave, so one of these is 0.
                num_found = max(self.name.count("Pn"), self.name.count("Sn"))
                dist_head = (curr_arrival.purist_dist -
                             self.dist[0]) / num_found
                head_td = (
                    curr_arrival.ray_param,
                    dist_head * curr_arrival.ray_param,
                    dist_head,
                    self.tau_model.moho_depth)
                path_list.append(head_td)

        if "kmps" in self.name:
            # kmps phases have no branches, so need to end them at the arrival
            # distance.
            head_td = (
                curr_arrival.ray_param,
                curr_arrival.purist_dist * curr_arrival.ray_param,
                curr_arrival.purist_dist,
                0)
            path_list.append(head_td)

        curr_arrival.path = np.array(path_list, dtype=TimeDist)
        np.cumsum(curr_arrival.path['time'], out=curr_arrival.path['time'])
        np.cumsum(curr_arrival.path['dist'], out=curr_arrival.path['dist'])

        return curr_arrival

    def handle_special_waves(self, curr_arrival, pierce, index):
        """
        Handle head or diffracted waves.

        It is assumed that a phase can be a diffracted wave or a head wave, but
        not both. Nor can it be a head wave or diffracted wave for both P and
        S.
        """
        for ps in ["Pn", "Sn", "Pdiff", "Sdiff"]:
            if ps in self.name:
                phase_seg = ps
                break
        else:
            raise TauModelError("No head/diff segment in" + str(self.name))

        if phase_seg in ["Pn", "Sn"]:
            head_depth = self.tau_model.moho_depth
        else:
            head_depth = self.tau_model.cmb_depth

        num_found = self.name.count(phase_seg)
        refract_dist = curr_arrival.purist_dist - self.dist[0]
        refract_time = refract_dist * curr_arrival.ray_param

        # This is a little weird as we are not checking where we are in
        # the phase name, but simply if the depth matches. This likely
        # works in most cases, but may not for head/diffracted waves that
        # undergo a phase change, if that type of phase can even exist.
        mask = pierce['depth'][:index] == head_depth
        adjust = np.cumsum(mask)
        pierce['time'][:index] += adjust * refract_time / num_found
        pierce['dist'][:index] += adjust * refract_dist / num_found

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
            msg = 'Please contact the developers. This error should not occur.'
            raise RuntimeError(msg) from e

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
        s_mod = tau_model.s_mod

        # counter for passes through each branch. 0 is P and 1 is S.
        times_branches = self.calc_branch_mult(tau_model)
        time = np.zeros(1)
        dist = np.zeros(1)
        ray_param = np.array([ray_param])

        # Sum the branches with the appropriate multiplier.
        for j in range(tau_model.tau_branches.shape[1]):
            if times_branches[0, j] != 0:
                br = tau_model.get_tau_branch(j, s_mod.p_wave)
                top_layer = s_mod.layer_number_below(br.top_depth,
                                                     s_mod.p_wave)
                bot_layer = s_mod.layer_number_above(br.bot_depth,
                                                     s_mod.p_wave)
                td = br.calc_time_dist(s_mod, top_layer, bot_layer, ray_param,
                                       allow_turn_in_layer=True)

                time += times_branches[0, j] * td['time']
                dist += times_branches[0, j] * td['dist']

            if times_branches[1, j] != 0:
                br = tau_model.get_tau_branch(j, s_mod.s_wave)
                top_layer = s_mod.layer_number_below(br.top_depth,
                                                     s_mod.s_wave)
                bot_layer = s_mod.layer_number_above(br.bot_depth,
                                                     s_mod.s_wave)
                td = br.calc_time_dist(s_mod, top_layer, bot_layer, ray_param,
                                       allow_turn_in_layer=True)

                time += times_branches[1, j] * td['time']
                dist += times_branches[1, j] * td['dist']

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
        v_mod = self.tau_model.s_mod.v_mod
        try:
            if self.down_going[0]:
                takeoff_velocity = v_mod.evaluate_below(self.source_depth,
                                                        self.name[0])
            else:
                takeoff_velocity = v_mod.evaluate_above(self.source_depth,
                                                        self.name[0])
        except (IndexError, LookupError) as e:
            msg = 'Please contact the developers. This error should not occur.'
            raise RuntimeError(msg) from e

        return ((self.tau_model.radius_of_planet - self.source_depth) *
                math.sin(np.radians(takeoff_degree)) / takeoff_velocity)

    def calc_takeoff_angle(self, ray_param):
        if self.name.endswith('kmps'):
            return 0

        v_mod = self.tau_model.s_mod.v_mod
        try:
            if self.down_going[0]:
                takeoff_velocity = v_mod.evaluate_below(self.source_depth,
                                                        self.name[0])
            else:
                takeoff_velocity = v_mod.evaluate_above(self.source_depth,
                                                        self.name[0])
        except (IndexError, LookupError) as e:
            msg = 'Please contact the developers. This error should not occur.'
            raise RuntimeError(msg) from e

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

        v_mod = self.tau_model.s_mod.v_mod
        # Very last item is "END", assume first char is P or S
        last_leg = self.legs[-2][0]
        try:
            if self.down_going[-1]:
                incident_velocity = v_mod.evaluate_above(self.receiver_depth,
                                                         last_leg)
            else:
                incident_velocity = v_mod.evaluate_below(self.receiver_depth,
                                                         last_leg)
        except (IndexError, LookupError) as e:
            msg = 'Please contact the developers. This error should not occur.'
            raise RuntimeError(msg) from e

        incident_angle = np.degrees(math.asin(np.clip(
            incident_velocity * ray_param /
            (self.tau_model.radius_of_planet - self.receiver_depth),
            -1.0, 1.0)))
        if self.down_going[-1]:
            incident_angle = 180 - incident_angle

        return incident_angle

    @classmethod
    def get_earliest_arrival(cls, rel_phases, degrees):
        raise NotImplementedError("baaa")


def closest_branch_to_depth(tau_model, depth_string):
    """
    Find the closest discontinuity to the given depth that can have
    reflections and phase transformations.
    """
    if depth_string == "m":
        return tau_model.moho_branch
    elif depth_string == "c":
        return tau_model.cmb_branch
    elif depth_string == "i":
        return tau_model.iocb_branch
    # Non-standard boundary, given by a number: must look for it.
    discon_branch = -1
    discon_max = 1e300
    discon_depth = float(depth_string)
    for i, t_branch in enumerate(tau_model.tau_branches[0]):
        if (abs(discon_depth - t_branch.top_depth) < discon_max and not
                any(ndc == t_branch.top_depth
                    for ndc in tau_model.no_discon_depths)):
            discon_branch = i
            discon_max = abs(discon_depth - t_branch.top_depth)
    return discon_branch


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
