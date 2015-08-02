#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Objects and functions dealing with seismic phases.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from itertools import count
import math
import re

import numpy as np

from .helper_classes import (Arrival, SlownessModelError, TauModelError,
                             TimeDist)

from .c_wrappers import clibtau


class SeismicPhase(object):
    """
    Stores and transforms seismic phase names to and from their
    corresponding sequence of branches. Will maybe contain "expert" mode
    wherein paths may start in the core. Principal use is to calculate leg
    contributions for scattered phases. Nomenclature: "K" - downgoing wave
    from source in core; "k" - upgoing wave from source in core.
    """
    DEBUG = False
    # Enables phases originating in core.
    expert = False
    # Used by addToBranch when the path turns within a segment. We assume
    # that no ray will turn downward so turning implies turning from
    # downward to upward, ie U.
    TURN = 0
    # Used by addToBranch when the path reflects off the top of the end of a
    # segment, ie ^.
    REFLECT_UNDERSIDE = 1
    # Used by addToBranch when the path reflects off the bottom of the end
    # of a segment, ie v.
    REFLECT_TOPSIDE = 2
    # Used by addToBranch when the path transmits up through the end of a
    # segment.
    TRANSUP = 3
    # Used by addToBranch when the path transmits down through the end of a
    # segment.
    TRANSDOWN = 4
    # The maximum degrees that a Pn or Sn can refract along the moho. Note
    # this is not the total distance, only the segment along the moho. The
    # default is 20 degrees.
    maxRefraction = 20
    # The maximum degrees that a Pdiff or Sdiff can diffract along the CMB.
    # Note this is not the total distance, only the segment along the CMB.
    # The default is 60 degrees.
    maxDiffraction = 60

    def __init__(self, name, tMod, receiver_depth=0.0):
        # Minimum/maximum ray parameters that exist for this phase.
        self.minRayParam = None
        self.maxRayParam = None
        # Index within TauModel.ray_param that corresponds to maxRayParam.
        # Note that maxRayParamIndex < minRayParamIndex as ray parameter
        # decreases with increasing index.
        self.maxRayParamIndex = -1
        # Index within TauModel.ray_param that corresponds to minRayParam.
        # Note that maxRayParamIndex < minRayParamIndex as ray parameter
        # decreases with increasing index.
        self.minRayParamIndex = -1
        # Temporary branch numbers determining where to start adding to the
        # branch sequence.
        self.currBranch = None
        # The phase name, e.g. PKiKP.
        self.name = name
        # The source depth within the TauModel that was used to generate
        # this phase.
        self.source_depth = tMod.source_depth
        # The receiver depth within the TauModel that was used to generate this
        # phase. Normally this is 0.0 for a surface station, but can be
        # different for borehole or scattering calculations.
        self.receiver_depth = receiver_depth
        # TauModel to generate phase for.
        self.tMod = tMod
        # Array of distances corresponding to the ray parameters stored in
        # ray_param.
        self.dist = None
        # Array of times corresponding to the ray parameters stored in
        # ray_param.
        self.time = None
        # Array of possible ray parameters for this phase.
        self.ray_param = None
        # The minimum distance that this phase can be theoretically observed.
        self.minDistance = 0.0
        # The maximum distance that this phase can be theoretically observed.
        self.maxDistance = 1e300
        # List (could make array!) of branch numbers for the given phase.
        # Note that this depends upon both the earth model and the source
        # depth.
        self.branchSeq = []
        # True if the current leg of the phase is down going. This allows a
        # check to make sure the path is correct.
        # Used in addToBranch() and parseName().
        self.downGoing = []
        # ArrayList of wave types corresponding to each leg of the phase.
        self.waveType = []
        # List containing strings for each leg.
        self.legs = leg_puller(name)
        # Name with depths corrected to be actual discontinuities in the model.
        self.puristName = self.create_purist_name(tMod)
        self.parse_name(tMod)
        self.sum_branches(tMod)

    def create_purist_name(self, tMod):
        currLeg = self.legs[0]
        # Deal with surface wave veocities first, since they are a special
        # case.
        if len(self.legs) == 2 and currLeg.endswith("kmps"):
            puristName = self.name
            return puristName
        puristName = ""
        # Only loop to penultimate element as last leg is always "END".
        for currLeg in self.legs[:-1]:
            # Find out if the next leg represents a phase conversion or
            # reflection depth.
            if currLeg[0] in "v^":
                disconBranch = closest_branch_to_depth(tMod, currLeg[1:])
                legDepth = tMod.tauBranches[0, disconBranch].topDepth
                puristName += currLeg[0]
                puristName += str(int(round(legDepth)))
            else:
                try:
                    float(currLeg)
                except ValueError:
                    # If currLeg is just a string:
                    puristName += currLeg
                else:
                    # If it is indeed a number:
                    disconBranch = closest_branch_to_depth(tMod, currLeg)
                    legDepth = tMod.tauBranches[0, disconBranch].topDepth
                    puristName += str(legDepth)
        return puristName

    def parse_name(self, tMod):
        """
        Construct a branch sequence from the given phase name and tau model.
        """
        currLeg = self.legs[0]
        nextLeg = currLeg
        isPWave = True
        isPWavePrev = isPWave
        endAction = self.TRANSDOWN

        # Deal with surface wave velocities first, since they are a special
        # case.
        if len(self.legs) == 2 and currLeg.endswith("kmps"):
            return

        # Make a check for J legs if the model doesn't allow J:
        if "J" in self.name and not tMod.sMod.allowInnerCoreS:
            raise TauModelError("J phases are not created for this model: {}"
                                .format(self.name))

        # Set currWave to be the wave type for this leg, P or S
        if currLeg in ("p", "K", "k", "I") or currLeg[0] == "P":
            isPWave = True
            isPWavePrev = isPWave
        elif currLeg in ("s", "J") or currLeg[0] == "S":
            isPWave = False
            isPWavePrev = isPWave
        else:
            raise TauModelError('Unknown starting phase: ' + currLeg)

        # First, decide whether the ray is upgoing or downgoing from the
        # source. If it is up going then the first branch number would be
        # model.sourceBranch-1 and downgoing would be model.sourceBranch.
        upgoingRecBranch = tMod.findBranch(self.receiver_depth)
        downgoingRecBranch = upgoingRecBranch - 1  # One branch shallower.
        if currLeg[0] in "sS":
            # Exclude S sources in fluids.
            sdep = tMod.source_depth
            if tMod.cmbDepth < sdep < tMod.iocbDepth:
                self.maxRayParam, self.minRayParam = -1, -1
                return

        # Set self.maxRayParam to be a horizontal ray leaving the source and
        # self.minRayParam to be a vertical (p=0) ray.
        if currLeg[0] in "PS" or (self.expert and currLeg[0] in "KIJ"):
            # Downgoing from source.
            self.currBranch = tMod.sourceBranch
            # Treat initial downgoing as if it were an underside reflection.
            endAction = self.REFLECT_UNDERSIDE
            try:
                sLayerNum = tMod.sMod.layerNumberBelow(tMod.source_depth,
                                                       isPWavePrev)
                layer = tMod.sMod.getSlownessLayer(sLayerNum, isPWavePrev)
                self.maxRayParam = layer['topP']
            except SlownessModelError as e:
                raise RuntimeError('Should not happen' + str(e))
            self.maxRayParam = tMod.getTauBranch(tMod.sourceBranch,
                                                 isPWave).maxRayParam
        elif currLeg in ("p", "s") or (self.expert and currLeg[0] == "k"):
            # Upgoing from source: treat initial downgoing as if it were a
            # topside reflection.
            endAction = self.REFLECT_TOPSIDE
            try:
                sLayerNum = tMod.sMod.layerNumberAbove(tMod.source_depth,
                                                       isPWavePrev)
                layer = tMod.sMod.getSlownessLayer(sLayerNum, isPWavePrev)
                self.maxRayParam = layer['botP']
            except SlownessModelError as e:
                raise RuntimeError('Should not happen' + str(e))
            if tMod.sourceBranch != 0:
                self.currBranch = tMod.sourceBranch - 1
            else:
                # p and s for zero source depth are only at zero distance
                # and then can be called P or S.
                self.maxRayParam = -1
                self.minRayParam = -1
                return
        else:
            raise TauModelError(
                'First phase not recognised {}: Must be one of P, Pg, Pn, '
                'Pdiff, p, Ped or the S equivalents.'.format(currLeg))
        if self.receiver_depth != 0:
            if self.legs[-2] in ('Ped', 'Sed'):
                # Downgoing at receiver
                self.maxRayParam = min(
                    tMod.getTauBranch(downgoingRecBranch,
                                      isPWave).minTurnRayParam,
                    self.maxRayParam)
            else:
                # upgoing at receiver
                self.maxRayParam = min(
                    tMod.getTauBranch(upgoingRecBranch,
                                      isPWave).minTurnRayParam,
                    self.maxRayParam)

        self.minRayParam = 0

        isLegDepth, isNextLegDepth = False, False

        # Now loop over all the phase legs and construct the proper branch
        # sequence.
        currLeg = "START"  # So the prevLeg isn't wrong on the first pass.
        for legNum in range(len(self.legs) - 1):
            prevLeg = currLeg
            currLeg = nextLeg
            nextLeg = self.legs[legNum + 1]
            isLegDepth = isNextLegDepth

            # Find out if the next leg represents a phase conversion depth.
            try:
                nextLegDepth = float(nextLeg)
                isNextLegDepth = True
            except ValueError:
                nextLegDepth = -1
                isNextLegDepth = False

            # Set currWave to be the wave type for this leg, "P" or "S".
            isPWavePrev = isPWave
            if currLeg in ("p", "k", "I") or currLeg[0] == "P":
                isPWave = True
            elif currLeg in ("s", "J") or currLeg[0] == "S":
                isPWave = False
            elif currLeg == "K":
                # Here we want to use whatever isPWave was on the last leg
                # so do nothing. This makes sure we use the correct
                # maxRayParam from the correct TauBranch within the outer
                # core. In other words K has a high slowness zone if it
                # entered the outer core as a mantle P wave, but doesn't if
                # it entered as a mantle S wave. It shouldn't matter for
                # inner core to outer core type legs.
                pass

            # Check to see if there has been a phase conversion.
            if len(self.branchSeq) > 0 and isPWavePrev != isPWave:
                self.phase_conversion(tMod, self.branchSeq[-1], endAction,
                                      isPWavePrev)

            if currLeg in ('Ped', 'Sed'):
                if nextLeg == "END":
                    if receiverDepth > 0:
                        endAction = REFLECT_TOPSIDE
                        self.add_to_branch(tMod, self.currBranch,
                                           downgoingRecBranch, isPWave,
                                           endAction)
                    else:
                        # This should be impossible except for 0 dist 0 source
                        # depth which can be called p or P.
                        self.maxRayParam = -1
                        self.minRayParam = -1
                        return
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            currLeg, nextLeg))

            # Deal with p and s case.
            elif currLeg in ("p", "s", "k"):
                if nextLeg[0] == "v":
                    raise TauModelError(
                        "p and s must always be upgoing and cannot come "
                        "immediately before a top-sided reflection.")
                elif nextLeg.startswith("^"):
                    disconBranch = closest_branch_to_depth(tMod, nextLeg[1:])
                    if self.currBranch >= disconBranch:
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch, disconBranch,
                                           isPWave, endAction)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "currBranch > disconBranch".format(currLeg,
                                                               nextLeg))
                elif nextLeg == "m" and self.currBranch >= tMod.mohoBranch:
                    endAction = self.TRANSUP
                    self.add_to_branch(tMod, self.currBranch, tMod.mohoBranch,
                                       isPWave, endAction)
                elif nextLeg[0] in ("P", "S") or nextLeg in ("K", "END"):
                    if nextLeg == 'END':
                        disconBranch = upgoingRecBranch
                    elif nextLeg == 'K':
                        disconBranch = tMod.cmbBranch
                    else:
                        disconBranch = 0
                    if currLeg == 'k' and nextLeg != 'K':
                        endAction = self.TRANSUP
                    else:
                        endAction = self.REFLECT_UNDERSIDE
                    self.add_to_branch(tMod, self.currBranch, disconBranch,
                                       isPWave, endAction)
                elif isNextLegDepth:
                    disconBranch = closest_branch_to_depth(tMod, nextLeg)
                    endAction = self.TRANSUP
                    self.add_to_branch(tMod, self.currBranch, disconBranch,
                                       isPWave, endAction)
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            currLeg, nextLeg))

            # Now deal with P and S case.
            elif currLeg in ("P", "S"):
                if nextLeg in ("P", "S", "Pn", "Sn", "END"):
                    if endAction == self.TRANSDOWN or \
                            endAction == self.REFLECT_UNDERSIDE:
                        # Was downgoing, so must first turn in mantle.
                        endAction = self.TURN
                        self.add_to_branch(tMod, self.currBranch,
                                           tMod.cmbBranch - 1, isPWave,
                                           endAction)
                    if nextLeg == 'END':
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch,
                                           upgoingRecBranch, isPWave,
                                           endAction)
                    else:
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch, 0, isPWave,
                                           endAction)
                elif nextLeg[0] == "v":
                    disconBranch = closest_branch_to_depth(tMod, nextLeg[1:])
                    if self.currBranch <= disconBranch - 1:
                        endAction = self.REFLECT_TOPSIDE
                        self.add_to_branch(tMod, self.currBranch,
                                           disconBranch - 1, isPWave,
                                           endAction)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "currBranch > disconBranch".format(currLeg,
                                                               nextLeg))
                elif nextLeg[0] == "^":
                    disconBranch = closest_branch_to_depth(tMod, nextLeg[1:])
                    if prevLeg == "K":
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch, disconBranch,
                                           isPWave, endAction)
                    elif prevLeg[0] == "^" or prevLeg in ("P", "S", "p", "s",
                                                          "START"):
                        endAction = self.TURN
                        self.add_to_branch(tMod, self.currBranch,
                                           tMod.cmbBranch - 1, isPWave,
                                           endAction)
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch, disconBranch,
                                           isPWave, endAction)
                    elif ((prevLeg[0] == "v" and
                            disconBranch < closest_branch_to_depth(
                                tMod, prevLeg[1:]) or
                           (prevLeg == "m" and
                               disconBranch < tMod.mohoBranch) or
                           (prevLeg == "c" and
                               disconBranch < tMod.cmbBranch))):
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch, disconBranch,
                                           isPWave, endAction)
                    else:
                        raise TauModelError(
                            "Phase not recognised: {} followed by {} when "
                            "currBranch > disconBranch".format(currLeg,
                                                               nextLeg))
                elif nextLeg == "c":
                    endAction = self.REFLECT_TOPSIDE
                    self.add_to_branch(tMod, self.currBranch,
                                       tMod.cmbBranch - 1, isPWave, endAction)
                elif nextLeg == "K":
                    endAction = self.TRANSDOWN
                    self.add_to_branch(tMod, self.currBranch,
                                       tMod.cmbBranch - 1, isPWave, endAction)
                elif nextLeg == "m" or (isNextLegDepth and
                                        nextLegDepth < tMod.cmbDepth):
                    # Treat the Moho in the same way as 410 type
                    # discontinuities.
                    disconBranch = closest_branch_to_depth(tMod, nextLeg)
                    if endAction == self.TURN \
                            or endAction == self.REFLECT_TOPSIDE \
                            or endAction == self.TRANSUP:
                        # Upgoing section
                        if disconBranch > self.currBranch:
                            # Check the discontinuity below the current
                            # branch when the ray should be upgoing
                            raise TauModelError(
                                "Phase not recognised: {} followed by {} when "
                                "currBranch > disconBranch".format(currLeg,
                                                                   nextLeg))
                        endAction = self.TRANSUP
                        self.add_to_branch(tMod, self.currBranch, disconBranch,
                                           isPWave, endAction)
                    else:
                        # Downgoing section, must look at leg after next to
                        # determine whether to convert on the downgoing or
                        # upgoing part of the path.
                        nextnextLeg = self.legs[legNum + 2]
                        if nextnextLeg == "p" or nextnextLeg == "s":
                            # Convert on upgoing section
                            endAction = self.TURN
                            self.add_to_branch(tMod, self.currBranch,
                                               tMod.cmbBranch - 1, isPWave,
                                               endAction)
                            endAction = self.TRANSUP
                            self.add_to_branch(tMod, self.currBranch,
                                               disconBranch, isPWave,
                                               endAction)
                        elif nextnextLeg == "P" or nextnextLeg == "S":
                            if disconBranch > self.currBranch:
                                # discon is below current loc
                                endAction = self.TRANSDOWN
                                self.add_to_branch(tMod, self.currBranch,
                                                   disconBranch - 1, isPWave,
                                                   endAction)
                            else:
                                # Discontinuity is above current location,
                                # but we have a downgoing ray, so this is an
                                # illegal ray for this source depth.
                                self.maxRayParam = -1
                                return
                        else:
                            raise TauModelError(
                                "Phase not recognized: {} followed by {} "
                                "followed by {}".format(currLeg, nextLeg,
                                                        nextnextLeg))
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            currLeg, nextLeg))

            elif currLeg[0] in "PS":
                if currLeg == "Pdiff" or currLeg == "Sdiff":
                    # In the diffracted case we trick addtoBranch into
                    # thinking we are turning, but then make maxRayParam
                    # equal to minRayParam, which is the deepest turning ray.
                    if (self.maxRayParam >= tMod.getTauBranch(
                            tMod.cmbBranch - 1, isPWave).minTurnRayParam >=
                            self.minRayParam):
                        endAction = self.TURN
                        self.add_to_branch(tMod, self.currBranch,
                                           tMod.cmbBranch - 1, isPWave,
                                           endAction)
                        self.maxRayParam = self.minRayParam
                        if nextLeg == "END":
                            endAction = self.REFLECT_UNDERSIDE
                            self.add_to_branch(tMod, self.currBranch,
                                               upgoingRecBranch, isPWave,
                                               endAction)
                        elif nextLeg[0] in "PS":
                            endAction = self.REFLECT_UNDERSIDE
                            self.add_to_branch(tMod, self.currBranch, 0,
                                               isPWave, endAction)
                    else:
                        # Can't have head wave as ray param is not within
                        # range.
                        self.maxRayParam = -1
                        return
                elif currLeg in ("Pg", "Sg", "Pn", "Sn"):
                    if self.currBranch >= tMod.mohoBranch:
                        # Pg, Pn, Sg and Sn must be above the moho and so is
                        # not valid for rays coming upwards from below,
                        # possibly due to the source depth. Setting
                        # maxRayParam = -1 effectively disallows this phase.
                        self.maxRayParam = -1
                        return
                    if currLeg in ("Pg", "Sg"):
                        endAction = self.TURN
                        self.add_to_branch(tMod, self.currBranch,
                                           tMod.mohoBranch - 1, isPWave,
                                           endAction)
                        endAction = self.REFLECT_UNDERSIDE
                        self.add_to_branch(tMod, self.currBranch,
                                           upgoingRecBranch, isPWave,
                                           endAction)
                    elif currLeg in ("Pn", "Sn"):
                        # In the diffracted case we trick addtoBranch into
                        # thinking we are turning below the Moho, but then
                        # make the minRayParam equal to maxRayParam,
                        # which is the head wave ray.
                        if (self.maxRayParam >= tMod.getTauBranch(
                                tMod.mohoBranch, isPWave).maxRayParam >=
                                self.minRayParam):
                            endAction = self.TURN
                            self.add_to_branch(tMod, self.currBranch,
                                               tMod.mohoBranch, isPWave,
                                               endAction)
                            endAction = self.TRANSUP
                            self.add_to_branch(tMod, self.currBranch,
                                               tMod.mohoBranch, isPWave,
                                               endAction)
                            self.minRayParam = self.maxRayParam
                            if nextLeg == "END":
                                endAction = self.REFLECT_UNDERSIDE
                                self.add_to_branch(tMod, self.currBranch,
                                                   upgoingRecBranch, isPWave,
                                                   endAction)
                            elif nextLeg[0] in "PS":
                                endAction = self.REFLECT_UNDERSIDE
                                self.add_to_branch(tMod, self.currBranch, 0,
                                                   isPWave, endAction)
                        else:
                            # Can't have head wave as ray param is not
                            # within range.
                            self.maxRayParam = -1
                            return
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            currLeg, nextLeg))

            elif currLeg == "K":
                if nextLeg in ("P", "S"):
                    if prevLeg in ("P", "S", "K", "k", "START"):
                        endAction = self.TURN
                        self.add_to_branch(tMod, self.currBranch,
                                           tMod.iocbBranch - 1, isPWave,
                                           endAction)
                    endAction = self.TRANSUP
                    self.add_to_branch(tMod, self.currBranch, tMod.cmbBranch,
                                       isPWave, endAction)
                elif nextLeg == "K":
                    if prevLeg in ("P", "S", "K"):
                        endAction = self.TURN
                        self.add_to_branch(tMod, self.currBranch,
                                           tMod.iocbBranch - 1, isPWave,
                                           endAction)
                    endAction = self.REFLECT_UNDERSIDE
                    self.add_to_branch(tMod, self.currBranch, tMod.cmbBranch,
                                       isPWave, endAction)
                elif nextLeg in ("I", "J"):
                    endAction = self.TRANSDOWN
                    self.add_to_branch(tMod, self.currBranch,
                                       tMod.iocbBranch - 1, isPWave, endAction)
                elif nextLeg == "i":
                    endAction = self.REFLECT_TOPSIDE
                    self.add_to_branch(tMod, self.currBranch,
                                       tMod.iocbBranch - 1, isPWave, endAction)
                else:
                    raise TauModelError(
                        "Phase not recognized: {} followed by {}".format(
                            currLeg, nextLeg))

            elif currLeg in ("I", "J"):
                endAction = self.TURN
                self.add_to_branch(tMod, self.currBranch,
                                   tMod.tauBranches.shape[1] - 1, isPWave,
                                   endAction)
                if nextLeg in ("I", "J"):
                    endAction = self.REFLECT_UNDERSIDE
                    self.add_to_branch(tMod, self.currBranch, tMod.iocbBranch,
                                       isPWave, endAction)
                elif nextLeg == "K":
                    endAction = self.TRANSUP
                    self.add_to_branch(tMod, self.currBranch, tMod.iocbBranch,
                                       isPWave, endAction)

            elif currLeg in ("m", "c", "i") or currLeg[0] == "^":
                pass

            elif currLeg[0] == "v":
                b = closest_branch_to_depth(tMod, currLeg[1:])
                if b == 0:
                    raise TauModelError(
                        "Phase not recognized: {} looks like a top side "
                        "reflection at the free surface.".format(currLeg))

            elif isLegDepth:
                # Check for phase like P0s, but could also be P2s if first
                # discontinuity is deeper.
                b = closest_branch_to_depth(tMod, currLeg)
                if b == 0 and nextLeg in ("p", "s"):
                    raise TauModelError(
                        "Phase not recognized: {} followed by {} looks like "
                        "an upgoing wave from the free surface as closest "
                        "discontinuity to {} is zero depth.".format(currLeg,
                                                                    nextLeg,
                                                                    currLeg))

            else:
                raise TauModelError(
                    "Phase not recognized: {} followed by {}".format(currLeg,
                                                                     nextLeg))

        if self.maxRayParam != -1:
            if (endAction == self.REFLECT_UNDERSIDE and
                    downgoingRecBranch == self.branchSeq[-1]):
                # Last action was upgoing, so last branch should be
                # upgoingRecBranch
                self.minRayParam = -1
                self.maxRayParam = -1
            elif (endAction == self.REFLECT_TOPSIDE and
                    upgoingRecBranch == self.branchSeq[-1]):
                # Last action was downgoing, so last branch should be
                # downgoingRecBranch
                self.minRayParam = -1
                self.maxRayParam = -1

    def phase_conversion(self, tMod, fromBranch, endAction, isPtoS):
        """
        Change maxRayParam and minRayParam where there is a phase conversion.

        For instance, SKP needs to change the maxRayParam because there are SKS
        ray parameters that cannot propagate from the CMB into the mantle as a
        P wave.
        """
        if endAction == self.TURN:
            # Can't phase convert for just a turn point
            raise TauModelError("Bad endAction: phase conversion is not "
                                "allowed at turn points.")
        elif endAction == self.REFLECT_UNDERSIDE:
            self.maxRayParam = \
                min(self.maxRayParam,
                    tMod.getTauBranch(fromBranch, isPtoS).maxRayParam,
                    tMod.getTauBranch(fromBranch, not isPtoS).maxRayParam)
        elif endAction == self.REFLECT_TOPSIDE:
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(fromBranch, isPtoS).minTurnRayParam,
                tMod.getTauBranch(fromBranch, not isPtoS).minTurnRayParam)
        elif endAction == self.TRANSUP:
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(fromBranch, isPtoS).maxRayParam,
                tMod.getTauBranch(fromBranch - 1, not isPtoS).minTurnRayParam)
        elif endAction == self.TRANSDOWN:
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(fromBranch, isPtoS).minRayParam,
                tMod.getTauBranch(fromBranch + 1, not isPtoS).maxRayParam)
        else:
            raise TauModelError("Illegal endAction = {}".format(endAction))

    def add_to_branch(self, tMod, startBranch, endBranch, isPWave, endAction):
        """
        Add branch numbers to branchSeq.

        Branches from startBranch to endBranch, inclusive, are added in order.
        Also, currBranch is set correctly based on the value of endAction.
        endAction can be one of TRANSUP, TRANSDOWN, REFLECT_UNDERSIDE,
        REFLECT_TOPSIDE, or TURN.
        """
        if endBranch < 0 or endBranch > tMod.tauBranches.shape[1]:
            raise ValueError('End branch outside range: %d' % (endBranch, ))

        if endAction == self.TURN:
            endOffset = 0
            isDownGoing = True
            self.minRayParam = max(
                self.minRayParam,
                tMod.getTauBranch(endBranch, isPWave).minTurnRayParam)
        elif endAction == self.REFLECT_UNDERSIDE:
            endOffset = 0
            isDownGoing = False
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(endBranch, isPWave).maxRayParam)
        elif endAction == self.REFLECT_TOPSIDE:
            endOffset = 0
            isDownGoing = True
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(endBranch, isPWave).minTurnRayParam)
        elif endAction == self.TRANSUP:
            endOffset = -1
            isDownGoing = False
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(endBranch, isPWave).maxRayParam)
        elif endAction == self.TRANSDOWN:
            endOffset = 1
            isDownGoing = True
            self.maxRayParam = min(
                self.maxRayParam,
                tMod.getTauBranch(endBranch, isPWave).minRayParam)
        else:
            raise TauModelError("Illegal endAction: {}".format(endAction))

        if isDownGoing:
            if startBranch > endBranch:
                # Can't be downgoing as we are already below.
                self.minRayParam = -1
                self.maxRayParam = -1
            else:
                # Must be downgoing, so increment i.
                for i in range(startBranch, endBranch + 1):
                    self.branchSeq.append(i)
                    self.downGoing.append(isDownGoing)
                    self.waveType.append(isPWave)
        else:
            if startBranch < endBranch:
                # Can't be upgoing as we are already above.
                self.minRayParam = -1
                self.maxRayParam = -1
            else:
                # Upgoing, so decrement i.
                for i in range(startBranch, endBranch - 1, -1):
                    self.branchSeq.append(i)
                    self.downGoing.append(isDownGoing)
                    self.waveType.append(isPWave)
        self.currBranch = endBranch + endOffset

    def sum_branches(self, tMod):
        """Sum the appropriate branches for this phase."""
        # Special case for surface waves.
        if self.name.endswith("kmps"):
            self.dist = np.zeros(2)
            self.time = np.zeros(2)
            self.ray_param = np.empty(2)

            self.ray_param[0] = tMod.radiusOfEarth / float(self.name[:-4])

            self.dist[1] = 2 * math.pi
            self.time[1] = \
                2 * math.pi * tMod.radiusOfEarth / float(self.name[:-4])
            self.ray_param[1] = self.ray_param[0]

            self.minDistance = 0
            self.maxDistance = 2 * math.pi
            self.downGoing.append(True)
            return

        if self.maxRayParam < 0 or self.minRayParam > self.maxRayParam:
            # Phase has no arrivals, possibly due to source depth.
            self.ray_param = np.empty(0)
            self.minRayParam = -1
            self.maxRayParam = -1
            self.dist = np.empty(0)
            self.time = np.empty(0)
            self.maxDistance = -1
            return

        # Find the ray parameter index that corresponds to the minRayParam
        # and maxRayParam.
        index = np.where(tMod.ray_params >= self.minRayParam)[0]
        if len(index):
            self.minRayParamIndex = index[-1]
        index = np.where(tMod.ray_params >= self.maxRayParam)[0]
        if len(index):
            self.maxRayParamIndex = index[-1]
        if self.maxRayParamIndex == 0 \
                and self.minRayParamIndex == len(tMod.ray_params) - 1:
            # All ray parameters are valid so just copy:
            self.ray_param = tMod.ray_param.copy()
        elif self.maxRayParamIndex == self.minRayParamIndex:
            # if "Sdiff" in self.name or "Pdiff" in self.name:
            # self.ray_param = [self.minRayParam, self.minRayParam]
            # elif "Pn" in self.name or "Sn" in self.name:
            # self.ray_param = [self.minRayParam, self.minRayParam]
            if self.name.endswith("kmps"):
                self.ray_param = np.array([0, self.maxRayParam])
            else:
                self.ray_param = np.array([self.minRayParam, self.minRayParam])
        else:
            # Only a subset of the ray parameters is valid so use these.
            self.ray_param = \
                tMod.ray_params[self.maxRayParamIndex:
                                self.minRayParamIndex + 1].copy()

        self.dist = np.zeros(shape=self.ray_param.shape)
        self.time = np.zeros(shape=self.ray_param.shape)

        # Initialise the counter for each branch to 0. 0 is P and 1 is S.
        timesBranches = np.zeros((2, tMod.tauBranches.shape[1]))
        # Count how many times each branch appears in the path.
        # waveType is at least as long as branchSeq
        for wt, bs in zip(self.waveType, self.branchSeq):
            if wt:
                timesBranches[0][bs] += 1
            else:
                timesBranches[1][bs] += 1

        # Sum the branches with the appropriate multiplier.
        size = self.minRayParamIndex - self.maxRayParamIndex + 1
        index = slice(self.maxRayParamIndex, self.minRayParamIndex + 1)
        for i in range(tMod.tauBranches.shape[1]):
            tb = timesBranches[0, i]
            tbs = timesBranches[1, i]
            taub = tMod.tauBranches[0, i]
            taubs = tMod.tauBranches[1, i]

            if tb != 0:
                self.dist[:size] += tb * taub.dist[index]
                self.time[:size] += tb * taub.time[index]
            if tbs != 0:
                self.dist[:size] += tbs * taubs.dist[index]
                self.time[:size] += tbs * taubs.time[index]

        if "Sdiff" in self.name or "Pdiff" in self.name:
            if tMod.sMod.depthInHighSlowness(tMod.cmbDepth - 1e-10,
                                             self.minRayParam,
                                             self.name[0] == "P"):
                # No diffraction if there is a high slowness zone at the CMB.
                self.minRayParam = -1
                self.maxRayParam = -1
                self.maxDistance = -1
                self.time = np.empty(0)
                self.dist = np.empty(0)
                self.ray_param = np.empty(0)
                return
            else:
                self.dist[1] = self.dist[0] + np.radians(self.maxDiffraction)
                self.time[1] = self.time[0] + \
                    np.radians(self.maxDiffraction) * self.minRayParam

        elif "Pn" in self.name or "Sn" in self.name:
            self.dist[1] = self.dist[0] + np.radians(self.maxRefraction)
            self.time[1] = (self.time[0] +
                            np.radians(self.maxRefraction) * self.minRayParam)

        elif self.maxRayParamIndex == self.minRayParamIndex:
            self.dist[1] = self.dist[0]
            self.time[1] = self.time[0]

        self.minDistance = np.min(self.dist)
        self.maxDistance = np.max(self.dist)

        # Now check to see if our ray parameter range includes any ray
        # parameters that are associated with high slowness zones. If so,
        # then we will need to insert a "shadow zone" into our time and
        # distance arrays. It is represented by a repeated ray parameter.
        for isPwave in [True, False]:
            hsz = tMod.sMod.highSlownessLayerDepthsP \
                if isPwave \
                else tMod.sMod.highSlownessLayerDepthsS
            indexOffset = 0
            for hszi in hsz:
                if self.maxRayParam > hszi.ray_param > self.minRayParam:
                    # There is a high slowness zone within our ray parameter
                    # range so might need to add a shadow zone. Need to
                    # check if the current wave type is part of the phase at
                    # this depth/ray parameter.
                    branchNum = tMod.findBranch(hszi.topDepth)
                    foundOverlap = False
                    for legNum in range(len(self.branchSeq)):
                        # Check for downgoing legs that cross the high
                        # slowness zone with the same wave type.
                        if (self.branchSeq[legNum] == branchNum and
                                self.waveType[legNum] == isPwave and
                                self.downGoing[legNum] is True and
                                self.branchSeq[legNum - 1] == branchNum - 1 and
                                self.waveType[legNum - 1] == isPwave and
                                self.downGoing[legNum - 1] is True):
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
                        for tb, tbs, taub, taubs in zip(timesBranches[0],
                                                        timesBranches[1],
                                                        tMod.tauBranches[0],
                                                        tMod.tauBranches[1]):
                            if tb != 0 and taub.topDepth < hszi.topDepth:
                                newdist[hszIndex] += tb * taub.dist[
                                    self.maxRayParamIndex + hszIndex -
                                    indexOffset]
                                newtime[hszIndex] += tb * taub.time[
                                    self.maxRayParamIndex + hszIndex -
                                    indexOffset]
                            if tbs != 0 and taubs.topDepth < hszi.topDepth:
                                newdist[hszIndex] += tbs * taubs.dist[
                                    self.maxRayParamIndex + hszIndex -
                                    indexOffset]
                                newtime[hszIndex] += tbs * taubs.time[
                                    self.maxRayParamIndex + hszIndex -
                                    indexOffset]

                        newdist[hszIndex + 1:] = self.dist[hszIndex:]
                        newtime[hszIndex + 1:] = self.time[hszIndex:]
                        new_ray_params[hszIndex + 1:] = \
                            self.ray_param[hszIndex:]

                        indexOffset += 1
                        self.dist = newdist
                        self.time = newtime
                        self.ray_param = new_ray_params

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
            self.maxDistance,
            self.dist,
            self.ray_param,
            r_dist,
            r_ray_num,
            len(self.dist)
        )

        arrivals = []
        for _i in range(phase_count):
            arrivals.append(self.linear_interp_arrival(
                degrees, r_dist[_i], r_ray_num[_i]))
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
        for i, rp in enumerate(self.tMod.ray_params[:-1]):
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
        pierce = np.empty(len(self.branchSeq) + 2, dtype=TimeDist)
        # First pierce point is always 0 distance at the source depth.
        pierce[0] = (distRayParam, 0, 0, self.tMod.source_depth)
        index = 1
        branchDist = 0
        branchTime = 0

        # Loop from 0 but already done 0 [I just copy the comments, sorry!],
        # so the pierce point when the ray leaves branch i is stored in i + 1.
        # Use linear interpolation between rays that we know.
        assert len(self.branchSeq) == len(self.waveType) == len(self.downGoing)
        for branchNum, isPWave, isDownGoing in zip(self.branchSeq,
                                                   self.waveType,
                                                   self.downGoing):
            # Save the turning depths for the ray parameter for both P and
            # S waves. This way we get the depth correct for any rays that
            # turn within a layer. We have to do this on a per branch basis
            # because of converted phases, e.g. SKS.
            tauBranch = self.tMod.getTauBranch(branchNum, isPWave)
            if distRayParam > tauBranch.maxRayParam:
                turnDepth = tauBranch.topDepth
            elif distRayParam <= tauBranch.minRayParam:
                turnDepth = tauBranch.botDepth
            else:
                if (isPWave or self.tMod.sMod.depthInFluid((
                        tauBranch.topDepth + tauBranch.botDepth) / 2)):
                    turnDepth = self.tMod.sMod.findDepth_from_depths(
                        distRayParam,
                        tauBranch.topDepth,
                        tauBranch.botDepth,
                        True)
                else:
                    turnDepth = self.tMod.sMod.findDepth_from_depths(
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
        Calculate the paths this phase takes through the Earth model.

        Only calls :meth:`calc_path_from_arrival`.
        """
        arrivals = self.calc_time(degrees)
        for arrival in arrivals:
            self.calc_path_from_arrival(arrival)
        return arrivals

    def calc_path_from_arrival(self, currArrival):
        """
        Calculate the paths this phase takes through the Earth model.
        """
        # Find the ray parameter index that corresponds to the arrival ray
        # parameter in the TauModel, i.e. it is between rayNum and rayNum + 1.
        tempTimeDist = np.array([(currArrival.ray_param, 0, 0,
                                  self.tMod.source_depth)], dtype=TimeDist)
        # pathList is a list of lists.
        pathList = [tempTimeDist]
        for i, branchNum, isPWave, isDownGoing in zip(count(), self.branchSeq,
                                                      self.waveType,
                                                      self.downGoing):
            tempTimeDist = self.tMod.getTauBranch(branchNum, isPWave)\
                .path(currArrival.ray_param, isDownGoing, self.tMod.sMod)
            if len(tempTimeDist):
                pathList.append(tempTimeDist)
                if np.any(tempTimeDist['dist'] < 0):
                    raise RuntimeError("Path is backtracking, "
                                       "this is impossible.")

            # Special case for head and diffracted waves:
            if(branchNum == self.tMod.cmbBranch - 1 and
               i < len(self.branchSeq) - 1 and
               self.branchSeq[i + 1] == self.tMod.cmbBranch - 1 and
               ("Pdiff" in self.name or "Sdiff" in self.name)):
                dist_diff = currArrival.purist_dist - self.dist[0]
                diffTD = np.array([(
                    currArrival.ray_param,
                    dist_diff * currArrival.ray_param,
                    dist_diff,
                    self.tMod.cmbDepth)], dtype=TimeDist)
                pathList.append(diffTD)

            elif(branchNum == self.tMod.mohoBranch - 1 and
                 i < len(self.branchSeq) - 1 and
                 self.branchSeq[i + 1] == self.tMod.mohoBranch - 1 and
                 ("Pn" in self.name or "Sn" in self.name)):
                # Can't have both Pn and Sn in a wave, so one of these is 0.
                numFound = max(self.name.count("Pn"), self.name.count("Sn"))
                dist_head = (currArrival.purist_dist - self.dist[0]) / numFound
                headTD = np.array([(
                    currArrival.ray_param,
                    dist_head * currArrival.ray_param,
                    dist_head,
                    self.tMod.mohoDepth)], dtype=TimeDist)
                pathList.append(headTD)

        if "kmps" in self.name:
            # kmps phases have no branches, so need to end them at the arrival
            # distance.
            headTD = np.array([(
                currArrival.ray_param,
                currArrival.purist_dist * currArrival.ray_param,
                currArrival.purist_dist,
                0)], dtype=TimeDist)
            pathList.append(headTD)

        currArrival.path = np.concatenate(pathList)
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
            headDepth = self.tMod.mohoDepth
        else:
            headDepth = self.tMod.cmbDepth

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

    def linear_interp_arrival(self, degrees, searchDist, rayNum):
        if rayNum == 0 and searchDist == self.dist[0]:
            # degenerate case
            return Arrival(self, degrees, self.time[0], searchDist,
                           self.ray_param[0], rayNum, self.name,
                           self.puristName, self.source_depth, 0, 0)

        arrivalTime = ((searchDist - self.dist[rayNum]) /
                       (self.dist[rayNum + 1] - self.dist[rayNum]) *
                       (self.time[rayNum + 1] - self.time[rayNum]) +
                       self.time[rayNum])
        arrivalRayParam = ((searchDist - self.dist[rayNum + 1]) *
                           (self.ray_param[rayNum] -
                            self.ray_param[rayNum + 1]) /
                           (self.dist[rayNum] - self.dist[rayNum + 1]) +
                           self.ray_param[rayNum + 1])
        if self.name.endswith("kmps"):
            takeoffAngle = 0
            incidentAngle = 0
        else:
            vMod = self.tMod.sMod.vMod
            if self.downGoing[0]:
                takeoffVelocity = vMod.evaluateBelow(self.source_depth,
                                                     self.name[0])
            else:
                # Fake negative velocity so angle is negative in case of
                # upgoing ray.
                takeoffVelocity = -1 * vMod.evaluateAbove(self.source_depth,
                                                          self.name[0])
            takeoffAngle = np.degrees(math.asin(np.clip(
                takeoffVelocity * arrivalRayParam /
                (self.tMod.radiusOfEarth - self.source_depth), -1.0, 1.0)))

            lastLeg = self.legs[-2][0]  # very last item is "END"
            if self.downGoing[-1]:
                # Fake negative velocity so angle is negative in case of
                # downgoing ray.
                incident_velocity = -1 * vMod.evaluateAbove(
                    self.receiver_depth,
                    lastLeg)
            else:
                incident_velocity = vMod.evaluateBelow(self.receiver_depth,
                                                       lastLeg)
            incidentAngle = np.degrees(math.asin(
                incident_velocity * arrivalRayParam /
                (self.tMod.radiusOfEarth - self.receiver_depth)))

        return Arrival(self, degrees, arrivalTime, searchDist, arrivalRayParam,
                       rayNum, self.name, self.puristName, self.source_depth,
                       takeoffAngle, incidentAngle)

    @classmethod
    def get_earliest_arrival(cls, relPhases, degrees):
        raise NotImplementedError("baaa")


def closest_branch_to_depth(tMod, depthString):
    """
    Find the closest discontinuity to the given depth that can have
    reflections and phase transformations.
    """
    if depthString == "m":
        return tMod.mohoBranch
    elif depthString == "c":
        return tMod.cmbBranch
    elif depthString == "i":
        return tMod.iocbBranch
    # Non-standard boundary, given by a number: must look for it.
    disconBranch = -1
    disconMax = 1e300
    disconDepth = float(depthString)
    for i, tBranch in enumerate(tMod.tauBranches[0]):
        if (abs(disconDepth - tBranch.topDepth) < disconMax and not
                any(ndc == tBranch.topDepth for ndc in tMod.noDisconDepths)):
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
