from .Arrival import Arrival
from .helper_classes import TauModelError
import math
from copy import deepcopy


class SeismicPhase(object):
    """Stores and transforms seismic phase names to and from their corresponding sequence of branches.
    Will maybe contain "expert" mode wherein paths may start in the core. Principal use is to calculate leg
    contributions for scattered phases. Nomenclature: "K" - downgoing wave from source in core;
     "k" - upgoing wave from source in core.
    """

    DEBUG = False
    # Enables phases originating in core.
    expert = False
    #  Used by addToBranch when the path turns within a segment. We assume that no ray will turn
    # downward so turning implies turning from downward to upward, ie U.
    TURN = 0
    # Used by addToBranch when the path reflects off the top of the end of a segment, ie ^.
    REFLECTTOP = 1
    # Used by addToBranch when the path reflects off the bottom of the end of a segment, ie v.
    REFLECTBOT = 2
    # Used by addToBranch when the path transmits up through the end of a segment.
    TRANSUP = 3
    # Used by addToBranch when the path transmits down through the end of a segment.
    TRANSDOWN = 4
    # The maximum degrees that a Pn or Sn can refract along the moho. Note this is not
    # the total distance, only the segment along the moho. The default is 20 degrees.
    maxRefraction = 20
    # The maximum degrees that a Pdiff or Sdiff can diffract along the CMB. Note this is
    # not the total distance, only the segment along the CMB. The default is 60 degrees.
    maxDiffraction = 60

    def __init__(self, name, tMod):
        # Minimum/maximum ray parameters that exist for this phase.
        self.minRayParam = None
        self.maxRayParam = None
        # Index within TauModel.rayParams that corresponds to maxRayParam. Note that
        # maxRayParamIndex < minRayParamIndex as ray parameter decreases with increasing index.
        self.maxRayParamIndex = -1
        # Index within TauModel.rayParams that corresponds to minRayParam. Note that
        # maxRayParamIndex < minRayParamIndex as ray parameter decreases with increasing index.
        self.minRayParamIndex = -1
        # Temporary branch numbers determining where to start adding to the branch sequence.
        self.currBranch = None
        # Temporary end action so we know what we did at the end of the last section of the branch sequence.
        # Used in addToBranch() and parseName().
        self.endAction = None
        # The phase name, e.g. PKiKP.
        self.name = name
        # The source depth within the TauModel that was used to generate this phase.
        self.sourceDepth = tMod.sourceDepth
        # TauModel to generate phase for.
        self.tMod = tMod
        # Array of distances corresponding to the ray parameters stored in rayParams.
        self.dist = []
        # Array of times corresponding to the ray parameters stored in rayParams.
        self.time = []
        # Array of possible ray parameters for this phase.
        self.rayParams = []
        # The minimum distance that this phase can be theoretically observed.
        self.minDistance = 0.0
        # The maximum distance that this phase can be theoretically observed.
        self.maxDistance = 1e300
        # List (could make array!) of branch numbers for the given phase. Note that this
        # depends upon both the earth model and the source depth.
        self.branchSeq = []
        # Temporary end action so we know what we did at the end of the last section of the branch sequence.
        # Used in addToBranch() and parseName().
        # endAction
        # Records the end action for the current leg. Will be one of SeismicPhase.TURN,
        # SeismicPhase.TRANSDOWN, SeismicPhase.TRANSUP, SeismicPhase.REFLECTBOT, or SeismicPhase.REFLECTTOP.
        # This allows a check to make sure the path is correct. Used in addToBranch() and parseName().
        self.legAction = []
        # True if the current leg of the phase is down going. This allows a check to make sure the path is correct.
        #  Used in addToBranch() and parseName().
        self.downGoing = []
        # ArrayList of wave types corresponding to each leg of the phase.
        self.waveType = []
        # List containing strings for each leg.
        self.legs = legPuller(name)
        # Name with depths corrected to be actual discontinuities in the model.
        self.puristName = self.createPuristName(tMod)
        self.parseName(tMod)
        self.sumBranches(tMod)

    def createPuristName(self, tMod):
        currLeg = self.legs[0]
        # Deal with surface wave veocities first, since they are a special case.
        if len(self.legs) == 2 and currLeg.endswith("kmps"):
            puristName = self.name
            return puristName
        puristName = ""
        # Only loop to penultimate element as last leg is always "END".
        for currLeg in self.legs[:-1]:
            # Find out if the next leg represents a phase conversion or reflection depth.
            if currLeg.startswith("v") or currLeg.startswith("^"):
                disconBranch = closestBranchToDepth(tMod, currLeg[1])
                legDepth = tMod.tauBranches[0][disconBranch].topDepth
                puristName += currLeg[0]
                puristName += str(legDepth)
            else:
                try:
                    float(currLeg)
                    # If it is indeed a number:
                    disconBranch = closestBranchToDepth(tMod, currLeg)
                    legDepth = tMod.tauBranches[0][disconBranch].topDepth
                    puristName += str(legDepth)
                except ValueError:
                    # If currLeg is just a string:
                    puristName += currLeg
        return puristName

    def parseName(self, tMod):
        """Constructs a branch sequence from the given phase name and tau model."""
        currLeg = self.legs[0]
        nextLeg = currLeg
        isPWave = True
        # Deal with surface wave velocities first, since they are a special case.
        if len(self.legs) == 2 and currLeg.endswith("kmps"):
            return
        # Make a check for J legs if the model doesn't allow J:
        if "J" in self.name and not tMod.sMod.allowInnerCoreS:
            raise TauModelError("J phases are not created for this model: {}".format(self.name))
        # Set currWave to be the wave type for this leg, P or S
        if currLeg == "p" or currLeg.startswith("P") or currLeg == "K" or currLeg == "k":
            isPWave = True
        elif currLeg == "s" or currLeg.startswith("S") or currLeg == "J":
            isPWave = False
        # First,  decide whether the ray is upgoing or downgoing from the source. If it is
        # up going then the first branch number would be tMod.sourceBranch-1 and downgoing
        # would be tMod.sourceBranch.
        if currLeg.startswith("s") or currLeg.startswith("S"):
            # Exclude S sources in fluids.
            sdep = tMod.sourceDepth
            if tMod.cmbDepth < sdep < tMod.iocbDepth:
                self.maxRayParam, self.minRayParam = -1, -1
                return
        if currLeg.startswith("P") or currLeg.startswith("S") or (self.expert and
                                                                  any(currLeg.startswith(a) for a in ("K", "I"))):
            # Downgoing from source.
            self.currBranch = tMod.sourceBranch
            # Treat initial downgoing as if it were an underside reflection.
            self.endAction = self.REFLECTBOT
        elif currLeg == "p" or currLeg == "s" or (self.expert and currLeg.startswith("k")):
            # Upgoing from source: treat initial downgoing as if it were a topside reflection.
            self.endAction = self.REFLECTTOP
            if tMod.sourceBranch != 0:
                self.currBranch = tMod.sourceBranch - 1
            else:
                # p and s for zero source depth are only at zero distance and then can be called P or S.
                self.maxRayParam = -1
                self.minRayParam = -1
                return
        else:
            raise TauModelError("First phase not recognised {}: ".format(currLeg) +
                                "Must be one of P, Pg, Pn, Pdiff, p or the S equivalents.")
        # Set maxRayParam to be a horizontal ray leaving the source and set minRayParam to be a vertical (p=0) ray.
        if tMod.sourceBranch != 0:
            self.maxRayParam = max(tMod.getTauBranch(tMod.sourceBranch - 1, isPWave).minTurnRayParam,
                                   tMod.getTauBranch(tMod.sourceBranch, isPWave).maxRayParam)
        else:
            self.maxRayParam = tMod.getTauBranch(tMod.sourceBranch, isPWave).maxRayParam
        self.minRayParam = 0
        isLegDepth, isNextLegDepth = False, False
        self.endAction = self.TRANSDOWN
        # Now loop over all the phase legs and construct the proper branch sequence.
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
            if currLeg == "p" or currLeg.startswith("P") or currLeg == "k" or currLeg == "I":
                isPWave = True
            elif currLeg == "s" or currLeg.startswith("S") or currLeg == "J":
                isPWave = False
            elif currLeg == "K":
                # Here we want to use whatever isPWave was on the last leg so do nothing. This makes sure we us the
                # correct maxRayParam from the correct TauBranch within the outer core. In other words K has a high
                # slowness zone if it entered the outer core as a mantle P wave, but doesn't if it entered as a mantle
                # S wave. It shouldn't matter for inner core to outer core type legs.
                pass
            # Check to see if there has been a phase conversion.
            if len(self.branchSeq) > 0 and isPWavePrev != isPWave:
                self.phaseConversion(tMod, int(self.branchSeq[-1]), self.endAction, isPWavePrev)
            # Deal with p and s case first.
            if currLeg == "p" or currLeg == "s" or currLeg == "k":
                if nextLeg.startswith("v"):
                    raise TauModelError("p and s must always be upgoing and cannot come immediately before"
                                        "a top-sided reflection.")
                elif nextLeg.startswith("^"):
                    disconBranch = closestBranchToDepth(tMod, nextLeg[1])
                    if self.currBranch >= disconBranch:
                        self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.REFLECTTOP)
                    else:
                        raise TauModelError("Phase not recognised: "
                                            "{currLeg} followed by {nextLeg} when currBranch > disconBranch".format(**locals()))
                elif nextLeg == "m" and self.currBranch >= tMod.mohoBranch:
                    self.addToBranch(tMod, self.currBranch, tMod.mohoBranch, isPWave, self.TRANSUP)
                elif nextLeg.startswith("P") or nextLeg.startswith("S") or nextLeg == "K" or nextLeg == "END":
                    disconBranch = tMod.cmbBranch if nextLeg == "K" else 0
                    self.addToBranch(tMod, self.currBranch, disconBranch, isPWave,
                                     (self.TRANSUP if currLeg == "k" and not nextLeg == "K" else self.REFLECTTOP))
                elif isNextLegDepth:
                    disconBranch = closestBranchToDepth(tMod, nextLeg)
                    self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.TRANSUP)
                else:
                    raise TauModelError("Phase not recognized: {} followed by {}".format(currLeg, nextLeg))
            # Now deal with P and S case.
            elif currLeg == "P" or currLeg == "S":
                if any(nextLeg == c for c in ("P", "S", "Pn", "Sn", "END")):
                    if self.endAction == self.TRANSDOWN or self.endAction == self.REFLECTTOP:
                        # Downgoing, so must first turn in mantle.
                        self.addToBranch(tMod, self.currBranch, tMod.cmbBranch - 1, isPWave, self.TURN)
                    self.addToBranch(tMod, self.currBranch, 0, isPWave, self.REFLECTTOP)
                elif nextLeg.startswith("v"):
                    disconBranch = closestBranchToDepth(tMod, nextLeg[1])
                    if self.currBranch <= disconBranch - 1:
                        self.addToBranch(tMod, self.currBranch, disconBranch - 1, isPWave, self.REFLECTBOT)
                    else:
                        raise TauModelError("Phase not recognised: "
                                            "{currLeg} followed by {nextLeg} when currBranch > disconBranch".format(**locals()))
                elif nextLeg.startswith("^"):
                    disconBranch = closestBranchToDepth(tMod, nextLeg[1])
                    if prevLeg == "K":
                        self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.REFLECTTOP)
                    elif prevLeg.startswith("^") or any(prevLeg == c for c in ("P", "S", "p", "s", "START")):
                        self.addToBranch(tMod, self.currBranch, tMod.cmbBranch - 1, isPWave, self.TURN)
                        self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.REFLECTTOP)
                    elif ((prevLeg.startswith("v") and disconBranch < closestBranchToDepth(tMod, prevLeg[1])
                          or (prevLeg == "m" and disconBranch < tMod.mohoBranch)
                          or (prevLeg == "c" and disconBranch < tMod.cmbBranch))):
                        self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.REFLECTTOP)
                    else:
                        raise TauModelError("Phase not recognised: "
                                            "{currLeg} followed by {nextLeg} when currBranch > disconBranch".format(**locals()))
                elif nextLeg == "c":
                    self.addToBranch(tMod, self.currBranch, tMod.cmbBranch - 1, isPWave, self.REFLECTBOT)
                elif nextLeg == "K":
                    self.addToBranch(tMod, self.currBranch, tMod.cmbBranch - 1, isPWave, self.TRANSDOWN)
                elif nextLeg == "m" or (isNextLegDepth and nextLegDepth < tMod.cmbDepth):
                    # Treat the Moho in the same way as 410 type discontinuities.
                    disconBranch = closestBranchToDepth(tMod, nextLeg)
                    if self.endAction == self.TURN or self.endAction == self.REFLECTBOT or self.endAction == self.TRANSUP:
                        # Upgoing section
                        if disconBranch > self.currBranch:
                            # Check the discontinuity below the current branch when the ray should be upgoing
                            raise TauModelError("Phase not recognised: "
                                                "{currLeg} followed by {nextLeg} when currBranch > disconBranch".format(**locals()))
                        self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.TRANSUP)
                    else:
                        # Downgoing section, must look at leg after next to determine whether to convert
                        # on the downgoing or upgoing part of the path.
                        nextnextLeg = self.legs[legNum+2]
                        if nextnextLeg == "p" or nextnextLeg == "s":
                            # Convert on upgoing section
                            self.addToBranch(tMod, self.currBranch, tMod.cmbBranch - 1, isPWave, self.TURN)
                            self.addToBranch(tMod, self.currBranch, disconBranch, isPWave, self.TRANSUP)
                        elif nextnextLeg == "P" or nextnextLeg == "S":
                            if disconBranch > self.currBranch:
                                # discon is below current loc
                                self.addToBranch(tMod, self.currBranch, disconBranch - 1, isPWave, self.TRANSDOWN)
                            else:
                                # Discontinuity is above current location, but we have a downgoing ray, so this is
                                # an illegal ray for this source depth.
                                self.maxRayParam = -1
                                return
                        else:
                            raise TauModelError("Phase not recognized: {} followed by {} followed by {}".format(currLeg, nextLeg, nextnextLeg))
                else:
                    raise TauModelError("Phase not recognized: {} followed by {}".format(currLeg, nextLeg))
            elif currLeg.startswith("P") or currLeg.startswith("S"):
                if currLeg == "Pdiff" or currLeg == "Sdiff":
                    # In the diffracted case we trick addtoBranch into thinking we are turning, but then make maxRayParam
                    # equal to minRayParam, which is the deepest turning ray.
                    if (self.maxRayParam >= tMod.getTauBranch(tMod.cmbBranch - 1, isPWave).minTurnRayParam
                            >= self.minRayParam):
                        self.addToBranch(tMod, self.currBranch, tMod.cmbBranch - 1, isPWave, self.TURN)
                        self.maxRayParam = self.minRayParam
                        if nextLeg == "END" or nextLeg.startswith("P") or nextLeg.startswith("S"):
                            self.addToBranch(tMod, self.currBranch, 0, isPWave, self.REFLECTTOP)
                    else:
                        # Can't have head wave as ray param is not within range.
                        self.maxRayParam = -1
                        return
                elif any(currLeg == p for p in ("Pg", "Sg", "Pn", "Sn")):
                    if self.currBranch >= tMod.mohoBranch:
                        # Pg, Pn, Sg and Sn must be above the moho and so is not valid for rays coming upwards
                        # from below, possibly due to the source depth. Setting maxRayParam = -1 effectively
                        # disallows this phase.
                        self.maxRayParam = -1
                        return
                    if currLeg == "Pg" or currLeg == "Sg":
                        self.addToBranch(tMod, self.currBranch, tMod.mohoBranch - 1, isPWave, self.TURN)
                        self.addToBranch(tMod, self.currBranch, 0, isPWave, self.REFLECTTOP)
                    elif currLeg == "Pn" or currLeg == "Sn":
                        # In the diffracted case we trick addtoBranch into thinking we are turning below the Moho,
                        # but then make the minRayParam equal to maxRayParam, which is the head wave ray.
                        if (self.maxRayParam >= tMod.getTauBranch(tMod.mohoBranch, isPWave).maxRayParam
                                >= self.minRayParam):
                            self.addToBranch(tMod, self.currBranch, tMod.mohoBranch, isPWave, self.TURN)
                            self.addToBranch(tMod, self.currBranch, tMod.mohoBranch, isPWave, self.TRANSUP)
                            self.minRayParam = self.maxRayParam
                            if nextLeg == "END" or nextLeg.startswith("P") or nextLeg.startswith("S"):
                                self.addToBranch(tMod, self.currBranch, 0, isPWave, self.REFLECTTOP)
                        else:
                            # Can't have head wave as ray param is not within range.
                            self.maxRayParam = -1
                            return
                else:
                    raise TauModelError("Phase not recognized: {} followed by {}".format(currLeg, nextLeg))
            elif currLeg == "K":
                if nextLeg == "P" or nextLeg == "S":
                    if any(prevLeg == p for p in ("P", "S", "K", "k", "START")):
                        self.addToBranch(tMod, self.currBranch, tMod.iocbBranch - 1, isPWave, self.TURN)
                    self.addToBranch(tMod, self.currBranch, tMod.cmbBranch, isPWave, self.TRANSUP)
                elif nextLeg == "K":
                    if any(prevLeg == p for p in ("P", "S", "K")):
                        self.addToBranch(tMod, self.currBranch, tMod.iocbBranch -1, isPWave, self.TURN)
                    self.addToBranch(tMod, self.currBranch, tMod.cmbBranch, isPWave, self.REFLECTTOP)
                elif nextLeg == "I" or nextLeg == "J":
                    self.addToBranch(tMod, self.currBranch, tMod.iocbBranch -1, isPWave, self.TRANSDOWN)
                elif nextLeg == "i":
                    self.addToBranch(tMod, self.currBranch, tMod.iocbBranch - 1, isPWave, self.REFLECTBOT)
                else:
                    raise TauModelError("Phase not recognized: {} followed by {}".format(currLeg, nextLeg))
            elif currLeg == "I" or currLeg == "J":
                self.addToBranch(tMod, self.currBranch, len(tMod.tauBranches[0]), isPWave, self.TURN)
                if nextLeg == "I" or nextLeg == "J":
                    self.addToBranch(tMod, self.currBranch, tMod.iocbBranch, isPWave, self.REFLECTTOP)
                elif nextLeg == "K":
                    self.addToBranch(tMod, self.currBranch, tMod.iocbBranch, isPWave, self.TRANSUP)
            elif (any(currLeg == p for p in ("m", "c", "i"))
                  or currLeg.startswith == "^" or currLeg.startswith == "v" or isLegDepth):
                pass
            else:
                raise TauModelError("Phase not recognized: {} followed by {}".format(currLeg, nextLeg))

    def phaseConversion(self, tMod, fromBranch, endAction, isPtoS):
        """ Changes maxRayParam and minRayParam whenever there is a phase conversion.
        For instance, SKP needs to change the maxRayParam because there are SKS
        ray parameters that cannot propagate from the cmb into the mantle as a p wave.
        """
        if endAction == self.TURN:
            # Can't phase convert for just a turn point
            raise TauModelError("Bad endAction: phase conversion is not allowed at turn points.")
        elif endAction == self.REFLECTTOP:
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(fromBranch, isPtoS).maxRayParam,
                                   tMod.getTauBranch(fromBranch, not isPtoS).maxRayParam)
        elif endAction == self.REFLECTBOT:
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(fromBranch, isPtoS).minTurnRayParam,
                                   tMod.getTauBranch(fromBranch, not isPtoS).minTurnRayParam)
        elif endAction == self.TRANSUP:
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(fromBranch, isPtoS).maxRayParam,
                                   tMod.getTauBranch(fromBranch - 1, not isPtoS).minTurnRayParam)
        elif endAction == self.TRANSDOWN:
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(fromBranch, isPtoS).minRayParam,
                                   tMod.getTauBranch(fromBranch + 1, not isPtoS).maxRayParam)
        else:
            raise TauModelError("Illegal endAction = {}".format(endAction))

    def addToBranch(self, tMod, startBranch, endBranch, isPWave, endAction):
        """Adds the branch numbers from startBranch to endBranch, inclusive, to branchSeq, in order. Also, currBranch
        is set correctly based on the value of endAction. endAction can be one of TRANSUP, TRANSDOWN, REFLECTTOP,
        REFLECTBOT, or TURN."""
        self.endAction = endAction
        if endAction == self.TURN:
            endOffset = 0
            isDownGoing = True
            self.minRayParam = max(self.minRayParam, tMod.getTauBranch(endBranch, isPWave).minTurnRayParam)
        elif endAction == self.REFLECTTOP:
            endOffset = 0
            isDownGoing = False
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(endBranch, isPWave).maxRayParam)
        elif endAction == self.REFLECTBOT:
            endOffset = 0
            isDownGoing = True
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(endBranch, isPWave).minTurnRayParam)
        elif endAction == self.TRANSUP:
            endOffset = -1
            isDownGoing = False
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(endBranch, isPWave).maxRayParam)
        elif endAction == self.TRANSDOWN:
            endOffset = 1
            isDownGoing = True
            self.maxRayParam = min(self.maxRayParam, tMod.getTauBranch(endBranch, isPWave).minRayParam)
        else:
            raise TauModelError("Illegal endAction: {}".format(endAction))
        if isDownGoing:
            # Must be downgoing, so increment i.
            for i in range(startBranch, endBranch + 1):
                self.branchSeq.append(i)
                self.downGoing.append(isDownGoing)
                self.waveType.append(isPWave)
                self.legAction.append(endAction)
        else:
            # Upgoing, so decrement i.
            for i in range(startBranch, endBranch - 1, -1):
                self.branchSeq.append(i)
                self.downGoing.append(isDownGoing)
                self.waveType.append(isPWave)
                self.legAction.append(endAction)
        self.currBranch = endBranch + endOffset

    def sumBranches(self, tMod):
        """Sum the appropriate branches for this phase."""
        if self.maxRayParam < 0 or self.minRayParam > self.maxRayParam:
            # Phase has no arrivals, possibly due to source depth.
            self.rayParams = []
            self.minRayParam = -1
            self.maxRayParam = -1
            self.dist = []
            self.time = []
            self.maxDistance = -1
            return
        # Special case for surface waves.
        if self.name.endswith("kmps"):
            self.dist = [0, 0]
            self.time = [0, 0]
            self.rayParams = [0, 0]
            self.rayParams[0] = tMod.radiusOfEarth / float(self.name[:-4])
            self.dist[1] = 2 * math.pi
            self.time[1] = 2 * math.pi * tMod.radiusOfEarth / float(self.name[:-4])
            self.rayParams[1] = self.rayParams[0]
            self.minDistance = 0
            self.maxDistance = 2 * math.pi
            self.downGoing.append(True)
            return
        # Find the ray parameter index that corresponds to the minRayParam and maxRayParam.
        for i, rp in enumerate(tMod.rayParams):
            if rp >= self.minRayParam:
                self.minRayParamIndex = i
            if rp >= self.maxRayParam:
                self.maxRayParamIndex = i
        if self.maxRayParamIndex == 0 and self.minRayParamIndex == len(tMod.rayParams) -1:
            # All ray parameters are valid so just copy:
            self.rayParams = deepcopy(tMod.rayParams)
        elif self.maxRayParamIndex == self.minRayParamIndex:
            #if "Sdiff" in self.name or "Pdiff" in self.name:
            #    self.rayParams = [self.minRayParam, self.minRayParam]
            #elif "Pn" in self.name or "Sn" in self.name:
            #    self.rayParams = [self.minRayParam, self.minRayParam]
            if self.name.endswith("kmps"):
                self.rayParams = [0, self.maxRayParam]
            else:
                self.rayParams = [self.minRayParam, self.minRayParam]
        else:
            # Only a subset of the ray parameters is valid so use these.
            self.rayParams = deepcopy(tMod.rayParams[self.maxRayParamIndex:self.minRayParamIndex + 1])
        self.dist = [0 for i in range(len(self.rayParams))]
        self.time = [0 for i in range(len(self.rayParams))]
        # Initialise the counter for each branch to 0. 0 is P and 1 is S.
        timesBranches = [[0 for i in range(len(tMod.tauBranches[0]))] for j in range(2)]
        # Count how many times each branch appears in the path.
        for wt, bs in zip(self.waveType, self.branchSeq):  # waveType is at least as long as branchSeq
            if wt:
                timesBranches[0][bs] += 1
            else:
                timesBranches[1][bs] += 1
        # Sum the branches with the appropriate multiplier.
        for tb, tbs, taub, taubs in zip(timesBranches[0], timesBranches[1], tMod.tauBranches[0], tMod.tauBranches[1]):
            if tb != 0:
                for i in range(self.maxRayParamIndex, self.minRayParamIndex + 1):
                    self.dist[i - self.maxRayParamIndex] += tb * taub.dist[i]
                    self.time[i - self.maxRayParamIndex] += tb * taub.time[i]
            if tbs != 0:
                for i in range(self.maxRayParamIndex, self.minRayParamIndex + 1):
                    self.dist[i - self.maxRayParamIndex] += tbs * taubs.dist[i]
                    self.time[i - self.maxRayParamIndex] += tbs * taubs.time[i]
        if "Sdiff" in self.name or "Pdiff" in self.name:
            if tMod.sMod.depthInHighSlowness(tMod.cmbDepth - 1e-10, self.minRayParam, self.name[0] == "P"):
                # No diffraction if there is a high slowness zone at the CMB.
                self.minRayParam = -1
                self.maxRayParam = -1
                self.maxDistance = -1
                self.time = []
                self.dist = []
                self.rayParams = []
                return
            else:
                self.dist[1] = self.dist[0] + self.maxDiffraction * math.pi / 180
                self.time[1] = self.time[0] + self.maxDiffraction * math.pi / 180 * self.minRayParam
        elif "Pn" in self.name or "Sn" in self.name:
            self.dist[1] = self.dist[0] + self.maxRefraction * math.pi / 180
            self.time[1] = self.time[0] + self.maxRefraction * math.pi / 180
        elif self.maxRayParamIndex == self.minRayParamIndex:
            self.dist[1] = self.dist[0]
            self.time[1] = self.time[0]
        self.minDistance = min(self.dist)
        self.maxDistance = max(self.dist)
        # Now check to see if our ray parameter range inclides any ray parameters that are associated with high
        # slowness zones. If so, then we will need to insert a "shadow zone" into our time and distance arrays.
        # It is represented by a repeated ray parameter.
        for isPwave in [True, False]:
            hsz = tMod.sMod.highSlownessLayerDepthsP if isPwave else tMod.sMod.highSlownessLayerDepthsS
            indexOffset = 0
            for hszi in hsz:
                if self.maxRayParam > hszi.rayParam > self.minRayParam:
                    # There is a high slowness zone within our ray parameter range so might need to add a shadow zone.
                    # Need to check if the current wave type is part of the phase at this depth/ray parameter.
                    branchNum = tMod.findBranch(hszi.topDepth)
                    foundOverlap = False
                    for legNum in range(len(self.branchSeq)):
                        # Check for downgoing legs that cross the high slowness zone with the same wave type.
                        if (self.branchSeq[legNum] == branchNum and self.waveType[legNum] == isPwave
                            and self.downGoing[legNum] is True and self.branchSeq[legNum - 1] == branchNum - 1
                            and self.waveType[legNum - 1] == isPwave and self.downGoing[legNum - 1] is True):
                            foundOverlap = True
                            break
                    if foundOverlap:
                        hszIndex = self.rayParams.index(hszi.rayParam)
                        newdist = deepcopy(self.dist[:hszIndex])
                        newtime = deepcopy(self.time[:hszIndex])
                        newrayParams = deepcopy(self.rayParams[:hszIndex])
                        newrayParams.append = hszi.rayParam
                        # Sum the branches with an appropriate multiplier.
                        newdist.append(0)
                        newtime.append(0)
                        for tb, tbs, taub, taubs in zip(timesBranches[0], timesBranches[1], tMod.tauBranches[0], tMod.tauBranches[1]):
                            if tb != 0 and taub.topDepth < hszi.topDepth:
                                newdist[hszIndex] += tb * taub.dist[self.maxRayParamIndex + hszIndex - indexOffset]
                                newtime[hszIndex] += tb * taub.time[self.maxRayParamIndex + hszIndex - indexOffset]
                            if tbs != 0 and taub.topDepth < hszi.topDepth:
                                newdist[hszIndex] += tbs * taubs.dist[self.maxRayParamIndex + hszIndex - indexOffset]
                                newtime[hszIndex] += tbs * taubs.time[self.maxRayParamIndex + hszIndex - indexOffset]
                        newdist.append(self.dist[hszIndex:])
                        newtime.append(self.time[hszIndex:])
                        newrayParams.append(self.rayParams[hszIndex:])
                        indexOffset += 1
                        self.dist = newdist
                        self.time = newtime
                        self.rayParams = newrayParams

    def calcTime(self, degrees):
        """ Calculates arrival times for this phase, sorted by time.
        :param degrees:
        :return arrivals:
        """
        # Degrees must be positive and between 0 and 180
        tempDeg = abs(degrees)
        # Don't just use modulo, as 180 would be equal to 0.
        while tempDeg > 360:
            tempDeg -= 360
        if tempDeg > 180:
            tempDeg = 360 - tempDeg
        radDist = tempDeg * math.pi / 180
        arrivals = []
        # Search all distances 2n*PI+radDist and 2(n+1)*PI-radDist that are less than the
        #  maximum distance for this phase. This ensures that we get the time for phases
        # that accumulate more than 180 degrees of distance, for instance PKKKKP might
        # wrap all of the way around. A special case exists at 180, so we skip the second case if tempDeg==180.
        n = 0
        while n*2*math.pi + radDist <= self.maxDistance:
            # Look for arrivals that are radDist + 2nPi, i.e. rays that have done more than n laps.
            searchDist = n*2*math.pi + radDist
            for rayNum in range(len(self.dist) - 1):
                if searchDist == self.dist[rayNum + 1] and rayNum + 1 != len(self.dist) - 1:
                    # So we don't get 2 arrivals for the same ray.
                    continue
                elif (self.dist[rayNum] - searchDist) * (searchDist - self.dist[rayNum + 1]) >= 0:
                    # Look for distances that bracket the search distance
                    if self.rayParams[rayNum] == self.rayParams[rayNum + 1] and len(self.rayParams) > 2:
                        # Here we have a shadow zone, so itis not really an arrival.
                        continue
                    arrivals.append(self.linearInterpArrival(searchDist, rayNum,
                                                             self.name, self.puristName, self.sourceDepth))
            # Look for arrivals that are 2(n+1)Pi-radDist, i.e. rays that have done more than
            # one half lap plus some number of whole laps.
            searchDist = (n+1)*2*math.pi - radDist
            if tempDeg != 180:
                for rayNum in range(len(self.dist) - 1):
                    if searchDist == self.dist[rayNum + 1] and rayNum + 1 != len(self.dist) - 1:
                        # So we don't get 2 arrivals for the same ray.
                        continue
                    elif (self.dist[rayNum] - searchDist) * (searchDist - self.dist[rayNum + 1]) >= 0:
                        if self.rayParams[rayNum] == self.rayParams[rayNum + 1] and len(self.rayParams) > 2:
                            # Here we have a shadow zone, so itis not really an arrival.
                            continue
                        arrivals.append(self.linearInterpArrival(searchDist, rayNum,
                                                                 self.name, self.puristName, self.sourceDepth))
            n += 1
        # Perhaps these are sorted by time in the java code?
        return arrivals

    def linearInterpArrival(self, searchDist, rayNum, name, puristName, sourceDepth):
        arrivalTime = ((searchDist - self.dist[rayNum]) / (self.dist[rayNum + 1] - self.dist[rayNum])
                       * (self.time[rayNum + 1] - self.time[rayNum]) + self.time[rayNum])
        arrivalRayParam = ((searchDist - self.dist[rayNum + 1]) * (self.rayParams[rayNum] - self.rayParams[rayNum + 1])
                           / (self.dist[rayNum] - self.dist[rayNum + 1]) + self.rayParams[rayNum + 1])
        if name.endswith("kmps"):
            takeoffAngle = 0
            incidentAngle = 0
        else:
            vMod = self.tMod.sMod.vMod
            if self.downGoing[0]:
                takeoffVelocity = vMod.evaluateBelow(sourceDepth, name[0])
            else:
                # Fake negative velocity so angle is negative in case of upgoing ray.
                takeoffVelocity = -1 * vMod.evaluateAbove(sourceDepth, name[0])
            takeoffAngle = (180 / math.pi) * math.asin(takeoffVelocity * arrivalRayParam
                                                       / (self.tMod.radiusOfEarth - self.sourceDepth))
            lastLeg = self.legs[-2][0]  # very last item is "END"
            incidentAngle = (180 / math.pi) * math.asin(vMod.evaluateBelow(0, lastLeg)
                                                        * arrivalRayParam / self.tMod.radiusOfEarth)
        return Arrival(self, arrivalTime, searchDist, arrivalRayParam, rayNum, name,
                       puristName, sourceDepth, takeoffAngle, incidentAngle)

    @classmethod
    def getEarliestArrival(cls, relPhases, degrees):
        raise NotImplementedError("baaa")


def closestBranchToDepth(tMod, depthString):
    """Finds the closest discontinuity to the given depth that can hae reflections and phase transformations."""
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


def legPuller(name):
    """Tokenizes a phase name into legs, ie PcS becomes 'P'+'c'+'S' while p^410P
    would become 'p'+'^410'+'P'. Once a phase name has been broken into
    tokens we can begin to construct the sequence of branches to which it
    corresponds. Only minor error checking is done at this point, for
    instance PIP generates an exception but ^410 doesn't. It also appends
    "END" as the last leg."""
    # Java static method, so I think that means making it a function.
    # or @classmethod? But it doesn't need the class.
    offset = 0
    legs = []
    # Special case for surface wave velocity.
    if name.endswith("kmps"):
        legs.append(name)
    else:
        while offset < len(name):
            nchar = name[offset]
            # Do the easy ones, i.e. K, k, I, i, J, p, s, m, c:
            if any(nchar == c for c in ("K", "k", "I", "i", "J", "p", "s", "m", "c")):
                legs.append(nchar)
                offset += 1
            elif nchar == "P" or "S":
                # Now it gets complicated, first see if the next char is part of a different leg or if it's the end.
                if (offset + 1 == len(name) or any(name[offset + 1] == c for c in ("P", "S", "K", "m", "c", "^", "v"))
                     or name[offset + 1].isdigit()):
                    legs.append(nchar)
                    offset += 1
                elif name[offset + 1] == "p" or name[offset + 1] == "s":
                    raise TauModelError("Invalid phase name: \n "
                                        "{} cannot be followed by {} in {}.".format(nchar, name[offset+1], name))
                elif any(name[offset+1] == c for c in ("g", "b", "n")):
                    # The leg is not described by one letter, check for two:
                    legs.append(name[offset:offset+2])
                    offset += 2
                elif len(name) >= offset + 5 and any(name[offset:offset+5] == c for c in ("Sdiff", "Pdiff")):
                    legs.append(name[offset:offset+5])
                    offset += 5
                else:
                    raise TauModelError("Invalid phase name: \n "
                                        "{nchar} in {name}".format(**locals()))
            elif nchar == "^" or nchar == "v":
                # Top side or bottom side reflections, check for standard boundaries and then check for numerical ones.
                if any(name[offset+1] == c for c in ("m", "c", "i")):
                    legs.append(name[offset:offset+2])
                    offset += 2
                elif name[offset+1].isdigit() or name[offset+1] == ".":
                    numString = name[offset]
                    offset += 1
                    while name[offset+1].isdigit() or name[offset+1] == ".":
                        numString += name[offset]
                        offset += 1
                    legs.append(numString)
                else:
                    raise TauModelError("Invalid phase name {nchar} in {name}.".format(**locals()))
            elif nchar.isdigit() or nchar == ".":
                numString = name[offset]
                offset += 1
                while name[offset+1].isdigit() or name[offset+1] == ".":
                    numString += name[offset]
                    offset += 1
                legs.append(numString)
            else:
                raise TauModelError("Invalid phase name {nchar} in {name}.".format(**locals()))
    legs.append("END")
    phaseValidate(legs)
    return legs


def phaseValidate(legs):
    # Raise an exception here if validation fails.
    # Validating the phase names is IMO not necessary now, wrong names raise an exception anyway.
    pass
