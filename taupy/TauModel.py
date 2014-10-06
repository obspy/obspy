from taupy.helper_classes import SlownessModelError, TauModelError
from .TauBranch import TauBranch
from itertools import count
from math import pi
import pickle
from copy import deepcopy


# noinspection PyPep8Naming
class TauModel(object):
    """Provides storage of all the TauBranches comprising a model."""
    
    DEBUG = False
    # True if this is a spherical slowness model. False if flat.
    spherical = True
    # Depth for which tau model was constructed.
    sourceDepth = 0.0
    # Branch with the source at its top.
    sourceBranch = 0
    # Depths that should not have reflections or phase conversions. For
    # instance, if the source is not at a branch boundary then noDisconDepths
    # contains source depth and reflections and phase conversions are not
    # allowed at this branch boundary. If the source happens to fall on a real
    # discontinuity then then it is not included.
    noDisconDepths = []

    ####### This code from Java left in for the documentation purposes, for now #######
    # # Depth o1f the moho.
    # protected double mohoDepth;
    # /** Branch with the moho at its top. */
    # protected int mohoBranch;
    # /** Depth of the cmb. */
    # protected double cmbDepth;
    # /** Branch with the cmb at its top. */
    # protected int cmbBranch;
    # /** Depth of the iocb. */
    # protected double iocbDepth;
    # /** Branch with the iocb at its top. */
    # protected int iocbBranch;
    # Radius of the Earth in km, usually input from the velocity model.
    # The slowness model that was used to generate the tau model. This in
    # needed in order to modify the tau branches from a surface focus event to
    # an event at depth. This is normally be set when the tau model is
    # generated to be a clone of the slowness model.
    # private SlownessModel sMod;

    radiusOfEarth = 6371.0
    # Ray parameters used to construct the tau branches. This may only be a
    # subset of the slownesses/ray parameters saved in the slowness model due
    # to high slowness zones (low velocity zones).
    rayParams = []
    # 2D "array" (list of lists in Python) containing a TauBranch object corresponding to each "branch" of the tau
    # model, First list is P, second is S. Branches correspond to depth regions between discontinuities or reversals in
    # slowness gradient for a wave type. Each branch contains time, distance, and tau increments for each ray parameter
    # in rayParams for the layer. Rays that turn above the branch layer get 0 for time, distance, and tau increments.
    tauBranches = [[], []]

    def __init__(self, sMod):
        
        self.sMod = sMod
        self.calcTauIncFrom()

        print("This is the init method of TauModel. Hello!")

    def calcTauIncFrom(self):
        """Calculates tau for each branch within a slowness model."""
        # First, we must have at least 1 slowness layer to calculate a
        #  distance. Otherwise we must signal an exception.
        if self.sMod.getNumLayers(True) == 0 or self.sMod.getNumLayers(False) == 0:
            raise SlownessModelError("Can't calculate tauInc when getNumLayers() = 0. I need more slowness samples.")
        if not self.sMod.validate():
            raise SlownessModelError("Validation of SlownessModel failed")
        radiusOfEarth = self.sMod.radiusOfEarth
        sourceDepth = 0
        sourceBranch = 0
        # Create an array holding the ray parameter that we will use for constructing the tau splines. Only store ray
        # parameters that are not in a high slowness zone, i.e. they are smaller than the minimum ray parameter
        # encountered so far.
        numBranches = len(self.sMod.criticalDepths) - 1
        # Use list comprehension to get array to correct size (could initialise with TauBranches, but probably slower):
        self.tauBranches = [[0 for j in range(numBranches)] for i in range(2)]
        # Here we find the list of ray parameters to be used for the tau model. We only need to find ray parameters for
        # S waves since P waves have been constructed to be a subset of the S samples.
        rayNum = 0
        minPSoFar = self.sMod.SLayers[0].topP
        tempRayParams = [0 for i in range(2*self.sMod.getNumLayers(False) + len(self.sMod.criticalDepths))]
        # Make sure we get the top slowness of the very top layer
        tempRayParams[rayNum] = minPSoFar
        rayNum += 1
        for currSLayer in self.sMod.SLayers:
            # Add the top if it is strictly less than the last sample added. Note that this will not be
            # added if the slowness is continuous across the layer boundary.
            if currSLayer.topP < minPSoFar:
                tempRayParams[rayNum] = currSLayer.topP
                rayNum += 1
                minPSoFar = currSLayer.topP
            # Add the bottom if it is strictly less than the last sample added. This will always happen unless we are
            # within a high slowness zone.
            if currSLayer.botP < minPSoFar:
                tempRayParams[rayNum] = currSLayer.botP
                rayNum += 1
                minPSoFar = currSLayer.botP
        # Copy tempRayParams to rayParams while chopping off trailing zeros (from the initialisation),
        # so the size is exactly right. NB slicing doesn't really mean deep copy, but it works for a list of doubles like this
        self.rayParams = tempRayParams[:rayNum]
        if self.DEBUG:
            print("Number of slowness samples for tau:" + str(rayNum))
        for waveNum, isPWave in enumerate([True, False]):
            # The minimum slowness seen so far.
            minPSoFar = self.sMod.getSlownessLayer(0, isPWave).topP
            # for critNum, (topCritDepth, botCritDepth) in enumerate(zip(self.sMod.criticalDepths[:-1],
            #                                                            self.sMod.criticalDepths[1:])):
            # Faster:
            for critNum, topCritDepth, botCritDepth in zip(count(), self.sMod.criticalDepths[:-1],
                                                           self.sMod.criticalDepths[1:]):
                topCritLayerNum = topCritDepth.pLayerNum if isPWave else topCritDepth.sLayerNum
                botCritLayerNum = (botCritDepth.pLayerNum if isPWave else botCritDepth.sLayerNum) - 1
                self.tauBranches[waveNum][critNum] = TauBranch(topCritDepth.depth, botCritDepth.depth, isPWave)
                self.tauBranches[waveNum][critNum].DEBUG = self.DEBUG
                self.tauBranches[waveNum][critNum].createBranch(self.sMod, minPSoFar, self.rayParams)
                # Update minPSoFar. Note that the new minPSoFar could be at the start of a discontinuity over a high
                # slowness zone, so we need to check the top, bottom and the layer just above the discontinuity.
                topSLayer = self.sMod.getSlownessLayer(topCritLayerNum, isPWave)
                botSLayer = self.sMod.getSlownessLayer(botCritLayerNum, isPWave)
                minPSoFar = min(minPSoFar, min(topSLayer.topP, botSLayer.botP))
                botSLayer = self.sMod.getSlownessLayer(self.sMod.layerNumberAbove(botCritDepth.depth,
                                                                                  isPWave), isPWave)
                minPSoFar = min(minPSoFar, botSLayer.botP)
        # Here we decide which branches are the closest to the Moho, CMB, and IOCB by comparing the depth of the
        # top of the branch with the depths in the Velocity Model.
        bestMoho = 1e300
        bestCmb = 1e300
        bestIocb = 1e300
        for branchNum, tBranch in enumerate(self.tauBranches[0]):
            if abs(tBranch.topDepth - self.sMod.vMod.mohoDepth) <= bestMoho:
                # Branch with Moho at its top.
                self.mohoBranch = branchNum
                bestMoho = abs(tBranch.topDepth - self.sMod.vMod.mohoDepth)
            if abs(tBranch.topDepth - self.sMod.vMod.cmbDepth) < bestCmb:
                self.cmbBranch = branchNum
                bestCmb = abs(tBranch.topDepth - self.sMod.vMod.cmbDepth)
            if abs(tBranch.topDepth - self.sMod.vMod.iocbDepth) < bestIocb:
                self.iocbBranch = branchNum
                bestIocb = abs(tBranch.topDepth - self.sMod.vMod.iocbDepth)
        # Now set mohoDepth etc. to the top of the branches we have decided on.
        self.mohoDepth = self.tauBranches[0][self.mohoBranch].topDepth
        self.cmbDepth = self.tauBranches[0][self.cmbBranch].topDepth
        self.iocbDepth = self.tauBranches[0][self.iocbBranch].topDepth
        if not self.validate():
            raise TauModelError("TauModel.calcTauIncFrom: Validation failed!")

    def writeModel(self, outfile):
        with open(outfile, 'w+b') as f:
            pickle.dump(self, f, -1)

    def __str__(self):
        desc = "Delta tau for each slowness sample and layer.\n"
        for j, rayParam in enumerate(self.rayParams):
            for i, tb in enumerate(self.tauBranches[0]):
                desc += (" i " + str(i) + " j " + str(j) + " rayParam " + str(rayParam)
                         + " tau " + str(tb.tau[j]) + " time "
                         + str(tb.time[j]) + " dist "
                         + str(tb.dist[j]) + " degrees "
                         + str(tb.dist[j] * 180 / pi) + "\n")
            desc += "\n"
        return desc

    def validate(self):
        # TODO: implement the model validation; not critical right now
        return True

    def depthCorrect(self, depth):
        """Called in TauP_Time. Computes a new tau model for a source at depth using the previously computed branches
        for a surface source. No change is needed to the branches above and below the branch containing the depth,
        except for the addition of a slowness sample. The branch containing the source depth is split into 2 branches,
        and up going branch and a downgoing branch. Additionally, the slowness at the source depth must be sampled
        exactly as it is an extremal point for each of these branches. Cf. Buland and Chapman p 1290.
        """
        if self.sourceDepth != 0:
            raise TauModelError("Can't depth correct a TauModel that is not originally for a surface source.")
        if depth > self.radiusOfEarth:
            raise TauModelError("Can't depth correct to a source deeper than the radius of the Earth.")
        depthCorrected = self.loadFromDepthCache(depth)
        if depthCorrected is None:
            depthCorrected = self.splitBranch(depth)
            depthCorrected.sourceDepth = depth
            depthCorrected.sourceBranch = depthCorrected.findBranch(depth)
            depthCorrected.validate()
            # Put in cache somehow: self.depthCache.put(depthCorrected)
        return depthCorrected

    def loadFromDepthCache(self, depth):
        # Could speed up by implementing cache.
        # Must return None if loading fails.
        return None

    def splitBranch(self, depth):
        """Returns a new TauModel with the branches containing depth split at depth.
         Used for putting a source at depth since a source can only be located on a branch boundary.
         """
        # First check to see if depth happens to already be a branch boundary, then just return original tMod.
        for tb in self.tauBranches[0]:
            if tb.topDepth == depth or tb.botDepth == depth:
                return deepcopy(self)
        # Depth is not a branch boundary, so must modify the tau model.
        indexP = -1
        PWaveRayParam = -1
        indexS = -1
        SWaveRayParam = -1
        outSMod = self.sMod
        outRayParams = deepcopy(self.rayParams)  # necessary?
        oldRayParams = self.rayParams
        # Do S wave first since the S ray param is > P ray param.
        for isPWave in [False, True]:
            splitInfo = outSMod.splitLayer(depth, isPWave)
            outSMod = splitInfo.sMod
            if splitInfo.neededSplit and not splitInfo.movedSample:
                # Split the slowness layers containing depth into two layers each.
                newRayParam = splitInfo.rayParam
                # Insert the new ray parameters into the rayParams array.
                for index, trp, brp in zip(count(), oldRayParams[:-1], oldRayParams[1:]):
                    if trp < newRayParam < brp:
                        outRayParams = oldRayParams[:index]
                        outRayParams.append(newRayParam)
                        outRayParams = outRayParams + oldRayParams[index:]  # Can't use append here!
                        if isPWave:
                            indexP = index
                            PWaveRayParam = newRayParam
                        else:
                            indexS = index
                            SWaveRayParam = newRayParam
                        break
        # Now add a sample to each branch above the depth, split the branch containing the depth,
        # and add a sample to each deeper branch.
        branchToSplit = self.findBranch(depth)
        newTauBranches = [[TauBranch() for j in range(len(self.tauBranches[0]) + 1)] for i in range(2)]  # just initialise for size
        for i in range(branchToSplit):
            newTauBranches[0][i] = self.tauBranches[0][i]
            newTauBranches[1][i] = self.tauBranches[1][i]
            # Add the new ray parameter(s) from splitting the S and/or P wave slowness layer to both the P and
            # S wave tau branches (if splitting occurred).
            if indexS != -1:
                newTauBranches[0][i].insert(SWaveRayParam, outSMod, indexS)
                newTauBranches[1][i].insert(SWaveRayParam, outSMod, indexS)
            if indexP != -1:
                newTauBranches[0][i].insert(PWaveRayParam, outSMod, indexP)
                newTauBranches[1][i].insert(PWaveRayParam, outSMod, indexP)
        for pOrS in range(2):
            newTauBranches[pOrS][branchToSplit] = TauBranch(self.tauBranches[pOrS][branchToSplit].topDepth,
                                                            depth, pOrS == 0)
            newTauBranches[pOrS][branchToSplit].createBranch(outSMod, self.tauBranches[pOrS][branchToSplit].maxRayParam,
                                                             outRayParams)
            newTauBranches[pOrS][branchToSplit + 1] = self.tauBranches[pOrS][branchToSplit].difference(newTauBranches[pOrS][branchToSplit],
                                                                                                       indexP, indexS, outSMod,
                                                                                                       newTauBranches[pOrS][branchToSplit].minRayParam,
                                                                                                       outRayParams)
        for i in range(branchToSplit + 1, len(self.tauBranches[0])):
            for pOrS in range(2):
                newTauBranches[pOrS][i + 1] = self.tauBranches[pOrS][i]
            if indexS != -1:
                # Add the new ray parameter from splitting the S wave slownes layer to both the P
                # and S wave tau branches.
                for pOrS in range(2):
                    newTauBranches[pOrS][i + 1].insert(SWaveRayParam, outSMod, indexS)
            if indexP != -1:
                # Add the new ray parameter from splitting the P wave slownes layer to both the P
                # and S wave tau branches.
                for pOrS in range(2):
                    newTauBranches[pOrS][i + 1].insert(PWaveRayParam, outSMod, indexS)
        # We have split a branch so possibly sourceBranch, mohoBranch, cmbBranch and iocbBranch are off by 1.
        outSourceBranch = self.sourceBranch
        if self.sourceDepth > depth:
            outSourceBranch += 1
        outmohoBranch = self.mohoBranch
        if self.mohoDepth > depth:
            outmohoBranch += 1
        outcmbBranch = self.cmbBranch
        if self.cmbDepth > depth:
            outcmbBranch += 1
        outiocbBranch = self.iocbBranch
        if self.iocbDepth > depth:
            outiocbBranch += 1
        # No overloaded constructors - so do it this way to bypass the calcTauIncFrom in the __init__.
        tMod = deepcopy(self)  # Objects apparently are shallowly copied...
        tMod.sourceBranch = outSourceBranch
        tMod.mohoBranch = outmohoBranch
        tMod.cmbBranch = outcmbBranch
        tMod.iocbBranch = outiocbBranch
        tMod.sMod = outSMod
        tMod.rayParams = outRayParams
        tMod.tauBranches = newTauBranches
        tMod.noDisconDepths.append(depth)
        if not tMod.validate():
            raise TauModelError("SplitBranch validation failed!")
        return tMod

    def findBranch(self, depth):
        """Finds the branch that either has the depth as its top boundary, or
        strictly contains the depth. Also, we allow the bottom-most branch to
        contain its bottom depth, so that the center of the earth is contained
        within the bottom branch."""
        for i, tb in enumerate(self.tauBranches[0]):
            if tb.topDepth <= depth < tb.botDepth:
                return i
        # Check to see if depth is centre of the Earth.
        if self.tauBranches[0][len(self.tauBranches[0]) - 1].botDepth == depth:
            return len(self.tauBranches) - 1
        else:
            raise TauModelError("No TauBranch contains this depth.")

    def getTauBranch(self, branchNum, isPWave):
        if isPWave:
            return self.tauBranches[0][branchNum]
        else:
            return self.tauBranches[1][branchNum]