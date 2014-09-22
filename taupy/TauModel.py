from taupy.helper_classes import SlownessModelError, TauModelError
from .TauBranch import TauBranch
from itertools import count
from math import pi
import pickle


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
        # so the size is exactly right. NB slicing means deep copy
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
