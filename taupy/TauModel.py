from taupy.helper_classes import SlownessModelError


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

    #TODO: In the java there is a second constructor with more values, maybe refer to how I did that in SlownessModel.
    def __init__(self, sMod):
        
        self.sMod = sMod
        self.calcTauIncFrom()

        print("This is the init method of TauModel clocking in. Hello!")
        print("The debug flag for TauModel is set to:"+str(self.DEBUG), 
              "the default here is false but it should have been modified TauP_Create.")

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
        # Use list comprehension to get array to correct size:
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
        # line 393

    def writeModel(self, outfile):
        with open(outfile, 'w') as f:
            f.write("A tau model should be here")

    def __str__(self):
        desc = 'This is the TauModel __str__ method.'
        return desc
