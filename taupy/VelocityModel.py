#!/usr/bin/env python
"""
Package for storage and manipulation of seismic earth models.
"""
import itertools
from header import TauPException

class VelocityModel(object):
    # Some default values as class attributes [km]
    radiusOfEarth = 6371.0
    default_moho = 35
    default_cmb = 2889.0
    default_iocb = 5153.9

    def __init__(self, modelName="unknown", radiusOfEarth=radiusOfEarth,
                 mohoDepth=default_moho, cmbDepth=default_cmb, iocbDepth=default_iocb,
                 minRadius=0.0, maxRadius=6371.0, isSpherical=True,
                 layers=None):
        """
        :type modelName: str
        :param modelName: name of the velocity model.
        :type radiusOfEarth: float
        :param radiusOfEarth: reference radius (km), usually radius of the
            earth.
        :type mohoDepth: float
        :param mohoDepth: Depth (km) of the moho. It can be input from velocity
            model (*.nd) or should be explicitly set. By default it is 35
            kilometers (from Iasp91).  For phase naming, the tau model will
            choose the closest 1st order discontinuity. Thus for most simple
            earth models these values are satisfactory. Take proper care if
            your model has a thicker crust and a discontinuity near 35 km
            depth.
        :type cmbDepth: float
        :param cmbDepth: Depth (km) of the cmb (core mantle boundary). It can
            be input from velocity model (*.nd) or should be explicitly set. By
            default it is 2889 kilometers (from Iasp91). For phase naming, the
            tau model will choose the closest 1st order discontinuity. Thus for
            most simple earth models these values are satisfactory.
        :type iocbDepth: float
        :param iocbDepth: Depth (km) of the iocb (inner core outer core
            boundary). It can be input from velocity model (*.nd) or should be
            explicitly set. By default it is 5153.9 kilometers (from Iasp91).
            For phase naming, the tau model will choose the closest 1st order
            discontinuity. Thus for most simple earth models these values are
            satisfactory.
        :type minRadius: float
        :param minRadius: Minimum radius of the model (km).
        :type maxRadius: float
        :param maxRadius: Maximum radius of the model (km).
        :type isSpherical: bool
        :param isSpherical: Is this a spherical model? Defaults to true.
        """
        self.modelName = modelName
        self.radiusOfEarth = radiusOfEarth
        self.mohoDepth = mohoDepth
        self.cmbDepth = cmbDepth
        self.iocbDepth = iocbDepth
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.isSpherical = isSpherical
        self.layers = layers if layers else []
        
    def __len__(self):
        return len(self.layers)

    # @property  ?
    def getDisconDepths(self):
        """
        Returns the depths of discontinuities within the velocity model.
        """
        discontinuities = []
        
        discontinuities.append(self.layers[0].topDepth)
        for above_layer, below_layer in itertools.izip(self.layers[:-1],
                                                       self.layers[1:]):
            if above_layer.botPVelocity != below_layer.topPVelocity or \
                    above_layer.botSVelocity != below_layer.topSVelocity:
                # Discontinuity found.
                discontinuities.append(above_layer.botDepth)
        discontinuities.append(self.layers[-1].botDepth)

        return discontinuities

#	No idea what this is meant to do. The self.layer.get(layerNum)
#	has been replaced by self.layers[i] in a similar method.        
#    def getVelocityLayerClone(self, layerNum):
#        """ generated source for method getVelocityLayerClone """
#        return (self.layer.get(layerNum)).clone()

    def getNumLayers(self):
        """ Returns the number of layers in this velocity model. """
        return len(self.layer)
    
    def getLayers(self):
        """ generated source for method getLayers """
        return self.layer.toArray([None]*0)

    def layerNumberAbove(self, depth):
        """
        Finds the layer containing the given depth. Note this returns the upper
        layer if the depth happens to be at a layer boundary.

        :returns: the layer number
        """
        for i, layer in enumerate(self.layers):
            if layer.topDepth < depth <= layer.botDepth:
                return i
        raise TauPException("No such layer.")

    def layerNumberBelow(self, depth):
        """
        Finds the layer containing the given depth. Note this returns the lower
        layer if the depth happens to be at a layer boundary.

        :returns: the layer number
        """
        for i, layer in enumerate(self.layers):
            if layer.topDepth <= depth < layer.botDepth:
                return i
        raise TauPException("No such layer.")
    
    
    def evaluateAbove(self, depth, materialProperty):
        
        """
        returns the value of the given material property, usually P or S
        velocity, at the given depth. Note this returns the value at the bottom
        of the upper layer if the depth happens to be at a layer boundary.
        
        :returns: the value of the given material property
        """
        layer = self.layers[self.layerNumberAbove(depth)]
        return layer.evaluateAt(depth, materialProperty)
    

    def evaluateBelow(self, depth, materialProperty):
        """
        returns the value of the given material property, usually P or S
        velocity, at the given depth. Note this returns the value at the top
        of the lower layer if the depth happens to be at a layer boundary.

        :returns: the value of the given material property
        """
        layer = self.layers[self.layerNumberBelow(depth)]
        return layer.evaluateAt(depth, materialProperty)

    # These two seem to be just WRONG even in the java code, let's see if they're necessary for anything before fixing
    # def evaluateAtTop(self, layerNumber, materialProperty):
    #    """ 
    #    Returns the value of the given material property, usually P or S
    #	velocity, at the top of the given layer.
    #	"""
    #    tempLayer = VelocityLayer()
    #    tempLayer = self.layers[layerNumber]
    #    return tempLayer.evaluateAtTop(materialProperty)
    # def evaluateAtBottom(self, layerNumber, materialProperty):
    
    def depthAtTop(self, layerNumber):
        """ returns the depth at the top of the given layer. """
        layer = self.layers[layerNumber]
        return layer.topDepth

    def depthAtBottom(self, layerNumber):
        """ returns the depth at the bottom of the given layer. """
        layer = self.layers[layerNumber]
        return layer.botDepth

    # TO DO#####################################
    # def replaceLayers(self, newLayers, name, matchTop, matchBot):
    #     """ 
    #     replaces layers in the velocity model with new layers. The number of old
    #     and new layers need not be the same. @param matchTop false if the top
    # 	should be a discontinuity, true if the top velocity should be forced to
    # 	match the existing velocity at the top. @param matchBot similar for the
    # 	bottom.
    # 	"""
    #     topLayerNum = self.layerNumberBelow(newLayers[0].topDepth)
    #     topLayer = self.layers[topLayerNum]
    #     botLayerNum = self.layerNumberAbove(newLayers[len(newLayers)].botDepth)
    #     botLayer = self.layers[botLayerNum]
    #     outLayers = ArrayList()
    #     outLayers.addAll(self.layer)
    #     try:
    #         if matchTop:
    #             newLayers[0] = VelocityLayer(newLayers[0].getLayerNum(), newLayers[0].topDepth, newLayers[0].botDepth, topLayer.evaluateAt(newLayers[0].topDepth, 'P'), newLayers[0].botPVelocity, topLayer.evaluateAt(newLayers[0].topDepth, 'S'), newLayers[0].botSVelocity, newLayers[0].topDensity, newLayers[0].botDensity, newLayers[0].topQp, newLayers[0].botQp, newLayers[0].topQs, newLayers[0].botQs)
    #         if matchBot:
    #             newLayers[len(newLayers)] = VelocityLayer(end.getLayerNum(), end.topDepth, end.botDepth, end.topPVelocity, botLayer.evaluateAt(newLayers[len(newLayers)].botDepth, 'P'), end.topSVelocity, botLayer.evaluateAt(newLayers[len(newLayers)].botDepth, 'S'), end.topDensity, end.botDensity, end.topQp, end.botQp, end.topQs, end.botQs)
    #     except NoSuchMatPropException as e:
    #         raise RuntimeException(e)
    #     if topLayer.botDepth > newLayers[0].topDepth:
    #         try:
    #             topLayer = VelocityLayer(topLayer.getLayerNum(), topLayer.topDepth, newLayers[0].topDepth, topLayer.topPVelocity, topLayer.evaluateAt(newLayers[0].topDepth, 'P'), topLayer.topSVelocity, topLayer.evaluateAt(newLayers[0].topDepth, 'S'), topLayer.topDensity, topLayer.botDensity)
    #             outLayers.set(topIndex, topLayer)
    #         except NoSuchMatPropException as e:
    #             raise RuntimeException(e)
    #         newVLayer.topPVelocity = topLayer.botPVelocity
    #         newVLayer.topSVelocity = topLayer.botSVelocity
    #         newVLayer.topDepth = topLayer.botDepth
    #         outLayers.add(topLayerNum + 1, newVLayer)
    #         botLayerNum += 1
    #         topLayerNum += 1
    #     if botLayer.botDepth > newLayers[len(newLayers)].botDepth:
    #         try:
    #             botLayer.botPVelocity = botLayer.evaluateAt(newLayers[len(newLayers)].botDepth, 'P')
    #             botLayer.botSVelocity = botLayer.evaluateAt(newLayers[len(newLayers)].botDepth, 'S')
    #             botLayer.botDepth = newLayers[len(newLayers)].botDepth
    #         except NoSuchMatPropException as e:
    #             print("Caught NoSuchMatPropException: " + e.getMessage(), file=sys.stderr)
    #             e.printStackTrace()
    #         newVLayer.topPVelocity = botLayer.botPVelocity
    #         newVLayer.topSVelocity = botLayer.botSVelocity
    #         newVLayer.topDepth = botLayer.botDepth
    #         outLayers.add(botLayerNum + 1, newVLayer)
    #         botLayerNum += 1
    #     i = topLayerNum
    #     while i <= botLayerNum:
    #         outLayers.remove(topLayerNum)
    #         i += 1
    #     i = 0
    #     while len(newLayers):
    #         outLayers.add(topLayerNum + i, newLayers[i])
    #         i += 1
    #     outVMod = VelocityModel(name, self.getRadiusOfEarth(), self.mohoDepth(), self.cmbDepth(), self.iocbDepth(), self.minRadius(), self.maxRadius(), self.spherical(), outLayers)
    #     outVMod.fixDisconDepths()
    #     outVMod.validate()
    #     return outVMod
    ################################################
        


    # The GMT methods aren't necessary for now, copy them in again from the j2py code when needed.
    # def printGMT(self, filename):
   
    # @printGMT.register(object, PrintWriter)
    # def printGMT_0(self, dos):
    
    # def printGMTforP(self, dos):
   
    # def printGMTforS(self, dos):
   


    def validate(self):
        """ Performs internal consistency checks on the velocity model. """
        currVelocityLayer = VelocityLayer()
        prevVelocityLayer = VelocityLayer()
        #/* is radiusOfEarth positive? */
        if self.radiusOfEarth <= 0.0:
            print("Radius of earth is not positive. radiusOfEarth = " + str(self.radiusOfEarth), file=sys.stderr)
            return False
    	#/* is mohoDepth non-negative? */
        if self.mohoDepth < 0.0:
            print("mohoDepth is not non-negative. mohoDepth = " + str(self.mohoDepth), file=sys.stderr)
            return False
        #/* is cmbDepth >= mohoDepth? */
        if self.cmbDepth < self.mohoDepth:
            print("cmbDepth < mohoDepth. cmbDepth = " + str(self.cmbDepth) + " mohoDepth = " + str(self.mohoDepth), file=sys.stderr)
            return False
        #/* is cmbDepth positive? */
        if self.cmbDepth <= 0.0:
            print("cmbDepth is not positive. cmbDepth = " + (self.cmbDepth), file=sys.stderr)
            return False
        #/* is iocbDepth >= cmbDepth? */
        if self.iocbDepth < self.cmbDepth:
            print("iocbDepth < cmbDepth. iocbDepth = " + str(self.iocbDepth) + " cmbDepth = " + str(self.cmbDepth), file=sys.stderr)
            return False
        #/* is iocbDepth positive? */
        if self.iocbDepth <= 0.0:
            print("iocbDepth is not positive. iocbDepth = " + str(self.iocbDepth), file=sys.stderr)
            return False
        #/* is minRadius non-negative? */
        if self.minRadius < 0.0:
            print("minRadius is not non-negative. minRadius = " + str(self.minRadius), file=sys.stderr)
            return False
        #/* is maxRadius positive? */
        if self.maxRadius <= 0.0:
            print("maxRadius is not positive. maxRadius = " + str(self.maxRadius), file=sys.stderr)
            return False
        #/* is maxRadius > minRadius? */
        if self.maxRadius <= self.minRadius:
            print("maxRadius <= minRadius. maxRadius = " + str(self.maxRadius) + " minRadius = " + str(self.minRadius), file=sys.stderr)
            return False
        
	# Iterate over all layers, comparing each to the previous one.            
        currVelocityLayer = self.layers[0]
        prevVelocityLayer = VelocityLayer(0, currVelocityLayer.topDepth, currVelocityLayer.topDepth, currVelocityLayer.topPVelocity(), currVelocityLayer.topPVelocity(), currVelocityLayer.topSVelocity(), currVelocityLayer.topSVelocity(), currVelocityLayer.topDensity(), currVelocityLayer.topDensity())
        for layerNum in range(0, self.getNumLayers):
            currVelocityLayer = self.layers[layerNum]
            if prevVelocityLayer.botDepth != currVelocityLayer.topDepth:
            #* There is a gap in the velocity model!
                print("There is a gap in the velocity model between layers " + str((layerNum - 1)) + " and ", layerNum)
                print("prevVelocityLayer=", prevVelocityLayer, file=sys.stderr)
                print("currVelocityLayer=", currVelocityLayer, file=sys.stderr)
                return False
            if currVelocityLayer.botDepth == currVelocityLayer.topDepth:
            #   more redundant comments in the original java
                print("There is a zero thickness layer in the velocity model at layer " + layerNum, file=sys.stderr)
                print("prevVelocityLayer=", prevVelocityLayer, file=sys.stderr)
                print("currVelocityLayer=", currVelocityLayer, file=sys.stderr)
                return False
            if currVelocityLayer.topPVelocity() <= 0.0 or currVelocityLayer.botPVelocity() <= 0.0:
                print("There is a negative P velocity layer in the velocity model at layer ", layerNum, file=sys.stderr)
                return False
            if currVelocityLayer.topSVelocity() < 0.0 or currVelocityLayer.botSVelocity() < 0.0:
                print("There is a negative S velocity layer in the velocity model at layer " + layerNum, file=sys.stderr)
                return False
            if (currVelocityLayer.topPVelocity() != 0.0 and currVelocityLayer.botPVelocity() == 0.0) or (currVelocityLayer.topPVelocity() == 0.0 and currVelocityLayer.botPVelocity() != 0.0):
                print("There is a layer that goes to zero P velocity without a discontinuity in the velocity model at layer " + str(layerNum) + "\nThis would cause a divide by zero within this depth range. Try making the velocity small, followed by a discontinuity to zero velocity.", file=sys.stderr)
                return False
            if (currVelocityLayer.topSVelocity() != 0.0 and currVelocityLayer.botSVelocity() == 0.0) or (currVelocityLayer.topSVelocity() == 0.0 and currVelocityLayer.botSVelocity() != 0.0):
                print("There is a layer that goes to zero S velocity without a discontinuity in the velocity model at layer " + str(layerNum) + "\nThis would cause a divide by zero within this depth range. Try making the velocity small, followed by a discontinuity to zero velocity.", file=sys.stderr)
                return False
            prevVelocityLayer = currVelocityLayer
        return True
    
    def __str__(self):
        """ generated source for method toString """
        desc = "modelName=" + str(self.modelName) + "\n" + "\n radiusOfEarth=" + str(self.radiusOfEarth) + "\n mohoDepth=" + str(self.mohoDepth) + "\n cmbDepth=" + str(self.cmbDepth) + "\n iocbDepth=" + str(self.iocbDepth) + "\n minRadius=" + str(self.minRadius) + "\n maxRadius=" + str(self.maxRadius) + "\n spherical=" + str(self.isSpherical)
        # desc += "\ngetNumLayers()=" + str(self.getNumLayers()) + "\n"
        return desc
    
    def print_(self):
        """ generated source for method print_ """
        i = 0
        for i in range(0, self.getNumLayers):
            print(self.layers[i])            
            
#   This is screwed up, let's see if we even need it (substring doesn't exist of course)
    @classmethod
    def getModelNameFromFileName(cls, filename):
        """ generated source for method getModelNameFromFileName """
        j = filename.lastIndexOf(System.getProperty("file.separator"))
        modelFilename = filename.substring(j + 1)
        modelName = modelFilename
        if modelFilename.endsWith(".tvel"):
            modelName = modelFilename.substring(0, 5 - len(modelFilename))
        elif modelFilename.endsWith(".nd"):
            modelName = modelFilename.substring(0, 3 - len(modelFilename))
        elif modelFilename.startsWith("GB."):
            modelName = modelFilename.substring(3, len(modelFilename))
        else:
            modelName = modelFilename
        return modelName

    @classmethod
    def readVelocityFile(cls, filename):
        """ Reads in a velocity file by given file name (must be a string). The type of file is determined from the file name. Calls readTVelFile or readNDFile.
        Raises exception if the type of file cannot be determined. """
        
        # filename formatting
        if filename.endsWith(".nd"):
            fileType = ".nd"
        elif filename.endsWith(".tvel"):
            fileType = ".tvel"
        else:
            raise TauPException("File type could not be determined, please rename your file to end with .tvel or .nd")
        if fileType.startsWith("."):
            fileType = fileType[1:]
            
        # the actual reading of the velocity file
        if fileType.lower() == "nd":
            vMod = readNDFile(filename)
        elif fileType.lower() == "tvel":
            vMod = readTVelFile(filename)
        else:
            raise TauPException("File type invalid")

        vMod.fixDisconDepths()
        return vMod
    



    @classmethod
    def readTVelFile(cls, filename):
        """   
        * This method reads in a velocity model from a "tvel" ASCII text file. The
        * name of the model file for model "modelname" should be "modelname.tvel".
        * The format of the file is: comment line - generally info about the P
        * velocity model comment line - generally info about the S velocity model
        * depth pVel sVel Density depth pVel sVel Density . . .
        * 
        * The velocities are assumed to be linear between sample points. Because
        * this type of model file doesn't give complete information we make the
        * following assumptions: modelname - from the filename, with ".tvel"
        * dropped if present radiusOfEarth - the largest depth in the model
        * meanDensity - 5517.0 G - 6.67e-11
        * 
        * Comments using # are also allowed.
        * 
        * @exception VelocityModelException
        *                occurs if an EOL should have been read but wasn't. This
        *                may indicate a poorly formatted tvel file.
        """
        layers = []
        myLayerNumber = 0	# needed for calling the layer maker later

        # Read all lines in the file. Each Layer needs top and bottom values,
        # i.e. info from two lines.
        with open(filename, 'rt') as f:
            # skip first two lines as they should be the header
            # (for line in itertools.islice(f, 2, None): also works, but less elegant)  
            header1 = f.readline()
            header2 = f.readline()
            
            # Read the first line to provide initial top values.
            line = f.readline
            line = line.partition('#')[0]    #needs the other comment options
            line = line.rstrip()     # or just .strip()?'
            columns = line.split()
            topDepth = float(columns[0])
            topPVel = float(columns[1])
            topSVel = float(columns[2])
            if topSVel > topPVel:
                raise TauPException("S velocity, ",topSVel," at depth ",topDepth," is greater than the P velocity, ",topPVel)
    	    # if density is present,read it.
            if len(columns) == 4:
                topDensity = float(columns[3])
            else:
                topDensity = 5571.0
        	
	    # Iterate over the rest of the file.
            for line in f:  
                line = line.partition('#')[0]    #needs the other comment options
                line = line.rstrip()     # or just .strip()?'
                columns = line.split()
                botDepth = float(columns[0])
                botPVel = float(columns[1])
                botSVel = float(columns[2])
                if botSVel > botPVel:
                    raise TauPException("S velocity, ",botSVel," at depth ",botDepth," is greater than the P velocity, ",botPVel)
                # if density is present,read it.
                if len(columns) == 4:
                    botDensity = float(columns[3])
                else:
                    botDensity = topDensity
                    
                if len(columns) > 4:
                    raise TauPException("Your file has too much information. Stick to 4 columns.") 
                
                tempLayer = VelocityLayer(myLayerNumber, topDepth, botDepth, topPVel, botPVel, topSVel, botSVel, topDensity, botDensity)
                topDepth = botDepth
                topPVel = botPVel
                topSVel = botSVel
                topDensity = botDensity
                if tempLayer.topDepth != tempLayer.botDepth:
                    # Don't use zero thickness layers, first order discontinuities
                    # are taken care of by storing top and bottom depths.
                    layers.add(tempLayer)
                    myLayerNumber += 1
                                        
        radiusOfEarth = topDepth
        maxRadius = topDepth	
	# I assume that this is a whole earth model
	# so the maximum depth ==  maximum radius == earth radius.
        modelname = filename[:-5]
        return VelocityModel(modelName, radiusOfEarth, cls.default_moho, cls.default_cmb, cls.default_iocb, 0, maxRadius, True, layers)

        
        
        
    @classmethod
    def readNDFile(cls, filename):
        # """ generated source for method readNDFile """
        # fileIn = FileReader(file_)
        # vmod = cls.readNDFile_actually(fileIn, cls.getModelNameFromFileName(file_.__name__))
        # fileIn.close()
        # return vmod

        """      This method reads in a velocity model from a "nd" ASCII text file, the
        format used by Xgbm. The name of the model file for model "modelname"
        should be "modelname.nd". The format of the file is: depth pVel sVel
        Density Qp Qs depth pVel sVel Density Qp Qs . . . with each major
        boundary separated with a line with "mantle", "outer-core" or
        "inner-core". "moho", "cmb" and "icocb" are allowed as synonyms respectively.
        This feature makes phase interpretation much easier to
        code. Also, as they are not needed for travel time calculations, the
        density, Qp and Qs may be omitted.
        
        The velocities are assumed to be linear between sample points. Because
        this type of model file doesn't give complete information we make the
        following assumptions: 
        
        modelname - from the filename, with ".nd" dropped
        
        radiusOfEarth - the largest depth in the model
        
        Also, because this method makes use of the string tokenizer, comments are
        allowed. # as well as // signify that the rest of the line is a comment.
        C style slash-star comments are also allowed.
        
        @exception VelocityModelException
        occurs if an EOL should have been read but wasn't. This
        may indicate a poorly formatted model file. 
        """

        # Some  variables
        layers = []
        myLayerNumber = 0
        # these are only potentially changed:
        topDensity = 2.6
        topQp = 1000
        topQs = 2000
        botDensity = topDensity
        botQp = topQp
        botQs = topQs
        
        with open(filename, 'rt') as f:

            # Read the first line to provide initial top values.
            line = f.readline()
            line = line.partition('#')[0]    # other comment options?
            line = line.rstrip()
            columns = line.split()
            topDepth = float(columns[0])
            topPVel = float(columns[1])
            topSVel = float(columns[2])
            if topSVel > topPVel:
                raise TauPException("S velocity, ",topSVel," at depth ",topDepth," is greater than the P velocity, ",topPVel)
    	    # if density, Qp and Qs are present,read them.
            if len(columns) > 3:
                topDensity = float(columns[3])
                if len(columns) > 4:
                    topQp = float(columns[4])
                    if len(columns) > 5:
                        topQs = float(columns[5])
                        if len(columns) > 6:
                            raise TauPException("Your file has too much information. Stick to 6 columns.")
            # Default values which should be supplied in an ND file!
            mohoDepth = cls.default_moho
            cmbdepth = cls.default_cmb
            iocbdepth = cls.default_iocb
            
            # Loop over all remaining lines.
            for line in f:
                line = line.partition('#')[0]    # other comment options?
                line = line.rstrip()
                
                # Check for a named discontinuity
                if line.lower() == "mantle" or line.lower() == "moho":
                    mohoDepth = topDepth
                if line.lower() == "outer-core" or line.lower() == "cmb":
                    cmbDepth = topDepth
                if line.lower() == "inner-core" or line.lower() == "icocb" or line.lower() == "iocb":
                    iocbDepth = topDepth    
                
                columns = line.split()
                botDepth = float(columns[0])
                botPVel = float(columns[1])
                botSVel = float(columns[2])
                if botSVel > botPVel:
                    raise TauPException("S velocity, ",botSVel," at depth ",botDepth," is greater than the P velocity, ",botPVel)
                # if density, Qp and Qs are present,read them.
                if len(columns) > 3:
                    botDensity = float(columns[3])
                    if len(columns) > 4:
                        botQp = float(columns[4])
                        if len(columns) > 5:
                            botQs = float(columns[5])
                            if len(columns) > 6:
                                raise TauPException("Your file has too much information. Stick to 6 columns.")
                            
                tempLayer = VelocityLayer(myLayerNumber, topDepth, botDepth, topPVel, botPVel, topSVel, botSVel, topDensity, botDensity, topQp, botQp, topQs, botQs)
                topDepth = botDepth
                topPVel = botPVel
                topSVel = botSVel
                topDensity = botDensity
                topQp = botQp
                topQs = botQs
                
                if tempLayer.topDepth != tempLayer.botDepth:
            # Don't use zero thickness layers, first order discontinuities
            # are taken care of by storing top and bottom depths.
                    layers.add(tempLayer)
                    myLayerNumber += 1               
        radiusOfEarth = topDepth
        maxRadius = topDepth	#I assume that this is a whole earth model
        			# so the maximum depth is equal to the
        			# maximum radius is equal to the earth radius.
        modelName = filename[:-3]
        return VelocityModel(modelName, radiusOfEarth, mohoDepth, cmbDepth, iocbDepth, 0, maxRadius, True, layers)

    def fixDisconDepths(self):
        """ Resets depths of major discontinuities to match those existing in the
     	 input velocity model. The initial values are set such that if there is no
     	 discontinuity within the top 100 km then the moho is set to 0.0.
     	 Similarly, if there are no discontinuities at all then the cmb is set to
     	 the radius of the earth. Similarly for the iocb, except it must be a
     	 fluid to solid boundary and deeper than 100km to avoid problems with
     	 shallower fluid layers, eg oceans. """
        changeMade = False
        aboveLayer = VelocityLayer()
        belowLayer = VelocityLayer()
        mohoMin = 65.0
        cmbMin = self.radiusOfEarth
        iocbMin = self.radiusOfEarth - 100.0
        tempMohoDepth = 0.0
        tempCmbDepth = self.radiusOfEarth
        tempIocbDepth = self.radiusOfEarth
        layerNum = 0
        while layerNum < self.getNumLayers() - 1:
            aboveLayer = self.layers[layerNum]
            belowLayer = self.layers[layerNum + 1]
            if aboveLayer.botPVelocity != belowLayer.topPVelocity or aboveLayer.botSVelocity != belowLayer.topSVelocity:	# a discontinuity
            
                if Math.abs(self.mohoDepth - aboveLayer.botDepth) < mohoMin:
                    tempMohoDepth = aboveLayer.botDepth
                    mohoMin = Math.abs(self.mohoDepth - aboveLayer.botDepth)
                if Math.abs(self.cmbDepth - aboveLayer.botDepth) < cmbMin:
                    tempCmbDepth = aboveLayer.botDepth
                    cmbMin = Math.abs(self.cmbDepth - aboveLayer.botDepth)
                if aboveLayer.botSVelocity == 0.0 and belowLayer.topSVelocity > 0.0 and Math.abs(self.iocbDepth - aboveLayer.botDepth) < iocbMin:
                    tempIocbDepth = aboveLayer.botDepth
                    iocbMin = Math.abs(self.iocbDepth - aboveLayer.botDepth)
            layerNum += 1
        if self.mohoDepth != tempMohoDepth or self.cmbDepth != tempCmbDepth or self.iocbDepth != tempIocbDepth:
            changeMade = True
        self.mohoDepth = tempMohoDepth
        self.cmbDepth = tempCmbDepth
        self.iocbDepth = (tempIocbDepth if tempCmbDepth != tempIocbDepth else self.radiusOfEarth)
        return changeMade

    def earthFlattenTransform(self):
        """ Returns a flat velocity model object equivalent to the spherical
         velocity model via the earth flattening transform.
     	 
     	 @return the flattened VelocityModel object.
     	 @exception VelocityModelException
     	                occurs ???. """
        newLayer = VelocityLayer()
        oldLayer = VelocityLayer()
        isSpherical = False
        layers = ArrayList(self.vectorLength)
        i = 0
        while i < self.getNumLayers():
            oldLayer = self.layers[i]
            newLayer = VelocityLayer(i, self.radiusOfEarth * Math.log(oldLayer.topDepth / self.radiusOfEarth), self.radiusOfEarth * Math.log(oldLayer.botDepth / self.radiusOfEarth), self.radiusOfEarth * oldLayer.topPVelocity / oldLayer.topDepth, self.radiusOfEarth * oldLayer.botPVelocity / oldLayer.botDepth, self.radiusOfEarth * oldLayer.topSVelocity / oldLayer.topDepth, self.radiusOfEarth * oldLayer.botSVelocity / oldLayer.botDepth)
            layers.add(newLayer)
            i += 1
        return VelocityModel(self.modelName, self.radiusOfEarth(), self.mohoDepth(), self.cmbDepth(), self.iocbDepth(), self.minRadius(), self.maxRadius(), spherical, layers)
