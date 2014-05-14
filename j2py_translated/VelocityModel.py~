#!/usr/bin/env python
""" generated source for module VelocityModel """
# 
#  * The TauP Toolkit: Flexible Seismic Travel-Time and Raypath Utilities.
#  * Copyright (C) 1998-2000 University of South Carolina
#  * 
#  * This program is free software; you can redistribute it and/or modify it under
#  * the terms of the GNU General Public License as published by the Free Software
#  * Foundation; either version 2 of the License, or (at your option) any later
#  * version.
#  * 
#  * This program is distributed in the hope that it will be useful, but WITHOUT
#  * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  * details.
#  * 
#  * You should have received a copy of the GNU General Public License along with
#  * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
#  * Place - Suite 330, Boston, MA 02111-1307, USA.
#  * 
#  * The current version can be found at <A
#  * HREF="www.seis.sc.edu">http://www.seis.sc.edu</A>
#  * 
#  * Bug reports and comments should be directed to H. Philip Crotwell,
#  * crotwell@seis.sc.edu or Tom Owens, owens@seis.sc.edu
#  * 
#  
# 
#  * package for storage and manipulation of seismic earth models.
#  * 
#  
# package: edu.sc.seis.TauP
import java.io.BufferedOutputStream

import java.io.BufferedWriter

import java.io.DataOutputStream

import java.io.File

import java.io.FileOutputStream

import java.io.FileReader

import java.io.FileWriter

import java.io.IOException

import java.io.PrintWriter

import java.io.Reader

import java.io.Serializable

import java.io.StreamTokenizer

import java.util.ArrayList

import java.util.List

# 
#  * This class defines basic classes to store and manipulate a velocity model.
#  * 
#  * @version 1.1.3 Wed Jul 18 15:00:35 GMT 2001
#  * 
#  * 
#  * 
#  * @author H. Philip Crotwell
#  
class VelocityModel(Cloneable, Serializable):
    """ generated source for class VelocityModel """
    def __init__(self, modelName, radiusOfEarth, mohoDepth, cmbDepth, iocbDepth, minRadius, maxRadius, spherical, layer):
        """ generated source for method __init__ """
        super(VelocityModel, self).__init__()
        self.modelName = modelName
        self.radiusOfEarth = radiusOfEarth
        self.mohoDepth = mohoDepth
        self.cmbDepth = cmbDepth
        self.iocbDepth = iocbDepth
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.spherical = spherical
        self.layer = layer

    #  name of the velocity model. 
    modelName = "unknown"

    # 
    #      * reference radius (km), usually radius of the earth, by default 6371
    #      * kilometers.
    #      
    radiusOfEarth = 6371.0

    #  kilometers
    DEFAULT_MOHO = 35
    DEFAULT_CMB = 2889.0
    DEFAULT_IOCB = 5153.9

    # 
    #      * Depth (km) of the moho. It can be input from velocity model (*.nd) or
    #      * should be explicitly set. By default it is 35 kilometers (from Iasp91).
    #      * For phase naming, the tau model will choose the closest 1st order
    #      * discontinuity. Thus for most simple earth models these values are
    #      * satisfactory. Take proper care if your model has a thicker crust and a
    #      * discontinuity near 35 km depth.
    #      
    mohoDepth = DEFAULT_MOHO

    #  kilometers
    # 
    #      * Depth (km) of the cmb (core mantle boundary). It can be input from
    #      * velocity model (*.nd) or should be explicitly set. By default it is 2889
    #      * kilometers (from Iasp91). For phase naming, the tau model will choose the
    #      * closest 1st order discontinuity. Thus for most simple earth models these
    #      * values are satisfactory.
    #      
    cmbDepth = DEFAULT_CMB

    #  kilometers
    # 
    #      * Depth (km) of the iocb (inner core outer core boundary). It can be input
    #      * from velocity model (*.nd) or should be explicitly set. By default it is
    #      * 5153.9 kilometers (from Iasp91). For phase naming, the tau model will
    #      * choose the closest 1st order discontinuity. Thus for most simple earth
    #      * models these values are satisfactory.
    #      
    iocbDepth = DEFAULT_IOCB

    #  kilometers
    #  minimum radius of the model (km), default 0.0 
    minRadius = 0.0

    #  kilometers
    #  maximum radius of the model (km), default 6371.0 
    maxRadius = 6371.0

    #  kilometers
    #  is this a spherical model? Default is true. 
    spherical = True

    #  the initial length of the layer vector. 
    vectorLength = 16

    # 
    #      * expandable array to hold the layers
    #      * 
    #      
    layer = List()

    # ----------------------------------------
    #      METHODS
    #      ----------------------------------------
    #  Accessor methods
    #  get the model name. 
    def getModelName(self):
        """ generated source for method getModelName """
        return self.modelName

    #  set the model name. 
    def setModelName(self, modelName):
        """ generated source for method setModelName """
        if 0 > len(modelName):
            self.modelName = modelName
        else:
            self.modelName = "unknown"

    # 
    #      * sets radius of the earth (km), by default 6371 kilometers.
    #      
    def setRadiusOfEarth(self, radiusOfEarth):
        """ generated source for method setRadiusOfEarth """
        self.radiusOfEarth = radiusOfEarth

    # 
    #      * gets radius of the earth (km), by default 6371 kilometers.
    #      
    def getRadiusOfEarth(self):
        """ generated source for method getRadiusOfEarth """
        return self.radiusOfEarth

    #  @return the depths of discontinuities within the velocity model 
    def getDisconDepths(self):
        """ generated source for method getDisconDepths """
        disconDepths = [None]*getNumLayers() + 2
        numFound = 0
        aboveLayer = VelocityLayer()
        belowLayer = VelocityLayer()
        __numFound_0 = numFound
        numFound += 1
        disconDepths[__numFound_0] = getVelocityLayer(0).getTopDepth()
        layerNum = 0
        while layerNum < getNumLayers() - 1:
        __numFound_1 = numFound
        numFound += 1
            aboveLayer = getVelocityLayer(layerNum)
            belowLayer = getVelocityLayer(layerNum + 1)
            if aboveLayer.getBotPVelocity() != belowLayer.getTopPVelocity() or aboveLayer.getBotSVelocity() != belowLayer.getTopSVelocity():
                #  a discontinuity
                disconDepths[__numFound_1] = aboveLayer.getBotDepth()
            layerNum += 1
        __numFound_2 = numFound
        numFound += 1
        disconDepths[__numFound_2] = getVelocityLayer(getNumLayers() - 1).getBotDepth()
        temp = [None]*numFound
        System.arraycopy(disconDepths, 0, temp, 0, numFound)
        return temp

    # 
    #      * @return depth (km) of the moho. It can be input from velocity model
    #      *          (*.nd) or should be explicitly set. By default it is 35
    #      *          kilometers (from Iasp91). For phase naming, the tau model will
    #      *          choose the closest 1st order discontinuity. Thus for most simple
    #      *          earth models these values are satisfactory. Take proper care if
    #      *          your model has a thicker crust and a discontinuity near 35 km
    #      *          depth.
    #      
    def getMohoDepth(self):
        """ generated source for method getMohoDepth """
        return self.mohoDepth

    def setMohoDepth(self, mohoDepth):
        """ generated source for method setMohoDepth """
        self.mohoDepth = mohoDepth

    # 
    #      * @return depth (km) of the cmb (core mantle boundary). It can be input
    #      *          from velocity model (*.nd) or should be explicitly set. By
    #      *          default it is 2889 kilometers (from Iasp91). For phase naming,
    #      *          the tau model will choose the closest 1st order discontinuity.
    #      *          Thus for most simple earth models these values are satisfactory.
    #      
    def getCmbDepth(self):
        """ generated source for method getCmbDepth """
        return self.cmbDepth

    def setCmbDepth(self, cmbDepth):
        """ generated source for method setCmbDepth """
        self.cmbDepth = cmbDepth

    # 
    #      * @return the depth (km) of the iocb (inner core outer core boundary). It
    #      *          can be input from velocity model (*.nd) or should be explicitly
    #      *          set. By default it is 5153.9 kilometers (from Iasp91). For phase
    #      *          naming, the tau model will choose the closest 1st order
    #      *          discontinuity. Thus for most simple earth models these values
    #      *          are satisfactory.
    #      
    def getIocbDepth(self):
        """ generated source for method getIocbDepth """
        return self.iocbDepth

    def setIocbDepth(self, iocbDepth):
        """ generated source for method setIocbDepth """
        self.iocbDepth = iocbDepth

    def getMinRadius(self):
        """ generated source for method getMinRadius """
        return self.minRadius

    def setMinRadius(self, minRadius):
        """ generated source for method setMinRadius """
        self.minRadius = minRadius

    def getMaxRadius(self):
        """ generated source for method getMaxRadius """
        return self.maxRadius

    def setMaxRadius(self, maxRadius):
        """ generated source for method setMaxRadius """
        self.maxRadius = maxRadius

    def getSpherical(self):
        """ generated source for method getSpherical """
        return self.spherical

    def setSpherical(self, spherical):
        """ generated source for method setSpherical """
        self.spherical = spherical

    def getVelocityLayerClone(self, layerNum):
        """ generated source for method getVelocityLayerClone """
        return (self.layer.get(layerNum)).clone()

    def getVelocityLayer(self, layerNum):
        """ generated source for method getVelocityLayer """
        return self.layer.get(layerNum)

    #  Returns the number of layers in this velocity model. 
    def getNumLayers(self):
        """ generated source for method getNumLayers """
        return len(self.layer)

    def getLayers(self):
        """ generated source for method getLayers """
        return self.layer.toArray([None]*0)

    #  normal methods
    # 
    #      * Finds the layer containing the given depth. Note this returns the upper
    #      * layer if the depth happens to be at a layer boundary.
    #      * 
    #      * @return the layer number
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer contains the given depth.
    #      
    def layerNumberAbove(self, depth):
        """ generated source for method layerNumberAbove """
        tempLayer = VelocityLayer()
        #  first check to see if depth is at top of top layer. 
        tempLayer = self.getVelocityLayer(0)
        if depth == tempLayer.getTopDepth():
            return 0
        else:
            if depth < tempLayer.getTopDepth() or self.getVelocityLayer(tooLargeNum).getBotDepth() < depth:
                raise NoSuchLayerException(depth)
            while not found:
                currentNum = Math.round((tooSmallNum + tooLargeNum) / 2.0)
                tempLayer = self.getVelocityLayer(currentNum)
                if tempLayer.getTopDepth() >= depth:
                    tooLargeNum = currentNum - 1
                elif tempLayer.getBotDepth() < depth:
                    tooSmallNum = currentNum + 1
                else:
                    found = True
            return currentNum

    # 
    #      * Finds the layer containing the given depth. Note this returns the lower
    #      * layer if the depth happens to be at a layer boundary.
    #      * 
    #      * @return the layer number
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer contains the given depth.
    #      
    def layerNumberBelow(self, depth):
        """ generated source for method layerNumberBelow """
        tempLayer = self.getVelocityLayer(0)
        tooSmallNum = 0
        tooLargeNum = self.getNumLayers() - 1
        currentNum = 0
        found = False
        #  first check to see if depth is at top of top layer. 
        if depth == tempLayer.getTopDepth():
            return 0
        elif self.getVelocityLayer(tooLargeNum).getBotDepth() == depth:
            #  and check the bottommost layer. 
            return tooLargeNum
        else:
            if depth < tempLayer.getTopDepth() or self.getVelocityLayer(tooLargeNum).getBotDepth() < depth:
                raise NoSuchLayerException(depth)
            while not found:
                currentNum = Math.round((tooSmallNum + tooLargeNum) / 2.0)
                tempLayer = self.getVelocityLayer(currentNum)
                if tempLayer.getTopDepth() > depth:
                    tooLargeNum = currentNum - 1
                elif tempLayer.getBotDepth() <= depth:
                    tooSmallNum = currentNum + 1
                else:
                    found = True
            return currentNum

    # 
    #      * returns the value of the given material property, usually P or S
    #      * velocity, at the given depth. Note this returns the value at the bottom
    #      * of the upper layer if the depth happens to be at a layer boundary.
    #      * 
    #      * @return the value of the given material property
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer contains the given depth.
    #      * @exception NoSuchMatPropException
    #      *                occurs if the material property is not recognized.
    #      
    def evaluateAbove(self, depth, materialProperty):
        """ generated source for method evaluateAbove """
        tempLayer = VelocityLayer()
        tempLayer = self.getVelocityLayer(self.layerNumberAbove(depth))
        return tempLayer.evaluateAt(depth, materialProperty)

    # 
    #      * returns the value of the given material property, usually P or S
    #      * velocity, at the given depth. Note this returns the value at the top of
    #      * the lower layer if the depth happens to be at a layer boundary.
    #      * 
    #      * @return the value of the given material property
    #      * @exception NoSuchLayerException
    #      *                occurs if no layer contains the given depth.
    #      * @exception NoSuchMatPropException
    #      *                occurs if the material property is not recognized.
    #      
    def evaluateBelow(self, depth, materialProperty):
        """ generated source for method evaluateBelow """
        tempLayer = VelocityLayer()
        tempLayer = self.getVelocityLayer(self.layerNumberBelow(depth))
        return tempLayer.evaluateAt(depth, materialProperty)

    # 
    #      * returns the value of the given material property, usually P or S
    #      * velocity, at the top of the given layer.
    #      * 
    #      * @return the value of the given material property
    #      * @exception NoSuchMatPropException
    #      *                occurs if the material property is not recognized.
    #      
    def evaluateAtTop(self, layerNumber, materialProperty):
        """ generated source for method evaluateAtTop """
        tempLayer = VelocityLayer()
        tempLayer = self.getVelocityLayer(layerNumber)
        return tempLayer.evaluateAtTop(materialProperty)

    # 
    #      * returns the value of the given material property, usually P or S
    #      * velocity, at the bottom of the given layer.
    #      * 
    #      * @return the value of the given material property
    #      * @exception NoSuchMatPropException
    #      *                occurs if the material property is not recognized.
    #      
    def evaluateAtBottom(self, layerNumber, materialProperty):
        """ generated source for method evaluateAtBottom """
        tempLayer = VelocityLayer()
        tempLayer = self.getVelocityLayer(layerNumber)
        return tempLayer.evaluateAtBottom(materialProperty)

    # 
    #      * returns the depth at the top of the given layer.
    #      * 
    #      * @return the depth.
    #      
    def depthAtTop(self, layerNumber):
        """ generated source for method depthAtTop """
        tempLayer = VelocityLayer()
        tempLayer = self.getVelocityLayer(layerNumber)
        return tempLayer.getTopDepth()

    # 
    #      * returns the depth at the bottom of the given layer.
    #      * 
    #      * @return the depth.
    #      * @exception NoSuchMatPropException
    #      *                occurs if the material property is not recognized.
    #      
    def depthAtBottom(self, layerNumber):
        """ generated source for method depthAtBottom """
        tempLayer = VelocityLayer()
        tempLayer = self.getVelocityLayer(layerNumber)
        return tempLayer.getBotDepth()

    # 
    #      * replaces layers in the velocity model with new layers. The number of old
    #      * and new layers need not be the same. @param matchTop false if the top
    #      * should be a discontinuity, true if the top velocity should be forced to
    #      * match the existing velocity at the top. @param matchBot similar for the
    #      * bottom.
    #      
    def replaceLayers(self, newLayers, name, matchTop, matchBot):
        """ generated source for method replaceLayers """
        topLayerNum = self.layerNumberBelow(newLayers[0].getTopDepth())
        topLayer = self.getVelocityLayer(topLayerNum)
        botLayerNum = self.layerNumberAbove(newLayers[len(newLayers)].getBotDepth())
        botLayer = self.getVelocityLayer(botLayerNum)
        outLayers = ArrayList()
        outLayers.addAll(self.layer)
        try:
            if matchTop:
                newLayers[0] = VelocityLayer(newLayers[0].getLayerNum(), newLayers[0].getTopDepth(), newLayers[0].getBotDepth(), topLayer.evaluateAt(newLayers[0].getTopDepth(), 'P'), newLayers[0].getBotPVelocity(), topLayer.evaluateAt(newLayers[0].getTopDepth(), 'S'), newLayers[0].getBotSVelocity(), newLayers[0].getTopDensity(), newLayers[0].getBotDensity(), newLayers[0].getTopQp(), newLayers[0].getBotQp(), newLayers[0].getTopQs(), newLayers[0].getBotQs())
            if matchBot:
                newLayers[len(newLayers)] = VelocityLayer(end.getLayerNum(), end.getTopDepth(), end.getBotDepth(), end.getTopPVelocity(), botLayer.evaluateAt(newLayers[len(newLayers)].getBotDepth(), 'P'), end.getTopSVelocity(), botLayer.evaluateAt(newLayers[len(newLayers)].getBotDepth(), 'S'), end.getTopDensity(), end.getBotDensity(), end.getTopQp(), end.getBotQp(), end.getTopQs(), end.getBotQs())
        except NoSuchMatPropException as e:
            raise RuntimeException(e)
        if topLayer.getBotDepth() > newLayers[0].getTopDepth():
            try:
                topLayer = VelocityLayer(topLayer.getLayerNum(), topLayer.getTopDepth(), newLayers[0].getTopDepth(), topLayer.getTopPVelocity(), topLayer.evaluateAt(newLayers[0].getTopDepth(), 'P'), topLayer.getTopSVelocity(), topLayer.evaluateAt(newLayers[0].getTopDepth(), 'S'), topLayer.getTopDensity(), topLayer.getBotDensity())
                outLayers.set(topIndex, topLayer)
            except NoSuchMatPropException as e:
                raise RuntimeException(e)
            newVLayer.setTopPVelocity(topLayer.getBotPVelocity())
            newVLayer.setTopSVelocity(topLayer.getBotSVelocity())
            newVLayer.setTopDepth(topLayer.getBotDepth())
            outLayers.add(topLayerNum + 1, newVLayer)
            botLayerNum += 1
            topLayerNum += 1
        if botLayer.getBotDepth() > newLayers[len(newLayers)].getBotDepth():
            try:
                botLayer.setBotPVelocity(botLayer.evaluateAt(newLayers[len(newLayers)].getBotDepth(), 'P'))
                botLayer.setBotSVelocity(botLayer.evaluateAt(newLayers[len(newLayers)].getBotDepth(), 'S'))
                botLayer.setBotDepth(newLayers[len(newLayers)].getBotDepth())
            except NoSuchMatPropException as e:
                System.err.println("Caught NoSuchMatPropException: " + e.getMessage())
                e.printStackTrace()
            newVLayer.setTopPVelocity(botLayer.getBotPVelocity())
            newVLayer.setTopSVelocity(botLayer.getBotSVelocity())
            newVLayer.setTopDepth(botLayer.getBotDepth())
            outLayers.add(botLayerNum + 1, newVLayer)
            botLayerNum += 1
        i = topLayerNum
        while i <= botLayerNum:
            outLayers.remove(topLayerNum)
            i += 1
        i = 0
        while len(newLayers):
            outLayers.add(topLayerNum + i, newLayers[i])
            i += 1
        outVMod = VelocityModel(name, self.getRadiusOfEarth(), self.getMohoDepth(), self.getCmbDepth(), self.getIocbDepth(), self.getMinRadius(), self.getMaxRadius(), self.getSpherical(), outLayers)
        outVMod.fixDisconDepths()
        outVMod.validate()
        return outVMod

    # originally: printGMT(String filename) 
    @overloaded
    def printGMT(self, filename):
        """ prints out the velocity model into a file in a form suitable 
        for plotting with GMT """
        psFile = str()
        if filename.endsWith(".gmt"):
            psFile = filename.substring(0, 4 - len(filename)) + ".ps"
        else:
            psFile = filename + ".ps"
        dos = PrintWriter(BufferedWriter(FileWriter(filename)))
        dos.println("#!/bin/sh")
        dos.println("#\n# This script will plot the " + self.getModelName() + " velocity model using GMT. If you want to\n" + "#use this as a data file for psxy in another script, delete these" + "\n# first lines, as well as the last line.\n#")
        dos.println("/bin/rm -f " + psFile + "\n")
        maxVel = 0
        for vLayer in layer:
            if vLayer.getTopPVelocity() > maxVel:
                maxVel = vLayer.getTopPVelocity()
            if vLayer.getBotPVelocity() > maxVel:
                maxVel = vLayer.getBotPVelocity()
            if vLayer.getTopSVelocity() > maxVel:
                maxVel = vLayer.getTopSVelocity()
            if vLayer.getBotSVelocity() > maxVel:
                maxVel = vLayer.getBotSVelocity()
        maxVel *= 1.05  #make little bit larger
        dos.println("PCOLOR=0/0/255")
        dos.println("SCOLOR=255/0/0")
        dos.println()
        dos.println("psbasemap -JX6i/-9i -P -R0/" + maxVel + "/0/" + self.getMaxRadius() + " -B1a2:'Velocity (km/s)':/200a400:'Depth (km)':/:.'" + self.getModelName() + "':WSen  -K > " + psFile)
        dos.println()
        dos.println("psxy -JX -P -R -W2p,${PCOLOR} -: -m -O -K >> " + psFile + " <<END")
        printGMTforP(dos)
        dos.println("END\n")
        dos.println("psxy -JX -P -R -W2p,${SCOLOR} -: -m -O >> " + psFile + " <<END")
        printGMTforS(dos)
        dos.println("END\n")
        dos.close()

    # originally: printGMT(PrintWriter dos) 
    @printGMT.register(object, PrintWriter)
    def printGMT_0(self, dos):
        """ prints out the velocity model into a file in a form suitable 
        for plotting with GMT """
        dos.println("> P velocity for " + self.modelName + "  below")
        printGMTforP(dos)
        dos.println("> S velocity for " + self.modelName + "  below")
        printGMTforP(dos)

    def printGMTforP(self, dos):
        """ generated source for method printGMTforP """
        pVel = -1.0
        layerNum = 0
        while layerNum < self.getNumLayers():
            if currVelocityLayer.getTopPVelocity() != pVel:
                dos.println(float(currVelocityLayer.getTopDepth()) + " " + float(currVelocityLayer.getTopPVelocity()))
            dos.println(float(currVelocityLayer.getBotDepth()) + " " + float(currVelocityLayer.getBotPVelocity()))
            pVel = currVelocityLayer.getBotPVelocity()
            layerNum += 1

    def printGMTforS(self, dos):
        """ generated source for method printGMTforS """
        sVel = -1.0
        layerNum = 0
        while layerNum < self.getNumLayers():
            if currVelocityLayer.getTopSVelocity() != sVel:
                dos.println(float(currVelocityLayer.getTopDepth()) + " " + float(currVelocityLayer.getTopSVelocity()))
            dos.println(float(currVelocityLayer.getBotDepth()) + " " + float(currVelocityLayer.getBotSVelocity()))
            sVel = currVelocityLayer.getBotSVelocity()
            layerNum += 1

    def validate(self):
        """ Performs internal consistency checks on the velocity model. """
        currVelocityLayer = VelocityLayer()
        prevVelocityLayer = VelocityLayer()
        #/* is radiusOfEarth positive? */
        if self.radiusOfEarth <= 0.0:
            System.err.println("Radius of earth is not positive. radiusOfEarth = " + self.radiusOfEarth)
            return False
    	#/* is mohoDepth non-negative? */
        if self.mohoDepth < 0.0:
            System.err.println("mohoDepth is not non-negative. mohoDepth = " + self.mohoDepth)
            return False
        #/* is cmbDepth >= mohoDepth? */
        if self.cmbDepth < self.mohoDepth:
            System.err.println("cmbDepth < mohoDepth. cmbDepth = " + self.cmbDepth + " mohoDepth = " + self.mohoDepth)
            return False
        #/* is cmbDepth positive? */
        if self.cmbDepth <= 0.0:
            System.err.println("cmbDepth is not positive. cmbDepth = " + self.cmbDepth)
            return False
        #/* is iocbDepth >= cmbDepth? */
        if self.iocbDepth < self.cmbDepth:
            System.err.println("iocbDepth < cmbDepth. iocbDepth = " + self.iocbDepth + " cmbDepth = " + self.cmbDepth)
            return False
        #/* is iocbDepth positive? */
        if self.iocbDepth <= 0.0:
            System.err.println("iocbDepth is not positive. iocbDepth = " + self.iocbDepth)
            return False
        #/* is minRadius non-negative? */
        if self.minRadius < 0.0:
            System.err.println("minRadius is not non-negative. minRadius = " + self.minRadius)
            return False
        #/* is maxRadius positive? */
        if self.maxRadius <= 0.0:
            System.err.println("maxRadius is not positive. maxRadius = " + self.maxRadius)
            return False
        #/* is maxRadius > minRadius? */
        if self.maxRadius <= self.minRadius:
            System.err.println("maxRadius <= minRadius. maxRadius = " + self.maxRadius + " minRadius = " + self.minRadius)
            return False
        currVelocityLayer = self.getVelocityLayer(0)
        prevVelocityLayer = VelocityLayer(0, currVelocityLayer.getTopDepth(), currVelocityLayer.getTopDepth(), currVelocityLayer.getTopPVelocity(), currVelocityLayer.getTopPVelocity(), currVelocityLayer.getTopSVelocity(), currVelocityLayer.getTopSVelocity(), currVelocityLayer.getTopDensity(), currVelocityLayer.getTopDensity())
        layerNum = 0
        while layerNum < self.getNumLayers():
            currVelocityLayer = self.getVelocityLayer(layerNum)
            if prevVelocityLayer.getBotDepth() != currVelocityLayer.getTopDepth():
            #* There is a gap in the velocity model!
                System.err.println("There is a gap in the velocity model " + "between layers " + (layerNum - 1) + " and " + layerNum)
                System.err.println("prevVelocityLayer=" + prevVelocityLayer)
                System.err.println("currVelocityLayer=" + currVelocityLayer)
                return False
            if currVelocityLayer.getBotDepth() == currVelocityLayer.getTopDepth():
            #   more redundant comments in the original java
                System.err.println("There is a zero thickness layer in the " + "velocity model at layer " + layerNum)
                System.err.println("prevVelocityLayer=" + prevVelocityLayer)
                System.err.println("currVelocityLayer=" + currVelocityLayer)
                return False
            if currVelocityLayer.getTopPVelocity() <= 0.0 or currVelocityLayer.getBotPVelocity() <= 0.0:
                System.err.println("There is a negative P velocity layer in the " + "velocity model at layer " + layerNum)
                return False
            if currVelocityLayer.getTopSVelocity() < 0.0 or currVelocityLayer.getBotSVelocity() < 0.0:
                System.err.println("There is a negative S velocity layer in the " + "velocity model at layer " + layerNum)
                return False
            if (currVelocityLayer.getTopPVelocity() != 0.0 and currVelocityLayer.getBotPVelocity() == 0.0) or (currVelocityLayer.getTopPVelocity() == 0.0 and currVelocityLayer.getBotPVelocity() != 0.0):
                System.err.println("There is a layer that goes to zero P velocity " + "without a discontinuity in the " + "velocity model at layer " + layerNum + "\nThis would cause a divide by zero within this " + "depth range. Try making the velocity small, followed by a " + "discontinuity to zero velocity.")
                return False
            if (currVelocityLayer.getTopSVelocity() != 0.0 and currVelocityLayer.getBotSVelocity() == 0.0) or (currVelocityLayer.getTopSVelocity() == 0.0 and currVelocityLayer.getBotSVelocity() != 0.0):
                System.err.println("There is a layer that goes to zero S velocity " + "without a discontinuity in the " + "velocity model at layer " + layerNum + "\nThis would cause a divide by zero within this " + "depth range. Try making the velocity small, followed by a " + "discontinuity to zero velocity.")
                return False
            prevVelocityLayer = currVelocityLayer
            layerNum += 1
        return True

    def __str__(self):
        """ generated source for method toString """
        desc = "modelName=" + self.modelName + "\n" + "\n radiusOfEarth=" + self.radiusOfEarth + "\n mohoDepth=" + self.mohoDepth + "\n cmbDepth=" + self.cmbDepth + "\n iocbDepth=" + self.iocbDepth + "\n minRadius=" + self.minRadius + "\n maxRadius=" + self.maxRadius + "\n spherical=" + self.spherical
        desc += "\ngetNumLayers()=" + self.getNumLayers() + "\n"
        return desc

    def print_(self):
        """ generated source for method print_ """
        i = 0
        while i < self.getNumLayers():
            print self.getVelocityLayer(i)
            i += 1

    @classmethod
    def getModelNameFromFileName(cls, filename):
        """ generated source for method getModelNameFromFileName """
        j = filename.lastIndexOf(System.getProperty("file.separator"))
        modelFilename = filename.substring(j + 1)
        modelName = modelFilename
        if modelFilename.endsWith("tvel"):
            modelName = modelFilename.substring(0, 5 - len(modelFilename))
        elif modelFilename.endsWith(".nd"):
            modelName = modelFilename.substring(0, 3 - len(modelFilename))
        elif modelFilename.startsWith("GB."):
            modelName = modelFilename.substring(3, len(modelFilename))
        else:
            modelName = modelFilename
        return modelName

    @classmethod
    def readVelocityFile(cls, filename, fileType):
        """ Reads in a velocity file. The type of file is determined by the 
	fileType parameter. Calls readTVelFile or readNDFile.
     	 
     	@exception VelocityModelException
     	if the type of file cannot be determined. """
        if fileType == None or fileType == "":
            if filename.endsWith(".nd"):
                fileType = ".nd"
            elif filename.endsWith(".tvel"):
                fileType = ".tvel"
        if fileType.startsWith("."):
            fileType = fileType.substring(1, len(fileType))
        f = File(filename)
        if not f.exists() and not filename.endsWith("." + fileType) and File(filename + "." + fileType).exists():
            f = File(filename + "." + fileType)
        vMod = VelocityModel()
        if fileType.lower() == "nd".lower():
            vMod = readNDFile(f)
        elif fileType.lower() == "tvel".lower():
            vMod = readTVelFile(f)
        else:
            raise VelocityModelException("What type of velocity file, .tvel or .nd?")
        vMod.fixDisconDepths()
        return vMod



     #* This method reads in a velocity model from a "tvel" ASCII text file. The
     #* name of the model file for model "modelname" should be "modelname.tvel".
     #* The format of the file is: comment line - generally info about the P
     #* velocity model comment line - generally info about the S velocity model
     #* depth pVel sVel Density depth pVel sVel Density . . .
     #* 
     #* The velocities are assumed to be linear between sample points. Because
     #* this type of model file doesn't give complete information we make the
     #* following assumptions: modelname - from the filename, with ".tvel"
     #* dropped if present radiusOfEarth - the largest depth in the model
     #* meanDensity - 5517.0 G - 6.67e-11
     #* 
     #* Also, because this method makes use of the string tokenizer, comments are
     #* allowed. # as well as // signify that the rest of the line is a comment.
     #* C style slash-star comments are also allowed.
     #* 
     #* @exception VelocityModelException
     #*                occurs if an EOL should have been read but wasn't. This
     #*                may indicate a poorly formatted tvel file.

    @classmethod
    @overloaded
    def readTVelFile(cls, file_):
        """ generated source for method readTVelFile """
        fileIn = FileReader(file_)
        vmod = cls.readTVelFile(fileIn, cls.getModelNameFromFileName(file_.__name__))
        # NB right here it should say readTVelFile_0, looking at the java.
        fileIn.close()
        return vmod

    @classmethod
    @readTVelFile.register(object, Reader, str)
    def readTVelFile_0(cls, in_, modelName):
        """ generated source for method readTVelFile_0 """
        tokenIn = StreamTokenizer(in_)
        tokenIn.commentChar('#') 	# '#' means ignore to end of line
        tokenIn.slashStarComments(True)	# '/*...*/' means a comment
        tokenIn.slashSlashComments(True)# '//' means ignore to end of line
        tokenIn.eolIsSignificant(True)	# end of line is important
        tokenIn.parseNumbers()		# Differentiate between words and numbers. 
        				# Note 1.1e3 is considered a string 
        				# instead of a number.
	# Read until we get 2 end of lines.
        while tokenIn.nextToken() != StreamTokenizer.TT_EOL:
            pass
        while tokenIn.nextToken() != StreamTokenizer.TT_EOL:
            pass
    	# Now we have passed both comment lines and are ready to read the
        # velocity model.
         
        # Some temporary variables to store the current line from the file and
        # the current layer.
        myLayerNumber = 0
        tempLayer = VelocityLayer()
        topDepth = float()
        topPVel = float()
        topSVel = float()
        topDensity = float()
        botDepth = float()
        botPVel = float()
        botSVel = float()
        botDensity = float()
        # Preload the first line of the model
        tokenIn.nextToken()
        topDepth = tokenIn.nval
        tokenIn.nextToken()
        topPVel = tokenIn.nval
        tokenIn.nextToken()
        topSVel = tokenIn.nval
        if topSVel > topPVel:
            raise VelocityModelException("S velocity, " + topSVel + " at depth " + topDepth + " is greater than the P velocity, " + topPVel)
        tokenIn.nextToken()
        if tokenIn.ttype != StreamTokenizer.TT_EOL:
        # density is not used and so is optional
            topDensity = tokenIn.nval
            tokenIn.nextToken()
        else:
            topDensity = 5571.0
        if tokenIn.ttype != StreamTokenizer.TT_EOL:
        # this token should be an EOL, if not
            raise VelocityModelException("Should have found an EOL but didn't" + " Layer=" + myLayerNumber + " tokenIn=" + tokenIn)
        else:
            tokenIn.nextToken()
        layers = ArrayList()
        while tokenIn.ttype != StreamTokenizer.TT_EOF:
        # Loop until we hit the end of file
            botDepth = tokenIn.nval
            tokenIn.nextToken()
            botPVel = tokenIn.nval
            tokenIn.nextToken()
            botSVel = tokenIn.nval
            if botSVel > botPVel:
                raise VelocityModelException("S velocity, " + botSVel + " at depth " + botDepth + " is greater than the P velocity, " + botPVel)
            tokenIn.nextToken()
            if tokenIn.ttype != StreamTokenizer.TT_EOL:
            # density is not used and is optional
                botDensity = tokenIn.nval
                tokenIn.nextToken()
            else:
                botDensity = topDensity
            tempLayer = VelocityLayer(myLayerNumber, topDepth, botDepth, topPVel, botPVel, topSVel, botSVel, topDensity, botDensity)
            topDepth = botDepth
            topPVel = botPVel
            topSVel = botSVel
            topDensity = botDensity
            if tokenIn.ttype != StreamTokenizer.TT_EOL:
            # this token should be an EOL, if not
                raise VelocityModelException("Should have found an EOL but didn't" + " Layer=" + myLayerNumber + " tokenIn=" + tokenIn)
            else:
                tokenIn.nextToken()
            if tempLayer.getTopDepth() != tempLayer.getBotDepth():
            # Don't use zero thickness layers, first order discontinuities
            # are taken care of by storing top and bottom depths.
                layers.add(tempLayer)
                myLayerNumber += 1
        radiusOfEarth = topDepth
        maxRadius = topDepth	# I assume that this is a whole earth model
        			# so the maximum depth is equal to the
        			# maximum radius is equal to the earth radius.
        return VelocityModel(modelName, radiusOfEarth, cls.DEFAULT_MOHO, cls.DEFAULT_CMB, cls.DEFAULT_IOCB, 0, maxRadius, True, layers)

 
     # This method reads in a velocity model from a "nd" ASCII text file, the
     # format used by Xgbm. The name of the model file for model "modelname"
     # should be "modelname.nd". The format of the file is: depth pVel sVel
     # Density Qp Qs depth pVel sVel Density Qp Qs . . . with each major
     # boundary separated with a line with "mantle", "outer-core" or
     # "inner-core". "moho", "cmb" and "icocb" are allowed as synonyms respectively.
     # This feature makes phase interpretation much easier to
     # code. Also, as they are not needed for travel time calculations, the
     # density, Qp and Qs may be omitted.
     # 
     # The velocities are assumed to be linear between sample points. Because
     # this type of model file doesn't give complete information we make the
     # following assumptions: 
     # 
     # modelname - from the filename, with ".nd" dropped, if present 
     # 
     # radiusOfEarth - the largest depth in the model
     # 
     # Also, because this method makes use of the string tokenizer, comments are
     # allowed. # as well as // signify that the rest of the line is a comment.
     # C style slash-star comments are also allowed.
     # 
     # @exception VelocityModelException
     #                occurs if an EOL should have been read but wasn't. This
     #                may indicate a poorly formatted model file.
    @classmethod
    @overloaded
    def readNDFile(cls, file_):
        """ generated source for method readNDFile """
        fileIn = FileReader(file_)
        vmod = cls.readNDFile(fileIn, cls.getModelNameFromFileName(file_.__name__))
        fileIn.close()
        return vmod

    @classmethod
    @readNDFile.register(object, Reader, str)
    def readNDFile_0(cls, in_, modelName):
        """ generated source for method readNDFile_0 """
        tokenIn = StreamTokenizer(in_)
        tokenIn.commentChar('#')	#'#' means ignore to end of line
        tokenIn.slashStarComments(True)	#'/*...*/' means a comment
        tokenIn.slashSlashComments(True)#'//' means ignore to end of line
        tokenIn.eolIsSignificant(True)	#end of line is important
        tokenIn.parseNumbers()	# 
                                # Differentiate between words and numbers. Note
                                # 1.1e3 is considered a string instead of a
                                # number.
        # Some temporary variables to store the current line from the file and
        # the current layer.
        myLayerNumber = 0
        tempLayer = VelocityLayer()
        topDepth = float()
        topPVel = float()
        topSVel = float()
        topDensity = 2.6
        topQp = 1000
        topQs = 2000
        botDepth = float()
        botPVel = float()
        botSVel = float()
        botDensity = topDensity
        botQp = topQp
        botQs = topQs
        # Preload the first line of the model
        tokenIn.nextToken()
        topDepth = tokenIn.nval
        tokenIn.nextToken()
        topPVel = tokenIn.nval
        tokenIn.nextToken()
        topSVel = tokenIn.nval
        if topSVel > topPVel:
            raise VelocityModelException("S velocity, " + topSVel + " at depth " + topDepth + " is greater than the P velocity, " + topPVel)
        tokenIn.nextToken()
        if tokenIn.ttype != StreamTokenizer.TT_EOL:
        # density is not used and so is optional
            topDensity = tokenIn.nval
            tokenIn.nextToken()
            if tokenIn.ttype != StreamTokenizer.TT_EOL:
            # Qp is not used and so is optional
                topQp = tokenIn.nval
                tokenIn.nextToken()
                if tokenIn.ttype != StreamTokenizer.TT_EOL:
                # Qs is not used and so is optional
                    topQs = tokenIn.nval
                    tokenIn.nextToken()
        if tokenIn.ttype != StreamTokenizer.TT_EOL:
        # this token should be an EOL, if not
            raise VelocityModelException("Should have found an EOL but didn't" + " Layer=" + myLayerNumber + " tokenIn=" + tokenIn)
        else:
            tokenIn.nextToken()
        mohoDepth = cls.DEFAULT_MOHO
        cmbDepth = cls.DEFAULT_CMB
        iocbDepth = cls.DEFAULT_IOCB
        layers = ArrayList()
        while tokenIn.ttype != StreamTokenizer.TT_EOF:
        # Loop until we hit the end of file
            if tokenIn.ttype == StreamTokenizer.TT_WORD:
                if tokenIn.sval.lower() == "mantle".lower() or tokenIn.sval.lower() == "moho".lower():
                    mohoDepth = topDepth	# moho
                if tokenIn.sval.lower() == "outer-core".lower() or tokenIn.sval.lower() == "cmb".lower():
                    cmbDepth = topDepth		# Core Mantle Boundary 
                if tokenIn.sval.lower() == "inner-core".lower() or tokenIn.sval.lower() == "icocb".lower():
                    iocbDepth = topDepth	# Inner Outer Core Boundary
                while tokenIn.ttype != StreamTokenizer.TT_EOL:
                    tokenIn.nextToken()
                tokenIn.nextToken()
                continue 
            botDepth = tokenIn.nval
            tokenIn.nextToken()
            botPVel = tokenIn.nval
            tokenIn.nextToken()
            botSVel = tokenIn.nval
            if botSVel > botPVel:
                raise VelocityModelException("S velocity, " + botSVel + " at depth " + botDepth + " is greater than the P velocity, " + botPVel)
            tokenIn.nextToken()
            if tokenIn.ttype != StreamTokenizer.TT_EOL:
            # density is not used and so is optional
                botDensity = tokenIn.nval
                tokenIn.nextToken()
                if tokenIn.ttype != StreamTokenizer.TT_EOL:
                # Qp is not used and so is optional
                    botQp = tokenIn.nval
                    tokenIn.nextToken()
                    if tokenIn.ttype != StreamTokenizer.TT_EOL:
                    # Qs is not used and so is optional
                        botQs = tokenIn.nval
                        tokenIn.nextToken()
            tempLayer = VelocityLayer(myLayerNumber, topDepth, botDepth, topPVel, botPVel, topSVel, botSVel, topDensity, botDensity, topQp, botQp, topQs, botQs)
            topDepth = botDepth
            topPVel = botPVel
            topSVel = botSVel
            topDensity = botDensity
            topQp = botQp
            topQs = botQs
            if tokenIn.ttype != StreamTokenizer.TT_EOL:
            # this token should be an EOL, if not
                raise VelocityModelException("Should have found an EOL but didn't" + " Layer=" + myLayerNumber + " tokenIn=" + tokenIn)
            else:
                tokenIn.nextToken()
            if tempLayer.getTopDepth() != tempLayer.getBotDepth():
            # Don't use zero thickness layers, first order discontinuities
            # are taken care of by storing top and bottom depths.
                layers.add(tempLayer)
                myLayerNumber += 1               
        radiusOfEarth = topDepth
        maxRadius = topDepth	#I assume that this is a whole earth model
        			# so the maximum depth is equal to the
        			# maximum radius is equal to the earth radius.
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
            aboveLayer = self.getVelocityLayer(layerNum)
            belowLayer = self.getVelocityLayer(layerNum + 1)
            if aboveLayer.getBotPVelocity() != belowLayer.getTopPVelocity() or aboveLayer.getBotSVelocity() != belowLayer.getTopSVelocity():	# a discontinuity
            
                if Math.abs(self.mohoDepth - aboveLayer.getBotDepth()) < mohoMin:
                    tempMohoDepth = aboveLayer.getBotDepth()
                    mohoMin = Math.abs(self.mohoDepth - aboveLayer.getBotDepth())
                if Math.abs(self.cmbDepth - aboveLayer.getBotDepth()) < cmbMin:
                    tempCmbDepth = aboveLayer.getBotDepth()
                    cmbMin = Math.abs(self.cmbDepth - aboveLayer.getBotDepth())
                if aboveLayer.getBotSVelocity() == 0.0 and belowLayer.getTopSVelocity() > 0.0 and Math.abs(self.iocbDepth - aboveLayer.getBotDepth()) < iocbMin:
                    tempIocbDepth = aboveLayer.getBotDepth()
                    iocbMin = Math.abs(self.iocbDepth - aboveLayer.getBotDepth())
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
        spherical = False
        layers = ArrayList(self.vectorLength)
        i = 0
        while i < self.getNumLayers():
            oldLayer = self.getVelocityLayer(i)
            newLayer = VelocityLayer(i, self.radiusOfEarth * Math.log(oldLayer.getTopDepth() / self.radiusOfEarth), self.radiusOfEarth * Math.log(oldLayer.getBotDepth() / self.radiusOfEarth), self.radiusOfEarth * oldLayer.getTopPVelocity() / oldLayer.getTopDepth(), self.radiusOfEarth * oldLayer.getBotPVelocity() / oldLayer.getBotDepth(), self.radiusOfEarth * oldLayer.getTopSVelocity() / oldLayer.getTopDepth(), self.radiusOfEarth * oldLayer.getBotSVelocity() / oldLayer.getBotDepth())
            layers.add(newLayer)
            i += 1
        return VelocityModel(self.modelName, self.getRadiusOfEarth(), self.getMohoDepth(), self.getCmbDepth(), self.getIocbDepth(), self.getMinRadius(), self.getMaxRadius(), spherical, layers)

