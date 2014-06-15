#!/usr/bin/env python
""" generated source for module TauP_Create """
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
# package: edu.sc.seis.TauP
import java.io.BufferedInputStream

import java.io.File

import java.io.FileInputStream

import java.io.IOException

import java.io.PrintWriter

import java.util.Properties

# 
#  * TauP_Create - Re-implementation of the seismic travel time calculation method
#  * described in "The Computation of Seismic Travel Times" by Buland and Chapman,
#  * BSSA vol. 73, No. 5, October 1983, pp 1271-1302. This creates the
#  * SlownessModel and tau branches and saves them for later use.
#  * 
#  * @version 1.1.3 Wed Jul 18 15:00:35 GMT 2001
#  * 
#  * 
#  * 
#  * @author H. Philip Crotwell
#  
class TauP_Create(object):
    """ generated source for class TauP_Create """
    verbose = False
    modelFilename = "iasp91.tvel"
    overlayModelFilename = None
    velFileType = "tvel"
    overlayVelFileType = str()
    directory = "."
    sMod = SlownessModel()
    vMod = VelocityModel()
    overlayVMod = VelocityModel()
    tMod = TauModel()
    DEBUG = False
    GUI = False
    toolProps = Properties()
    plotVmod = False
    plotVmodFilename = str()
    plotSmod = False
    plotSmodFilename = str()
    plotTmod = False
    plotTmodFilename = str()

    #  constructor 
    def __init__(self):
        """ generated source for method __init__ """
        Alert.setGUI(self.GUI)
        try:
            self.toolProps = PropertyLoader.load()
        except Exception as e:
            Alert.warning("Unable to load properties, using defaults.", e.getMessage())
            self.toolProps = Properties()

    #  Accessor methods 
    def setDirectory(self, directory):
        """ generated source for method setDirectory """
        self.directory = directory

    def getDirectory(self):
        """ generated source for method getDirectory """
        return self.directory

    def setModelFilename(self, modelFilename):
        """ generated source for method setModelFilename """
        self.modelFilename = modelFilename

    def getModelFilename(self):
        """ generated source for method getModelFilename """
        return self.modelFilename

    def getVelFileType(self):
        """ generated source for method getVelFileType """
        return self.velFileType

    def setVelFileType(self, type_):
        """ generated source for method setVelFileType """
        self.velFileType = type_

    def setDEBUG(self, DEBUG):
        """ generated source for method setDEBUG """
        self.DEBUG = DEBUG

    def getDEBUG(self):
        """ generated source for method getDEBUG """
        return self.DEBUG

    def setVelocityModel(self, vMod):
        """ generated source for method setVelocityModel """
        self.vMod = vMod

    def setMinDeltaP(self, minDeltaP):
        """ generated source for method setMinDeltaP """
        self.toolProps.setProperty("taup.create.minDeltaP", "" + minDeltaP)

    def setMaxDeltaP(self, maxDeltaP):
        """ generated source for method setMaxDeltaP """
        self.toolProps.setProperty("taup.create.maxDeltaP", "" + maxDeltaP)

    def setMaxDepthInterval(self, maxDepthInterval):
        """ generated source for method setMaxDepthInterval """
        self.toolProps.setProperty("taup.create.maxDepthInterval", "" + maxDepthInterval)

    def setMaxRangeInterval(self, maxRangeInterval):
        """ generated source for method setMaxRangeInterval """
        self.toolProps.setProperty("taup.create.maxRangeInterval", "" + maxRangeInterval)

    def setMaxInterpError(self, maxInterpError):
        """ generated source for method setMaxInterpError """
        self.toolProps.setProperty("taup.create.maxInterpError", "" + maxInterpError)

    def setAllowInnerCoreS(self, allowInnerCoreS):
        """ generated source for method setAllowInnerCoreS """
        self.toolProps.setProperty("taup.create.allowInnerCoreS", "" + allowInnerCoreS)

    def printUsage(self):
        """ generated source for method printUsage """
        className = self.__class__.__name__
        className = className.substring(className.lastIndexOf('.') + 1, len(className))
        print "Usage: " + className.lower() + " [arguments]"
        print "  or, for purists, java " + self.__class__.__name__ + " [arguments]"
        print "\nArguments are:"
        print "\n   To specify the velocity model:"
        print "-nd modelfile       -- \"named discontinuities\" velocity file"
        print "-tvel modelfile     -- \".tvel\" velocity file, ala ttimes\n"
        print "--vplot file.gmt     -- plot velocity as a GMT script\n"

        # The following commentary is from the java file:
        # \\plotting sMod and tMod not yet implemented
        # \\System.out.println("--splot file.gmt     -- plot slowness as a GMT script\n");
       # \\System.out.println("--tplot file.gmt     -- plot tau as a GMT script\n");
        print "-debug              -- enable debugging output\n" + "-verbose            -- enable verbose output\n" + "-version            -- print the version\n" + "-help               -- print this out, but you already know that!\n\n"

    @classmethod
    def dashEquals(cls, argName, arg):
        """ generated source for method dashEquals """
        return TauP_Time.dashEquals(argName, arg)

    def parseCmdLineArgs(self, args):
        """ generated source for method parseCmdLineArgs """
        i = 0
        noComprendoArgs = [None]*
        numNoComprendoArgs = 0
        __numNoComprendoArgs_0 = numNoComprendoArgs
        numNoComprendoArgs += 1
        __numNoComprendoArgs_1 = numNoComprendoArgs
        numNoComprendoArgs += 1
        __numNoComprendoArgs_2 = numNoComprendoArgs
        numNoComprendoArgs += 1
        __numNoComprendoArgs_3 = numNoComprendoArgs
        numNoComprendoArgs += 1
        while len(args):
            if self.dashEquals("help", args[i]):
                self.printUsage()
                noComprendoArgs[__numNoComprendoArgs_0] = args[i]
                return noComprendoArgs
            elif self.dashEquals("version", args[i]):
                print BuildVersion.getDetailedVersion()
                noComprendoArgs[__numNoComprendoArgs_1] = args[i]
                return noComprendoArgs
            elif self.dashEquals("debug", args[i]):
                self.verbose = True
                self.DEBUG = True
            elif self.dashEquals("verbose", args[i]):
                self.verbose = True
            elif self.dashEquals("gui", args[i]):
                self.GUI = True
            elif i < len(args) and self.dashEquals("p", args[i]):
                try:
                    self.toolProps.load(BufferedInputStream(FileInputStream(args[i + 1])))
                    i += 1
                except IOException as e:
                    noComprendoArgs[__numNoComprendoArgs_2] = args[i + 1]
            elif i < len(args) and self.dashEquals("nd", args[i]):
                self.velFileType = "nd"
                parseFileName(args[i + 1])
                i += 1
            elif i < len(args) and self.dashEquals("tvel", args[i]):
                self.velFileType = "tvel"
                parseFileName(args[i + 1])
                i += 1
            elif i < len(args) and self.dashEquals("overlayND", args[i]):
                self.overlayVelFileType = "nd"
                self.overlayModelFilename = args[i + 1]
                i += 1
            elif i < len(args) and self.dashEquals("vplot", args[i]):
                self.plotVmod = True
                self.plotVmodFilename = args[i + 1]
                i += 1
            elif i < len(args) and self.dashEquals("splot", args[i]):
                self.plotSmod = True
                self.plotSmodFilename = args[i + 1]
                i += 1
            elif i < len(args) and self.dashEquals("tplot", args[i]):
                self.plotTmod = True
                self.plotTmodFilename = args[i + 1]
                i += 1
            elif args[i].startsWith("GB."):
                self.velFileType = "nd"
                parseFileName(args[i])
            elif args[i].endsWith(".nd"):
                self.velFileType = "nd"
                parseFileName(args[i])
            elif args[i].endsWith(".tvel"):
                self.velFileType = "tvel"
                parseFileName(args[i])
            else:
                # I don't know how to interpret this argument, so pass it
                # java: noComprendoArgs[numNoComprendoArgs++] = args[i];
                noComprendoArgs[__numNoComprendoArgs_3] = args[i]
            i += 1
        if self.modelFilename == None:
            print "Velocity model not specified, use one of -nd or -tvel"
            self.printUsage()
            # bad, should do something else here...
            System.exit(1)
        if numNoComprendoArgs > 0:
            System.arraycopy(noComprendoArgs, 0, temp, 0, numNoComprendoArgs)
            return temp
        else:
            return [None]*0

    @classmethod
    def main(cls, args):
        """ generated source for method main """
        print "TauP_Create starting..."
        tauPCreate = TauP_Create()
        noComprendoArgs = tauPCreate.parseCmdLineArgs(args)
        TauP_Time.printNoComprendoArgs(noComprendoArgs)
        try:
            tauPCreate.loadVMod()
            tauPCreate.start()
            print "Done!"
        except IOException as e:
            print "Tried to read!\n Caught IOException " + e.getMessage() + "\nCheck that the file exists and is readable."
        except VelocityModelException as e:
            print "Caught VelocityModelException " + e.getMessage() + "\nCheck your velocity model."

    def parseFileName(self, modelFilename):
        """ generated source for method parseFileName """
        j = modelFilename.lastIndexOf(System.getProperty("file.separator"))
        self.modelFilename = modelFilename.substring(j + 1)
        if j == -1:
            self.directory = "."
        else:
            self.directory = modelFilename.substring(0, j)

    def loadVMod(self):
        """ generated source for method loadVMod """
        file_sep = System.getProperty("file.separator")
        filename = self.directory + file_sep + self.modelFilename
        f = File(filename)
        if self.verbose:
            print "filename =" + self.directory + file_sep + self.modelFilename
        self.vMod = VelocityModel.readVelocityFile(filename, self.velFileType)
        if self.vMod == None:
            self.vMod = TauModelLoader.loadVelocityModel(self.modelFilename)
        if self.vMod == None:
            raise IOException("Velocity model file not found: " + self.modelFilename + ", tried internally and from file: " + f)
        if self.verbose:
            print "Done reading velocity model."
            print "Radius of model " + self.vMod.getModelName() + " is " + self.vMod.getRadiusOfEarth()
        if self.overlayModelFilename != None:
            if self.DEBUG:
                print "orig model: " + self.vMod
            self.overlayVMod = VelocityModel.readVelocityFile(self.directory + file_sep + self.overlayModelFilename, self.overlayVelFileType)
            self.vMod = self.vMod.replaceLayers(self.overlayVMod.getLayers(), self.overlayVMod.getModelName(), True, True)
        if self.DEBUG:
            print "velocity mode: " + self.vMod
        return self.vMod

    def createTauModel(self, vMod):
        """ generated source for method createTauModel """
        if vMod == None:
            raise IllegalArgumentException("vMod cannot be null")
        if not vMod.getSpherical():
            raise SlownessModelException("Flat slowness model not yet implemented.")
        SlownessModel.DEBUG = self.DEBUG
        self.sMod = SphericalSModel(vMod, Double.valueOf(self.toolProps.getProperty("taup.create.minDeltaP", "0.1")).doubleValue(), Double.valueOf(self.toolProps.getProperty("taup.create.maxDeltaP", "11.0")).doubleValue(), Double.valueOf(self.toolProps.getProperty("taup.create.maxDepthInterval", "115.0")).doubleValue(), Double.valueOf(self.toolProps.getProperty("taup.create.maxRangeInterval", "2.5")).doubleValue() * Math.PI / 180, Double.valueOf(self.toolProps.getProperty("taup.create.maxInterpError", "0.05")).doubleValue(), Boolean.valueOf(self.toolProps.getProperty("taup.create.allowInnerCoreS", "true")).booleanValue(), SlownessModel.DEFAULT_SLOWNESS_TOLERANCE)
        if self.verbose:
            print "Parameters are:"
            print "taup.create.minDeltaP = " + self.sMod.getMinDeltaP() + " sec / radian"
            print "taup.create.maxDeltaP = " + self.sMod.getMaxDeltaP() + " sec / radian"
            print "taup.create.maxDepthInterval = " + self.sMod.getMaxDepthInterval() + " kilometers"
            print "taup.create.maxRangeInterval = " + self.sMod.getMaxRangeInterval() + " degrees"
            print "taup.create.maxInterpError = " + self.sMod.getMaxInterpError() + " seconds"
            print "taup.create.allowInnerCoreS = " + self.sMod.isAllowInnerCoreS()
            print "Slow model " + " " + self.sMod.getNumLayers(True) + " P layers," + self.sMod.getNumLayers(False) + " S layers"
        if self.DEBUG:
            print self.sMod
        TauModel.DEBUG = self.DEBUG
        SlownessModel.DEBUG = self.DEBUG
        return TauModel(self.sMod)

    def start(self):
        """ generated source for method start """
        try:
            if self.plotVmod or self.plotSmod or self.plotTmod:
                if self.plotVmod:
                    self.vMod.printGMT(self.plotVmodFilename)
                    # not implemented yet: sMod and tMod plotting
            else:
                # j2py screwed up here, and ignored the following:
                # String file_sep =
                # System.getProperty("file.separator"); TauModel tMod
                # = createTauModel(vMod);


                if self.DEBUG:
                    print "Done calculating Tau branches."
                if self.DEBUG:
                    self.tMod.print_()
                    if self.directory == ".":
                    outFile = self.directory + file_sep + self.vMod.getModelName() + ".taup"
                else:
                    outFile = self.vMod.getModelName() + ".taup"
                self.tMod.writeModel(outFile)
                if self.verbose:
                    print "Done Saving " + outFile
        except IOException as e:
            print "Tried to write!\n Caught IOException " + e.getMessage() + "\nDo you have write permission in this directory?"
        except VelocityModelException as e:
            print "Caught VelocityModelException " + e.getMessage()
        finally:
            if self.verbose:
                print "Done!"


if __name__ == '__main__':
    import sys
    TauP_Create.main(sys.argv)

