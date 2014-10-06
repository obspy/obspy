#!/usr/bin/env python3

import os
import inspect
import argparse
from taupy.VelocityModel import VelocityModel
from taupy.SlownessModel import SlownessModel
from taupy.TauModel import TauModel


class TauP_Create(object):
    """TauP_Create - Re-implementation of the seismic travel time
    calculation method described in "The Computation of Seismic Travel
    Times" by Buland and Chapman, BSSA vol. 73, No. 5, October 1983,
    pp 1271-1302. This creates the SlownessModel and tau branches and
    saves them for later use.
    """

    overlayModelFilename = None
    plotVmod = False
    plotSmod = False
    plotTmod = False

    #  "constructor"
    def __init__(self):
        """ generated source for method __init__ """
        
        # Parse command line arguments. Very clever module, e.g. it
        # can print usage automatically.
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', '-d', '--debug', action='store_true',
                            help='increase output verbosity')
        parser.add_argument('-i', '--input_dir',
                            help='set directory of input velocity models (default: ./data/)')
        parser.add_argument('-o', '--output_dir',
                            help='set where to write the .taup model - be careful, this will overwrite any previous models of the same name (default: current dir)')
        parser.add_argument('-f', '--filename',
                            help='the velocity model name (default: iasp91.tvel)')
        args = parser.parse_args()
                
        self.DEBUG = args.verbose
        self.directory = args.input_dir
        self.outdir = args.output_dir
        self.modelFilename = args.filename

        if self.directory is None:
            # if no directory given, assume data is in ./data:
            self.directory = os.path.join(os.path.dirname(os.path.abspath(
                inspect.getfile(inspect.currentframe()))), "data")
        if self.modelFilename is None:
            self.modelFilename = "iasp91.tvel"

    @classmethod
    def main(cls):
        """ Do loadVMod, then start. """

        print("TauP_Create starting...")
        tauPCreate = TauP_Create()
    
        # try:
        #     print("Loading velocity model.")
        #     tauPCreate.loadVMod()
        #     print("Running tauPCreate.start.")
        #     tauPCreate.start()
        #     print("Done!")
        # except Exception as e:
        #     print("Something went fundamentally wrong:", e)
        # I need a stack trace:
        print("Loading velocity model.")
        tauPCreate.loadVMod()
        print("Running tauPCreate.start.")
        tauPCreate.start()

    def loadVMod(self):
        """ Tries to load a velocity model first via readVelocityFile,
        or if unsuccessful load internally from a previously stored
        model.
        """

        # Read the velocity model file.
        filename = os.path.join(self.directory, self.modelFilename)
        if self.DEBUG:
            print("filename =", filename)
        
        self.vMod = VelocityModel.readVelocityFile(filename)
        if self.vMod is None:
            # try and load internally
            # self.vMod = TauModelLoader.loadVelocityModel(self.modelFilename)
            # Frankly, I don't think it's a good idea to somehow load the model from a non-obvious path.
            # Better to just raise the exception and force user to be clear about what VelocityModel to read from where.
            # Maybe this could be done sensibly, as in if a model is specified but no path, some standard models
            # can be used? But given that they have to be in the package somewhere anyway, that's cosmetic.
            pass
        if self.vMod is None:
            raise IOError("Velocity model file not found: " + self.modelFilename + ", tried internally (not really) and from file: " + filename)
        # if model was read:
        if self.DEBUG:
            print("Done reading velocity model.")
            print("Radius of model " + self.vMod.modelName + " is " + str(self.vMod.radiusOfEarth))
        # if self.overlayModelFilename != None:
        # ... not sure what that is really meant to do ... #

        # if self.DEBUG:
        #    print("velocity mode: " + self.vMod)
        return self.vMod

    def createTauModel(self, vMod):
        """ Takes a v model and makes a SlownessModel out of it,
        then passes that to TauModel """
        if vMod is None:
            raise ValueError("vMod cannot be null")
        if vMod.isSpherical is False:
            raise Exception("Flat slowness model not yet implemented.")
        SlownessModel.DEBUG = self.DEBUG
        if self.DEBUG:
            print("Using parameters provided in TauP_config.ini (or defaults if not) to call SlownessModel...")
        import configparser
        config = configparser.ConfigParser()
        try:
            config.read('TauP_config.ini')
            ssm = config['SlownessModel_created_from_VelocityModel']
            # Read values from the appropriate section if defined, else use default values.
            minDeltaP = float(ssm.get('mindeltaP', 0.1))  # (are actually case-insensitive)
            maxDeltaP = float(ssm.get('maxDeltaP', 11))
            maxDepthInterval = float(ssm.get('maxDepthInterval', 115))
            maxRangeInterval = float(ssm.get('maxRangeInterval', 2.5))
            maxInterpError = float(ssm.get('maxInterpError', 0.05))
            allowInnerCoreS = ssm.getboolean('allowInnerCoreS', True)
        except KeyError:
            print("Couldn't read config file, using defaults. Config file must be a file named TauP_config (in working "
                  "dir) with a section named SlownessModel_created_from_VelocityModel.")
            minDeltaP = 0.1
            maxDeltaP = 11
            maxDepthInterval = 115
            maxRangeInterval = 2.5
            maxInterpError = 0.05
            allowInnerCoreS = True

        from math import pi
        self.sMod = SlownessModel(vMod,
                                  minDeltaP, maxDeltaP, maxDepthInterval, maxRangeInterval * pi / 180, maxInterpError,
                                  allowInnerCoreS, SlownessModel.DEFAULT_SLOWNESS_TOLERANCE)
        if self.DEBUG:
            print("Parameters are:")
            print("taup.create.minDeltaP = " + str(self.sMod.minDeltaP) + " sec / radian")
            print("taup.create.maxDeltaP = " + str(self.sMod.maxDeltaP) + " sec / radian")
            print("taup.create.maxDepthInterval = " + str(self.sMod.maxDepthInterval) + " kilometers")
            print("taup.create.maxRangeInterval = " + str(self.sMod.maxRangeInterval) + " degrees")
            print("taup.create.maxInterpError = " + str(self.sMod.maxInterpError) + " seconds")
            print("taup.create.allowInnerCoreS = " + str(self.sMod.allowInnerCoreS))
            print("Slow model " + " " + str(self.sMod.getNumLayers(True)) + " P layers," + str(self.sMod.getNumLayers(False)) + " S layers")
        #if self.DEBUG:
        #    print(self.sMod)
        # set the debug flags to value given here:
        TauModel.DEBUG = self.DEBUG
        SlownessModel.DEBUG = self.DEBUG
        # Creates tau model from slownesses.
        return TauModel(self.sMod)

    def start(self):
        """ Creates a tau model from a velocity model. Called by
        TauP_Create.main after loadVMod; calls createTauModel and
        writes the result to a .taup file. """
        try:
            if self.plotVmod or self.plotSmod or self.plotTmod:
                print("Plotting is not implemented for smod and tmod even in java. vmod would call self.vMod.printGMT(self.plotVmodFilename)")
            else:
                self.tMod = self.createTauModel(self.vMod)
                # this reassigns tMod! Used to be TauModel() class,
                # now it's an instance of it.
                if self.DEBUG:
                    print("Done calculating Tau branches.")
                #if self.DEBUG:
                #    print(self.tMod)

                # The java behaviour for finding where to store the
                # file was pretty silly. It stored the out file to the
                # working dir, but in two different ways. Here, just store to current dir:
                outModelFileName = self.vMod.modelName + ".taup"
                if self.outdir is None:
                    outFile = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), outModelFileName)
                else:
                    outFile = os.path.join(self.outdir, outModelFileName)
                self.tMod.writeModel(outFile)
                if self.DEBUG:
                    print("Done Saving " + outFile)
        except IOError as e:
            print("Tried to write!\n Caught IOError. Do you have write permission in this directory?", e)
        except KeyError as e:
            print('file not found or wrong key?', e)
        # except VelocityModelException as e:
        #     print("Caught VelocityModelException.", e)
        finally:
            if self.DEBUG:
                print("Method start is done, but not necessarily successful.")


if __name__ == '__main__':
    TauP_Create.main()

