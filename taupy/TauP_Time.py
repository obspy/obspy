#!/usr/bin/env python3

import os
import inspect
import argparse
import taupy.TauModelLoader as TauModelLoader
from taupy.helper_classes import TauModelError
from taupy.SeismicPhase import SeismicPhase
from math import pi


class TauP_Time(object):
    """Calculate travel times for different branches using linear interpolation
    between known slowness samples."""
    DEBUG = False
    verbose = False
    # Allow phases originating in the core
    expert = False
    # Names of phases to be used, e.g. PKIKP
    phaseNames = []
    modelName = "iasp91"
    # This is needed to check later if assignment has happened on cmd line, or if reading conf is needed.
    tMod = None
    # TauModel derived from tMod by correcting it for a non-surface source.
    tModDepth = None
    # List to hold the SeismicPhases for the phases named in phaseNames.
    phases = []
    # The following are 'initialised' for the purpose of checking later whether their value has been given in
    depth = 0
    degrees = None
    azimuth = None
    backAzimuth = None
    stationLat = None
    stationLon = None
    eventLat = None
    eventLon = None
    arrivals = []
    relativePhaseName = None
    relativeArrival = None  # That's not even necessary, but otherwise attribute is added outside of constructor - is that so bad? Who knows.

    def __init__(self):
        # Could have the command line args read here...
        # No, better do it in  if name == main  because if it's not the main script that
        # shouldn't happen!
        pass

    def init(self):
        """Performs initialisation of the tool. Config file is queried for the default
        model to load, which source depth and phases to use etc."""
        self.readConfig()
        self.readcmdLineArgs()
        self.readTauModel()

    def readTauModel(self):
        """Do the reading simply for now."""
        # Todo: Give this the full capabilities.
        # This should be the same model path that was used by TauP_Create for writing!
        # Maybe that should write to the config file?
        modelPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # current dir
        tModLoaded = TauModelLoader.load(self.modelName, modelPath, self.verbose)
        if tModLoaded is not None:
            self.tMod = tModLoaded
            # tModDepth will be depth-corrected if source depth is not 0.
            self.tModDepth = self.tMod
            self.modelName = self.tMod.sMod.vMod.modelName
        else:
            raise TauModelError("Unable to load " + str(self.modelName))

    def start(self):
        """Called after init. Merge the two?"""
        if self.degrees is not None or all(x is not None for x in (self.stationLat, self.stationLon,
                                                                   self.eventLat, self.eventLon)):
            # Enough information has been given on the command line, just do simple calculation.
            if self.degrees is None:
                # Calculate the degrees from station and event locations.
                # self.degrees = SphericalCoords.distance
                # Check for maybe numpy libraries that handle this kind of calculations!
                pass
            self.depthCorrect(self.depth)
            self.calculate(self.degrees)
            self.printResult()
        else:
            # Get the info from interactive mode. Not necessary to implement just now.
            raise TypeError("Not enough info given on cmd line.")

    def depthCorrect(self, depth):
        """Corrects the TauModel for the given source depth (if not already corrected)."""
        if self.tModDepth is None or self.tModDepth.sourceDepth != depth:
            self.tModDepth = self.tMod.depthCorrect(depth)  # This is not recursion!
            self.arrivals = []
            self.recalcPhases()
        self.sourceDepth = depth

    def recalcPhases(self):
        """Recalculates the given phases using a possibly new or changed tau model.
        This should not need to be called by outside classes as it is called by depthCorrect and calculate.
        """
        newPhases = []
        for tempPhaseName in self.phaseNames:
            alreadyAdded = False
            for phaseNum, seismicPhase in enumerate(self.phases):
                if seismicPhase.name == tempPhaseName:
                    self.phases.pop(phaseNum)
                    if seismicPhase.sourceDepth == self.depth and seismicPhase.tMod == self.tModDepth:
                        # OK so copy to newPhases:
                        newPhases.append(seismicPhase)
                        alreadyAdded = True
                        break
            if not alreadyAdded:
                # Didn't find it precomputed, so recalculate:
                try:
                    seismicPhase = SeismicPhase(tempPhaseName, self.tModDepth)
                    newPhases.append(seismicPhase)
                except TauModelError:
                    print("Error with this phase, skipping it: " + str(tempPhaseName))
                # Uncomment if stacktrace is needed.
                #seismicPhase = SeismicPhase(tempPhaseName, self.tModDepth)
                #newPhases.append(seismicPhase)
            self.phases = newPhases

    def calculate(self, degrees):
        """Calls the actual calculations of the arrival times."""
        self.depthCorrect(self.sourceDepth)
        self.recalcPhases()  # Called before, but maybe depthCorrect to sourceDepth has changed the phases??
        self.calcTime(degrees)
        if self.relativePhaseName is not None:
            relPhases = []
            splitNames = getPhaseNames(self.relativePhaseName)  # Static!
            for sName in splitNames:
                relPhases.append(SeismicPhase(sName, self.tModDepth))
            self.relativeArrival = SeismicPhase.getEarliestArrival(relPhases, degrees)

    def calcTime(self, degrees):
        """
        :param degrees:
        :return self.arrivals:
        """
        self.degrees = degrees
        self.arrivals = []
        for phase in self.phases:
            phaseArrivals = phase.calcTime(degrees)
            self.arrivals += phaseArrivals
        self.sortArrivals()

    def sortArrivals(self):
        self.arrivals = sorted(self.arrivals, key=lambda arrivals: arrivals.time)
        pass

    def printResult(self):
        # Do  only a simple way for now.
        print("\nModel:", self.modelName)
        lineOne = "Distance   Depth   Phase   Travel    Ray Param   Takeoff  Incident  Purist     Purist"
        lineTwo = "   (deg)    (km)   Name    Time (s)  p  (s/deg)    (deg)     (deg)  Distance   Name "
        print(lineOne)
        print(lineTwo)
        print("-"*len(lineOne))
        for arrival in self.arrivals:
            out = "{:>8.2f}".format(arrival.getModuloDistDeg()) + "   "
            out += "{:>5.1f}".format(self.depth) + "   "
            out += "{:<5s}".format(arrival.name) + "   "
            out += "{:>8.2f}".format(arrival.time) + "   "
            out += "{:>9.3f}".format(arrival.rayParam * pi/180) + "   "
            out += "{:>6.2f}".format(arrival.takeoffAngle) + "   "
            out += "{:>7.2f}".format(arrival.incidentAngle) + "   "
            out += "{:>7.2f}".format(arrival.dist*180/pi) + ("  = " if arrival.puristName == arrival.name else "  * ")
            out += "{:<5s}".format(arrival.puristName) + "   "
            print(out)

    def readConfig(self):
        pass

    def readcmdLineArgs(self):
        """
        Reads the command line arguments, if present.
        :return:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', '-d', '--debug', action='store_true',
                            help='increase output verbosity')
        parser.add_argument('-ph', '--phase_list',
                            help='comma separated phase list, no white space!')
        parser.add_argument('-pf', '--phase_file',
                            help='file containing phases')
        parser.add_argument('-mod', '--modelname',
                            help='Use this velocity model for calculations. Default is iasp91.')
        parser.add_argument('--depth',
                            help='source depth in km')
        parser.add_argument('-deg', '--degrees',
                            help='distance in degrees')
        parser.add_argument('-km', '--kilometres',
                            help='distance in kilometres')
        # Can add station/event lat long instead
        parser.add_argument('-o', '--outfile',
                            help='output is redirected to "outfile"')
        args = parser.parse_args()
        self.DEBUG = args.verbose
        self.phaseNames = args.phase_list.split(',')
        self.phaseFile = args.phase_file
        self.mod = args.modelname
        self.depth = float(args.depth)
        self.degrees = float(args.degrees)
        self.kilometres = float(args.kilometres) if args.kilometres else None
        self.outFile = args.outfile


if __name__ == '__main__':
    # Replace the Java main method, which is a static (i.e. class) method called whenever the
    # program, that is TauP_Time, is executed.
    tauPTime = TauP_Time()
    # read cmd line args:
    #tauPTime.phaseNames = ["S", "P"]  #must be with no whitespace around!
    tauPTime.modelName = "iasp91"
    tauPTime.degrees = 89
    tauPTime.depth = 200
    tauPTime.init()
    tauPTime.start()


def getPhaseNames(phaseName):
    names = []
    if(phaseName.lower() == "ttp"
            or phaseName.lower() == "tts"
            or phaseName.lower() == "ttbasic"
            or phaseName.lower() == "tts+"
            or phaseName.lower() == "ttp+"
            or phaseName.lower() == "ttall"):
        if(phaseName.lower() == "ttp"
           or phaseName.lower() == "ttp+"
           or phaseName.lower() == "ttbasic"
           or phaseName.lower() == "ttall"):
            names.append("p")
            names.append("P")
            names.append("Pn")
            names.append("Pdiff")
            names.append("PKP")
            names.append("PKiKP")
            names.append("PKIKP")
        if(phaseName.lower() == "tts"
                or phaseName.lower() == "tts+"
                or phaseName.lower() == "ttbasic"
                or phaseName.lower() == "ttall"):
            names.append("s")
            names.append("S")
            names.append("Sn")
            names.append("Sdiff")
            names.append("SKS")
            names.append("SKIKS")
        if(phaseName.lower() == "ttp+"
                or phaseName.lower() == "ttbasic"
                or phaseName.lower() == "ttall"):
            names.append("PcP")
            names.append("pP")
            names.append("pPdiff")
            names.append("pPKP")
            names.append("pPKIKP")
            names.append("pPKiKP")
            names.append("sP")
            names.append("sPdiff")
            names.append("sPKP")
            names.append("sPKIKP")
            names.append("sPKiKP")
        if(phaseName.lower() == "tts+"
                or phaseName.lower() == "ttbasic"
                or phaseName.lower() == "ttall"):
            names.append("sS")
            names.append("sSdiff")
            names.append("sSKS")
            names.append("sSKIKS")
            names.append("ScS")
            names.append("pS")
            names.append("pSdiff")
            names.append("pSKS")
            names.append("pSKIKS")
        if(phaseName.lower() == "ttbasic"
                or phaseName.lower() == "ttall"):
            names.append("ScP")
            names.append("SKP")
            names.append("SKIKP")
            names.append("PKKP")
            names.append("PKIKKIKP")
            names.append("SKKP")
            names.append("SKIKKIKP")
            names.append("PP")
            names.append("PKPPKP")
            names.append("PKIKPPKIKP")
        if phaseName.lower() == "ttall":
            names.append("SKiKP")
            names.append("PP")
            names.append("ScS")
            names.append("PcS")
            names.append("PKS")
            names.append("PKIKS")
            names.append("PKKS")
            names.append("PKIKKIKS")
            names.append("SKKS")
            names.append("SKIKKIKS")
            names.append("SKSSKS")
            names.append("SKIKSSKIKS")
            names.append("SS")
            names.append("SP")
            names.append("PS")
    else:
        names.append(phaseName)
    return names