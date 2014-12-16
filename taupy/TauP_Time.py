#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import inspect
import argparse
import math

import taupy.TauModelLoader as TauModelLoader
from taupy.helper_classes import TauModelError
from taupy.SeismicPhase import SeismicPhase


class TauP_Time(object):
    """
    Calculate travel times for different branches using linear interpolation
    between known slowness samples.
    """
    DEBUG = False
    verbose = False

    def __init__(self, phaseList=None, modelName="iasp91", depth=0,
                 degrees=None, stationLat=None, stationLon=None,
                 eventLat=None, eventLon=None):
        phaseList = phaseList if phaseList is not None else []
        # Allow phases originating in the core
        self.expert = False
        # Names of phases to be used, e.g. PKIKP
        self.phaseList = phaseList
        self.phaseNames = []
        self.modelName = modelName
        # This is needed to check later if assignment has happened on cmd
        # line.
        self.tMod = None
        # TauModel derived from tMod by correcting it for a non-surface source.
        self.tModDepth = None
        # List to hold the SeismicPhases for the phases named in phaseNames.
        self.phases = []
        # The following are 'initialised' for the purpose of checking later
        # whether their value has been given on the cmd line.
        self.depth = depth
        self.degrees = degrees
        #self.azimuth = None
        #self.backAzimuth = None
        self.stationLat = stationLat
        self.stationLon = stationLon
        self.eventLat = eventLat
        self.eventLon = eventLon
        self.arrivals = []
        self.relativePhaseName = None
        # That's not even necessary, but otherwise attribute is added
        # outside of constructor - is that so bad? Who knows.
        self.relativeArrival = None

    def run(self, printOutput=False):
        """Does the calculations and prints the result."""
        self.phaseNames = parsePhaseList(self.phaseList)
        self.readTauModel()
        if self.degrees is not None or all(x is not None for x in (
                self.stationLat, self.stationLon, self.eventLat,
                self.eventLon)):
            # Enough information has been given on the command line, just do
            #  simple calculation.
            if self.degrees is None:
                stn = (self.stationLat, self.stationLon)
                event = (self.eventLat, self.eventLon)
                self.degrees = great_circle_dist(stn, event)
            self.depthCorrect(self.depth)
            self.calculate(self.degrees)
            if printOutput:
                self.printResult()
        else:
            # Get the info from interactive mode. Not necessary to implement
            #  just now.
            raise TypeError("Not enough info given on cmd line. "
                            "Use -h for help")

    def readTauModel(self):
        """
        Do the reading simply for now.
        """
        # This should be the same model path that was used by TauP_Create
        # for writing!
        # Current directory:
        modelPath = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        tModLoaded = TauModelLoader.load(self.modelName, modelPath,
                                         self.verbose)
        if tModLoaded is not None:
            self.tMod = tModLoaded
            # tModDepth will be depth-corrected if source depth is not 0.
            self.tModDepth = self.tMod
            self.modelName = self.tMod.sMod.vMod.modelName
        else:
            raise TauModelError("Unable to load " + str(self.modelName))

    def depthCorrect(self, depth):
        """
        Corrects the TauModel for the given source depth (if not already
        corrected).
        """
        if self.tModDepth is None or self.tModDepth.sourceDepth != depth:
            # This is not recursion!
            self.tModDepth = self.tMod.depthCorrect(depth)
            self.arrivals = []
            self.recalcPhases()
        self.sourceDepth = depth

    def recalcPhases(self):
        """
        Recalculates the given phases using a possibly new or changed tau
        model.
        """
        newPhases = []
        for tempPhaseName in self.phaseNames:
            alreadyAdded = False
            for phaseNum, seismicPhase in enumerate(self.phases):
                if seismicPhase.name == tempPhaseName:
                    self.phases.pop(phaseNum)
                    if (seismicPhase.sourceDepth == self.depth
                            and seismicPhase.tMod == self.tModDepth):
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
                    print("Error with this phase, skipping it: " +
                          str(tempPhaseName))
            self.phases = newPhases

    def calculate(self, degrees):
        """Calls the actual calculations of the arrival times."""
        self.depthCorrect(self.sourceDepth)
        # Called before, but depthCorrect might have changed the phases.
        self.recalcPhases()
        self.calcTime(degrees)
        if self.relativePhaseName is not None:
            relPhases = []
            splitNames = getPhaseNames(self.relativePhaseName)  # Static!
            for sName in splitNames:
                relPhases.append(SeismicPhase(sName, self.tModDepth))
            self.relativeArrival = SeismicPhase.getEarliestArrival(relPhases,
                                                                   degrees)

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
        self.arrivals = sorted(self.arrivals,
                               key=lambda arrivals: arrivals.time)
        pass

    def printResult(self):
        # Do  only a simple way for now.
        print("\nModel:", self.modelName)
        namespacewidth = len(max([arrival.name for arrival in self.arrivals],
                            key=len)) - 2

        lineOne = "Distance   Depth   Phase" + " "*namespacewidth +  \
                  "Travel    Ray Param   Takeoff  Incident  Purist     Purist"
        lineTwo = "   (deg)    (km)   Name " + " "*namespacewidth + \
                  "Time (s)  p (s/deg)     (deg)     (deg)  Distance   Name "
        print(lineOne)
        print(lineTwo)
        print("-"*(len(lineOne)-2))  # for output comparison to Java
        for arrival in self.arrivals:
            out = "{:>8.2f}".format(arrival.getModuloDistDeg()) + "   "
            out += "{:>5.1f}".format(self.depth) + "   "
            out += "{0:<{1}s}".format(arrival.name, namespacewidth + 2) + "   "
            out += "{:>8.2f}".format(arrival.time) + "   "
            out += "{:>8.3f}".format(arrival.rayParam * math.pi/180) + "   "
            if arrival.takeoffAngle == -0.0:
                arrival.takeoffAngle = 0  # for output comparability
            out += "{:>6.2f}".format(arrival.takeoffAngle) + "   "
            out += "{:>7.2f}".format(arrival.incidentAngle) + "   "
            out += "{:>7.2f}".format(arrival.dist*180/math.pi) + \
                   ("  = " if arrival.puristName == arrival.name else "  * ")
            out += "{:<5s}".format(arrival.puristName) + "   "
            print(out)

    def readcmdLineArgs(self):
        """
        Reads the command line arguments, if present.
        :return:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', '--debug',
                            action='store_true',
                            help='increase output verbosity')
        parser.add_argument('-ph', '--phase_list',
                            help='comma separated phase list, no white space!')
        parser.add_argument('-mod', '--modelname',
                            help='Use this velocity model for calculations. '
                                 'Default is iasp91.')
        parser.add_argument('-d', '--depth',
                            help='source depth in km')
        parser.add_argument('-deg', '--degrees',
                            help='distance in degrees')
        parser.add_argument('-km', '--kilometres',
                            help='distance in kilometres')
        parser.add_argument('-staLat', help='station latitude')
        parser.add_argument('-staLon', help='station longitude')
        parser.add_argument('-evtLat', help='event latitude')
        parser.add_argument('-evtLon', help='event longitude')
        # Can add station/event lat long instead
        parser.add_argument('-o', '--outfile',
                            help='output is redirected to "outfile"')
        args = parser.parse_args()
        # Avoid overwriting already set variables with None:
        self.DEBUG = args.verbose if args.verbose else self.DEBUG
        self.phaseList = args.phase_list.split(',') \
            if args.phase_list else self.phaseList
        self.modelName = args.modelname if args.modelname else self.modelName
        self.depth = float(args.depth) if args.depth else self.depth
        self.degrees = float(args.degrees) if args.degrees else self.degrees
        self.kilometres = float(args.kilometres) if args.kilometres else None
        self.stationLat = float(args.staLat) if args.staLat else None
        self.stationLon = float(args.staLon) if args.staLon else None
        self.eventLat = float(args.evtLat) if args.evtLat else None
        self.eventLon = float(args.evtLon) if args.evtLon else None
        self.outFile = args.outfile


def parsePhaseList(phaseList):
    """
    Takes a list of phases, returns a list of individual phases. Performs e.g.
    replacing e.g. ttall with the relevant phases.
    :param phaseList:
    :return phaseNames:
    """
    phaseNames = []
    for phaseName in phaseList:
        phaseNames += getPhaseNames(phaseName)
    # Remove duplicates:
    phaseNames = list(set(phaseNames))
    return phaseNames


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


def great_circle_dist(stn, event):
    rtod = 180 / math.pi
    dtor = math.pi / 180
    latA = stn[0]
    lonA = stn[1]
    latB = event[0]
    lonB = event[1]
    return rtod * math.acos(math.sin(latA * dtor) * math.sin(latB * dtor)
                            + math.cos(latA * dtor) * math.cos(latB * dtor)
                            * math.cos((lonB - lonA) * dtor))

if __name__ == '__main__':
    # Replace the Java main method, which is a static (i.e. class) method
    # called whenever the program, that is TauP_Time, is executed.
    tauPTime = TauP_Time()
    tauPTime.readcmdLineArgs()
    tauPTime.run(printOutput=True)

