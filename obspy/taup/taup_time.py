#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import argparse
import math

from .helper_classes import TauModelError
from .seismic_phase import SeismicPhase


class TauP_Time(object):
    """
    Calculate travel times for different branches using linear interpolation
    between known slowness samples.
    """
    def __init__(self, model, phase_list, depth, degrees):
        self.model = model
        # tModDepth will be depth-corrected if source depth is not 0.
        self.tModDepth = self.model
        self.modelName = self.model.sMod.vMod.modelName

        # Names of phases to be used, e.g. PKIKP
        self.phaseNames = parsePhaseList(phase_list)
        # List to hold the SeismicPhases for the phases named in phaseNames.
        self.phases = []
        # The following are 'initialised' for the purpose of checking later
        # whether their value has been given on the cmd line.
        self.depth = depth
        self.degrees = degrees
        self.arrivals = []

    def run(self):
        """
        Do all the calculations and print the output if told to. The resulting
        arrival times will be in self.arrivals.
        """
        self.depthCorrect(self.depth)
        self.calculate(self.degrees)

    def depthCorrect(self, depth):
        """
        Corrects the TauModel for the given source depth (if not already
        corrected).
        """
        if self.tModDepth is None or self.tModDepth.source_depth != depth:
            self.tModDepth = self.model.depthCorrect(depth)
            # This is not recursion!
            self.arrivals = []
            self.recalc_phases()
        self.source_depth = depth

    def recalc_phases(self):
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
                    if (seismicPhase.source_depth == self.depth
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
        """
        Calculate the arrival times.
        """
        self.depthCorrect(self.source_depth)
        # Called before, but depthCorrect might have changed the phases.
        self.recalc_phases()
        self.calc_time(degrees)

    def calc_time(self, degrees):
        """
        Calls the calc_time method of SeismicPhase to calculate arrival
        times for every phase, each sorted by time.
        """
        self.degrees = degrees
        self.arrivals = []
        for phase in self.phases:
            phaseArrivals = phase.calc_time(degrees)
            self.arrivals += phaseArrivals
        self.arrivals = sorted(self.arrivals,
                               key=lambda arrivals: arrivals.time)


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
    """
    Called by parsePhaseList to replace e.g. ttall with the relevant phases.
    """
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
