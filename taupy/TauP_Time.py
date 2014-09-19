#!/usr/bin/env python3

import os
import inspect
import argparse


class TauP_Time(object):
    """Calculate travel times for different branches using linear interpolation
    between known slowness samples."""

    DEBUG = False
    verbose = False
    # Allow phases originating in the core
    expert = False
    modelName = "iasp91"
    # This is needed to check later if assignment has happened on cmd line, or if reading conf is needed.
    tMod = None

    # List to hold the SeismicPhases for the phases named in phaseNames.
    #phases = []
    # etc but I am not sure if it's necessary... some of it seems just like cludgy values to see if they are set later?

    def __init__(self):
        # Could have the command line args read here...
        # No, better do it in  if name == main  because if it's not the main script that
        # shouldn't happen!
        # Names of phases to be used, e.g. PKIKP
        self.phaseNames = []

    def init(self):
        """Performs initialisation of the tool. Config file is queried for the default
        model to load, which source depth and phases to use etc."""
        if len(self.phaseNames):
            # Read phaseNames from config file or, failing that, use defaults
            pass
        # Also depth, but surely not if it's set in the cmd line??
        if self.tMod is None or self.tMod.vMod.modelName != "name in the properties":
            self.modelName = "name in the properties"
        self.readTauModel()
        # TODO make sure you get 100% what Java is doing:
        # Are these properties, like, mutable within the program?
        # Do the cmd line args overwrite them? When? How? Where? What?



    def start(self):
        pass

    def destroy(self):
        pass


if __name__ == '__main__':
    tauPTime = TauP_Time()
    tauPTime.init()
    tauPTime.start()
    tauPTime.destroy()
