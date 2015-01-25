#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *
import os


def internalLoad(modelName, model_path, verbose):
    """
    Try to un-pickle a .taup model of given name in a given path.
    """
    if modelName.endswith(".taup"):
        filename = modelName
    else:
        filename = modelName + ".taup"
    # Java tries to:
    # - find the model in the distributed taup.jar
    # - find it in the classpath (?)
    # - find it in the path specified by the taup.model.path property.
    #    This is what's passed in as searchPath!
    # - find it in the current directory
    # - try to load a velocity model of the same name and run
    # TauP_Create.createTauModel
    #
    # I'll keep it simple for now; stick to the specified path.
    # I also don't think finding the model just somewhere in the paths is
    # good practice.
    #  Maybe put the following in a TauModel.readModel method?
    modelFilename = os.path.join(model_path, filename)
    import pickle
    with open(modelFilename, 'rb') as f:
        return pickle.load(f)


def load(modelName, model_path, verbose):
    """
    Read a .taup model that was previously saved, given a filename and path.
    Uses unpickle, but could also handle loading from cache (not yet
    implemented).
    """
    out = loadFromCache(modelName)
    if out is None:
        out = internalLoad(modelName, model_path, verbose)
        # cache(out)
    return out


def loadFromCache(modelName):
    """
    Caching could be useful for many reruns, but is not yet implemented.
    """
    return None
