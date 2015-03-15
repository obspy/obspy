#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Internal TauModel class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import base64
import json
import os
from copy import deepcopy
from itertools import count
from math import pi

import numpy as np

from obspy.core.compatibility import frombuffer

from .helper_classes import SlownessModelError, TauModelError
from .slowness_model import SlownessModel
from .tau_branch import TauBranch
from .velocity_model import VelocityModel


TAU_MODEL_KEYS = ['cmbBranch', 'cmbDepth', 'debug', 'iocbBranch',
                  'iocbDepth', 'mohoBranch', 'mohoDepth', 'noDisconDepths',
                  'radiusOfEarth', 'ray_params', 'sMod', 'sourceBranch',
                  'source_depth', 'spherical', 'tauBranches']
TAU_BRANCH_KEYS = ['botDepth', 'DEBUG', 'dist', 'isPWave', 'maxRayParam',
                   'minRayParam', 'minTurnRayParam', 'tau', 'time', 'topDepth']
TAU_SLOWNESS_MODEL_KEYS = [
    'allowInnerCoreS', 'criticalDepths', 'DEBUG', 'DEFAULT_SLOWNESS_TOLERANCE',
    'maxDeltaP', 'maxDepthInterval', 'maxInterpError', 'maxRangeInterval',
    'minDeltaP', 'PLayers', 'PWAVE', 'radiusOfEarth', 'SLayers',
    'slowness_tolerance', 'SWAVE', 'vMod']
TAU_VELOCITY_MODEL_KEYS = [
    'cmbDepth', 'default_cmb', 'default_iocb', 'default_moho', 'iocbDepth',
    'isSpherical', 'layers', 'maxRadius', 'minRadius', 'modelName',
    'mohoDepth', 'radiusOfEarth']


class TauModel(object):
    """
    Provides storage of all the TauBranches comprising a model.
    """
    # The following really need to be class attributes. For some reason.

    # Depth for which tau model as constructed.
    source_depth = 0.0
    radiusOfEarth = 6371.0
    # Branch with the source at its top.
    sourceBranch = 0
    # Depths that should not have reflections or phase conversions. For
    # instance, if the source is not at a branch boundary then
    # noDisconDepths contains source depth and reflections and phase
    # conversions are not allowed at this branch boundary. If the source
    # happens to fall on a real discontinuity then then it is not
    # included.
    noDisconDepths = []

    def __init__(self, sMod, spherical=True, debug=False, skip_calc=False):
        self.debug = debug
        self.radiusOfEarth = 6371.0
        # True if this is a spherical slowness model. False if flat.
        self.spherical = spherical
        # Ray parameters used to construct the tau branches. This may only be
        # a subset of the slownesses/ray parameters saved in the slowness
        # model due to high slowness zones (low velocity zones).
        self.ray_params = None
        # 2D NumPy array containing a TauBranch object
        # corresponding to each "branch" of the tau model, First list is P,
        # second is S. Branches correspond to depth regions between
        # discontinuities or reversals in slowness gradient for a wave type.
        # Each branch contains time, distance, and tau increments for each ray
        # parameter in ray_param for the layer. Rays that turn above the branch
        # layer get 0 for time, distance, and tau increments.
        self.tauBranches = None

        self.sMod = sMod

        if not skip_calc:
            self.calcTauIncFrom()

    def calcTauIncFrom(self):
        """
        Calculates tau for each branch within a slowness model.
        """
        # First, we must have at least 1 slowness layer to calculate a
        #  distance. Otherwise we must signal an exception.
        if self.sMod.getNumLayers(True) == 0 \
                or self.sMod.getNumLayers(False) == 0:
            raise SlownessModelError(
                "Can't calculate tauInc when getNumLayers() = 0. I need more "
                "slowness samples.")
        self.sMod.validate()
        # Create an array holding the ray parameter that we will use for
        # constructing the tau splines. Only store ray parameters that are
        # not in a high slowness zone, i.e. they are smaller than the
        # minimum ray parameter encountered so far.
        numBranches = len(self.sMod.criticalDepths) - 1
        self.tauBranches = np.empty((2, numBranches), dtype=TauBranch)
        # Here we find the list of ray parameters to be used for the tau
        # model. We only need to find ray parameters for S waves since P
        # waves have been constructed to be a subset of the S samples.
        rayNum = 0
        minPSoFar = self.sMod.SLayers[0]['topP']
        tempRayParams = np.empty(
            2 * self.sMod.getNumLayers(False) + len(self.sMod.criticalDepths))
        # Make sure we get the top slowness of the very top layer
        tempRayParams[rayNum] = minPSoFar
        rayNum += 1
        for currSLayer in self.sMod.SLayers:
            # Add the top if it is strictly less than the last sample added.
            # Note that this will not be added if the slowness is continuous
            #  across the layer boundary.
            if currSLayer['topP'] < minPSoFar:
                tempRayParams[rayNum] = currSLayer['topP']
                rayNum += 1
                minPSoFar = currSLayer['topP']
            if currSLayer['botP'] < minPSoFar:
                # Add the bottom if it is strictly less than the last sample
                # added. This will always happen unless we are
                # within a high slowness zone.
                tempRayParams[rayNum] = currSLayer['botP']
                rayNum += 1
                minPSoFar = currSLayer['botP']
        # Copy tempRayParams to ray_param while chopping off trailing zeros
        # (from the initialisation), so the size is exactly right. NB
        # slicing doesn't really mean deep copy, but it works for a list of
        # doubles like this
        self.ray_params = tempRayParams[:rayNum]
        if self.debug:
            print("Number of slowness samples for tau:" + str(rayNum))
        for waveNum, isPWave in enumerate([True, False]):
            # The minimum slowness seen so far.
            minPSoFar = self.sMod.getSlownessLayer(0, isPWave)['topP']
            # for critNum, (topCritDepth, botCritDepth) in enumerate(zip(
            # self.sMod.criticalDepths[:-1], self.sMod.criticalDepths[1:])):
            # Faster:
            for critNum, topCritDepth, botCritDepth in zip(
                    count(), self.sMod.criticalDepths[:-1],
                    self.sMod.criticalDepths[1:]):
                topCritLayerNum = topCritDepth['pLayerNum'] \
                    if isPWave else topCritDepth['sLayerNum']
                botCritLayerNum = (botCritDepth['pLayerNum'] if isPWave
                                   else botCritDepth['sLayerNum']) - 1
                self.tauBranches[waveNum, critNum] = \
                    TauBranch(topCritDepth['depth'], botCritDepth['depth'],
                              isPWave)
                self.tauBranches[waveNum, critNum].DEBUG = self.debug
                self.tauBranches[waveNum, critNum].createBranch(
                    self.sMod, minPSoFar, self.ray_params)
                # Update minPSoFar. Note that the new minPSoFar could be at
                # the start of a discontinuity over a high slowness zone,
                # so we need to check the top, bottom and the layer just
                # above the discontinuity.
                topSLayer = self.sMod.getSlownessLayer(topCritLayerNum,
                                                       isPWave)
                botSLayer = self.sMod.getSlownessLayer(botCritLayerNum,
                                                       isPWave)
                minPSoFar = min(minPSoFar,
                                min(topSLayer['topP'], botSLayer['botP']))
                botSLayer = self.sMod.getSlownessLayer(
                    self.sMod.layerNumberAbove(botCritDepth['depth'], isPWave),
                    isPWave)
                minPSoFar = min(minPSoFar, botSLayer['botP'])
        # Here we decide which branches are the closest to the Moho, CMB,
        # and IOCB by comparing the depth of the top of the branch with the
        # depths in the Velocity Model.
        bestMoho = 1e300
        bestCmb = 1e300
        bestIocb = 1e300
        for branchNum, tBranch in enumerate(self.tauBranches[0]):
            if abs(tBranch.topDepth - self.sMod.vMod.mohoDepth) <= bestMoho:
                # Branch with Moho at its top.
                self.mohoBranch = branchNum
                bestMoho = abs(tBranch.topDepth - self.sMod.vMod.mohoDepth)
            if abs(tBranch.topDepth - self.sMod.vMod.cmbDepth) < bestCmb:
                self.cmbBranch = branchNum
                bestCmb = abs(tBranch.topDepth - self.sMod.vMod.cmbDepth)
            if abs(tBranch.topDepth - self.sMod.vMod.iocbDepth) < bestIocb:
                self.iocbBranch = branchNum
                bestIocb = abs(tBranch.topDepth - self.sMod.vMod.iocbDepth)
        # Now set mohoDepth etc. to the top of the branches we have decided on.
        self.mohoDepth = self.tauBranches[0, self.mohoBranch].topDepth
        self.cmbDepth = self.tauBranches[0, self.cmbBranch].topDepth
        self.iocbDepth = self.tauBranches[0, self.iocbBranch].topDepth
        self.validate()

    def __str__(self):
        desc = "Delta tau for each slowness sample and layer.\n"
        for j, ray_param in enumerate(self.ray_params):
            for i, tb in enumerate(self.tauBranches[0]):
                desc += (
                    " i " + str(i) + " j " + str(j) + " ray_param " +
                    str(ray_param) +
                    " tau " + str(tb.tau[j]) + " time " +
                    str(tb.time[j]) + " dist " +
                    str(tb.dist[j]) + " degrees " +
                    str(tb.dist[j] * 180 / pi) + "\n")
            desc += "\n"
        return desc

    def validate(self):
        # Could implement the model validation; not critical right now
        return True

    def depth_correct(self, depth):
        """
        Called in TauP_Time. Computes a new tau model for a source at depth
        using the previously computed branches for a surface source. No
        change is needed to the branches above and below the branch
        containing the depth, except for the addition of a slowness sample.
        The branch containing the source depth is split into 2 branches,
        and up going branch and a downgoing branch. Additionally,
        the slowness at the source depth must be sampled exactly as it is an
        extremal point for each of these branches. Cf. Buland and Chapman p
        1290.
        """
        if self.source_depth != 0:
            raise TauModelError("Can't depth correct a TauModel that is not "
                                "originally for a surface source.")
        if depth > self.radiusOfEarth:
            raise TauModelError("Can't depth correct to a source deeper than "
                                "the radius of the Earth.")
        depthCorrected = self.loadFromDepthCache(depth)
        if depthCorrected is None:
            depthCorrected = self.splitBranch(depth)
            depthCorrected.source_depth = depth
            depthCorrected.sourceBranch = depthCorrected.findBranch(depth)
            depthCorrected.validate()
            # Put in cache somehow: self.depthCache.put(depthCorrected)
        return depthCorrected

    def loadFromDepthCache(self, depth):
        # Could speed up by implementing cache.
        # Must return None if loading fails.
        return None

    def splitBranch(self, depth):
        """
        Returns a new TauModel with the branches containing depth split at
        depth. Used for putting a source at depth since a source can only be
        located on a branch boundary.
         """
        # First check to see if depth happens to already be a branch
        # boundary, then just return original model.
        for tb in self.tauBranches[0]:
            if tb.topDepth == depth or tb.botDepth == depth:
                return deepcopy(self)
        # Depth is not a branch boundary, so must modify the tau model.
        indexP = -1
        PWaveRayParam = -1
        indexS = -1
        SWaveRayParam = -1
        outSMod = self.sMod
        outRayParams = self.ray_params
        oldRayParams = self.ray_params
        # Do S wave first since the S ray param is > P ray param.
        for isPWave in [False, True]:
            splitInfo = outSMod.splitLayer(depth, isPWave)
            outSMod = splitInfo.sMod
            if splitInfo.neededSplit and not splitInfo.movedSample:
                # Split the slowness layers containing depth into two layers
                # each.
                newRayParam = splitInfo.ray_param
                # Insert the new ray parameters into the ray_param array.
                above = oldRayParams[:-1]
                below = oldRayParams[1:]
                index = (above < newRayParam) & (newRayParam < below)
                if np.any(index):
                    index = np.where(index)[0][0]
                    # FIXME: The original code uses oldRayParams, but that
                    # seems like it would not work if you need to insert both
                    # P and S waves. This part of the code doesn't seem to be
                    # triggered, though.
                    outRayParams = np.insert(oldRayParams, index, newRayParam)

                    if isPWave:
                        indexP = index
                        PWaveRayParam = newRayParam
                    else:
                        indexS = index
                        SWaveRayParam = newRayParam

        # Now add a sample to each branch above the depth, split the branch
        # containing the depth, and add a sample to each deeper branch.
        branchToSplit = self.findBranch(depth)
        newTauBranches = np.empty((2, self.tauBranches.shape[1] + 1),
                                  dtype=TauBranch)
        for i in range(branchToSplit):
            newTauBranches[0, i] = self.tauBranches[0, i]
            newTauBranches[1, i] = self.tauBranches[1, i]
            # Add the new ray parameter(s) from splitting the S and/or P
            # wave slowness layer to both the P and S wave tau branches (if
            # splitting occurred).
            if indexS != -1:
                newTauBranches[0, i].insert(SWaveRayParam, outSMod, indexS)
                newTauBranches[1, i].insert(SWaveRayParam, outSMod, indexS)
            if indexP != -1:
                newTauBranches[0, i].insert(PWaveRayParam, outSMod, indexP)
                newTauBranches[1, i].insert(PWaveRayParam, outSMod, indexP)
        for pOrS in range(2):
            newTauBranches[pOrS, branchToSplit] = TauBranch(
                self.tauBranches[pOrS, branchToSplit].topDepth, depth,
                pOrS == 0)
            newTauBranches[pOrS, branchToSplit].createBranch(
                outSMod, self.tauBranches[pOrS, branchToSplit].maxRayParam,
                outRayParams)
            newTauBranches[pOrS, branchToSplit + 1] = \
                self.tauBranches[pOrS, branchToSplit].difference(
                    newTauBranches[pOrS, branchToSplit],
                    indexP, indexS, outSMod,
                    newTauBranches[pOrS, branchToSplit].minRayParam,
                    outRayParams)
        for i in range(branchToSplit + 1, len(self.tauBranches[0])):
            for pOrS in range(2):
                newTauBranches[pOrS, i + 1] = self.tauBranches[pOrS, i]
            if indexS != -1:
                # Add the new ray parameter from splitting the S wave
                # slownes layer to both the P and S wave tau branches.
                for pOrS in range(2):
                    newTauBranches[pOrS, i + 1].insert(SWaveRayParam, outSMod,
                                                       indexS)
            if indexP != -1:
                # Add the new ray parameter from splitting the P wave
                # slownes layer to both the P and S wave tau branches.
                for pOrS in range(2):
                    newTauBranches[pOrS, i + 1].insert(PWaveRayParam, outSMod,
                                                       indexS)
        # We have split a branch so possibly sourceBranch, mohoBranch,
        # cmbBranch and iocbBranch are off by 1.
        outSourceBranch = self.sourceBranch
        if self.source_depth > depth:
            outSourceBranch += 1
        outmohoBranch = self.mohoBranch
        if self.mohoDepth > depth:
            outmohoBranch += 1
        outcmbBranch = self.cmbBranch
        if self.cmbDepth > depth:
            outcmbBranch += 1
        outiocbBranch = self.iocbBranch
        if self.iocbDepth > depth:
            outiocbBranch += 1
        # No overloaded constructors - so do it this way to bypass the
        # calcTauIncFrom in the __init__.
        tMod = TauModel(outSMod, spherical=self.spherical, debug=self.debug,
                        skip_calc=True)
        tMod.source_depth = self.source_depth
        tMod.sourceBranch = outSourceBranch
        tMod.mohoBranch = outmohoBranch
        tMod.mohoDepth = self.mohoDepth
        tMod.cmbBranch = outcmbBranch
        tMod.cmbDepth = self.cmbDepth
        tMod.iocbBranch = outiocbBranch
        tMod.iocbDepth = self.iocbDepth
        tMod.ray_params = outRayParams
        tMod.tauBranches = newTauBranches
        tMod.noDisconDepths = self.noDisconDepths + [depth]
        tMod.validate()
        return tMod

    def findBranch(self, depth):
        """Finds the branch that either has the depth as its top boundary, or
        strictly contains the depth. Also, we allow the bottom-most branch to
        contain its bottom depth, so that the center of the earth is contained
        within the bottom branch."""
        for i, tb in enumerate(self.tauBranches[0]):
            if tb.topDepth <= depth < tb.botDepth:
                return i
        # Check to see if depth is centre of the Earth.
        if self.tauBranches[0, -1].botDepth == depth:
            return len(self.tauBranches) - 1
        else:
            raise TauModelError("No TauBranch contains this depth.")

    def getTauBranch(self, branchNum, isPWave):
        if isPWave:
            return self.tauBranches[0, branchNum]
        else:
            return self.tauBranches[1, branchNum]

    def getBranchDepths(self):
        """
        Return an array of the depths that are boundaries between branches.
        :return:
        """
        branchDepths = [self.getTauBranch(0, True).topDepth]
        branchDepths += [self.getTauBranch(
            i - 1, True).botDepth for i in range(1, len(self.tauBranches[0]))]
        return branchDepths

    def serialize(self, filename):
        data = _dumps(self)
        with open(filename, "w") as fh:
            fh.write(data.decode())

    @staticmethod
    def deserialize(filename):
        with open(filename, "r") as fh:
            data = fh.read()
        return _loads(data)

    @staticmethod
    def from_file(model_name):
        if os.path.exists(model_name):
            filename = model_name
        else:
            filename = os.path.join(os.path.dirname(__file__), "data",
                                    "models", model_name.lower() + ".json")
        return TauModel.deserialize(filename)


class TauEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray, TauBranch or TauModel it will be
        converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, np.ndarray):
            obj = np.require(obj, requirements=['C_CONTIGUOUS'])
            # handle array of TauBranch objects
            if obj.flatten()[0].__class__ == TauBranch:
                __ndarray_list__ = [_dumps(x) for x in obj.flatten()]
                data_b64 = [base64.b64encode(x) for x in __ndarray_list__]
                return dict(__ndarray_list__=data_b64,
                            dtype="TauBranch",
                            shape=obj.shape)
            # handle other arrays (e.g. int, float)
            else:
                data_b64 = base64.b64encode(obj.data)
                return dict(__ndarray__=data_b64,
                            dtype=str(obj.dtype),
                            shape=obj.shape)
        elif isinstance(obj, TauBranch):
            data = dict([(key, getattr(obj, key)) for key in TAU_BRANCH_KEYS])
            data['dtype'] = "TauBranch"
            return data
        elif isinstance(obj, TauModel):
            data = dict([(key, getattr(obj, key)) for key in TAU_MODEL_KEYS])
            data['dtype'] = "TauModel"
            return data
        elif isinstance(obj, SlownessModel):
            data = dict([(key, getattr(obj, key))
                         for key in TAU_SLOWNESS_MODEL_KEYS])
            data['dtype'] = "SlownessModel"
            return data
        elif isinstance(obj, VelocityModel):
            data = dict([(key, getattr(obj, key))
                         for key in TAU_VELOCITY_MODEL_KEYS])
            data['dtype'] = "VelocityModel"
            return data
        return json.JSONEncoder(self, obj)


def _json_obj_hook(dct):
    """
    Decodes previously encoded numpy ndarrays and TauBranch and TauModel.

    :type dct: dict
    :param dct: json encoded ndarray, TauBranch or TauModel
    :return: deserialized object
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        try:
            return frombuffer(data, dct['dtype']).reshape(dct['shape'])
        # if we run into an exception we have a structured array, likely
        except ValueError:
            dtype = np.array(dct['dtype'].split("'")[1::2])
            dtype = dtype.reshape((-1, 2)).tolist()
            dtype = [tuple([native_str(x), native_str(y)]) for x, y in dtype]
            return frombuffer(data, dtype=dtype).reshape(dct['shape'])
    elif isinstance(dct, dict) and '__ndarray_list__' in dct:
        data = [_loads(base64.b64decode(x)) for x in dct['__ndarray_list__']]
        return np.array(data, dtype=TauBranch).reshape(dct['shape'])
    elif isinstance(dct, dict) and dct.get("dtype", "") == "TauBranch":
        return _deserialize_TauBranch(dct)
    elif isinstance(dct, dict) and dct.get("dtype", "") == "TauModel":
        return _deserialize_TauModel(dct)
    elif isinstance(dct, dict) and dct.get("dtype", "") == "SlownessModel":
        return _deserialize_SlownessModel(dct)
    elif isinstance(dct, dict) and dct.get("dtype", "") == "VelocityModel":
        return _deserialize_VelocityModel(dct)
    return dct


def _deserialize_TauBranch(dct):
    tb = TauBranch()
    dct.pop("dtype")
    for k, v in dct.iteritems():
        setattr(tb, k, v)
    return tb


def _deserialize_TauModel(dct):
    tm = TauModel(sMod=None, skip_calc=True)
    dct.pop("dtype")
    for k, v in dct.iteritems():
        setattr(tm, k, v)
    return tm


def _deserialize_SlownessModel(dct):
    sm = SlownessModel(vMod=None, skip_model_creation=True)
    dct.pop("dtype")
    for k, v in dct.iteritems():
        setattr(sm, k, v)
    return sm


def _deserialize_VelocityModel(dct):
    vm = VelocityModel()
    dct.pop("dtype")
    for k, v in dct.iteritems():
        setattr(vm, k, v)
    return vm


def _dumps(*args, **kwargs):
    kwargs.setdefault('cls', TauEncoder)
    return json.dumps(*args, **kwargs)


def _loads(*args, **kwargs):
    kwargs.setdefault('object_hook', _json_obj_hook)
    return json.loads(*args, **kwargs)
