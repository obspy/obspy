#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Internal TauModel class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import os
from copy import deepcopy
from itertools import count
from math import pi

import numpy as np

from .helper_classes import DepthRange, SlownessModelError, TauModelError
from .slowness_model import SlownessModel
from .tau_branch import TauBranch
from .velocity_model import VelocityModel


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
        extremal point for each of these branches. Cf. [Buland1983]_, page
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
        """
        Serialize model to numpy npz binary file.

        Summary of contents that have to be handled during serialization::

            TauModel
            ========
            cmbBranch <type 'int'>
            cmbDepth <type 'float'>
            debug <type 'bool'>
            iocbBranch <type 'int'>
            iocbDepth <type 'float'>
            mohoBranch <type 'int'>
            mohoDepth <type 'float'>
            noDisconDepths <type 'list'> (of float!?)
            radiusOfEarth <type 'float'>
            ray_params <type 'numpy.ndarray'> (1D, float)
            sMod <class 'obspy.taup.slowness_model.SlownessModel'>
            sourceBranch <type 'int'>
            source_depth <type 'float'>
            spherical <type 'bool'>
            tauBranches <type 'numpy.ndarray'> (2D, type TauBranch)

            TauBranch
            =========
            DEBUG <type 'bool'>
            botDepth <type 'float'>
            dist <type 'numpy.ndarray'>
            isPWave <type 'bool'>
            maxRayParam <type 'float'>
            minRayParam <type 'float'>
            minTurnRayParam <type 'float'>
            tau <type 'numpy.ndarray'>
            time <type 'numpy.ndarray'>
            topDepth <type 'float'>

            SlownessModel
            =============
            DEBUG <type 'bool'>
            DEFAULT_SLOWNESS_TOLERANCE <type 'float'>
            PLayers <type 'numpy.ndarray'>
            PWAVE <type 'bool'>
            SLayers <type 'numpy.ndarray'>
            SWAVE <type 'bool'>
            allowInnerCoreS <type 'bool'>
            criticalDepths <type 'numpy.ndarray'>
            fluidLayerDepths <type 'list'> (of DepthRange)
            highSlownessLayerDepthsP <type 'list'> (of DepthRange)
            highSlownessLayerDepthsS <type 'list'> (of DepthRange)
            maxDeltaP <type 'float'>
            maxDepthInterval <type 'float'>
            maxInterpError <type 'float'>
            maxRangeInterval <type 'float'>
            minDeltaP <type 'float'>
            radiusOfEarth <type 'float'>
            slowness_tolerance <type 'float'>
            vMod <class 'obspy.taup.velocity_model.VelocityModel'>

            VelocityModel
            =============
            cmbDepth <type 'float'>
            default_cmb <type 'float'>
            default_iocb <type 'float'>
            default_moho <type 'int'>
            iocbDepth <type 'float'>
            isSpherical <type 'bool'>
            layers <type 'numpy.ndarray'>
            maxRadius <type 'float'>
            minRadius <type 'int'>
            modelName <type 'unicode'>
            mohoDepth <type 'float'>
            radiusOfEarth <type 'float'>
        """
        # a) handle simple contents
        keys = ['cmbBranch', 'cmbDepth', 'debug', 'iocbBranch', 'iocbDepth',
                'mohoBranch', 'mohoDepth', 'noDisconDepths', 'radiusOfEarth',
                'ray_params', 'sourceBranch', 'source_depth', 'spherical']
        arrays = dict([(key, getattr(self, key)) for key in keys])
        # b) handle .tauBranches
        i, j = self.tauBranches.shape
        for j_ in range(j):
            for i_ in range(i):
                # just store the shape of self.tauBranches in the key names for
                # later reconstruction of array in deserialization.
                key = 'tauBranches_%i/%i_%i/%i' % (j_, j, i_, i)
                arrays[key] = self.tauBranches[i_][j_]._to_array()
        # c) handle simple contents of .sMod
        dtypes = [(native_str('DEBUG'), np.bool_),
                  (native_str('DEFAULT_SLOWNESS_TOLERANCE'), np.float_),
                  (native_str('PWAVE'), np.bool_),
                  (native_str('SWAVE'), np.bool_),
                  (native_str('allowInnerCoreS'), np.bool_),
                  (native_str('maxDeltaP'), np.float_),
                  (native_str('maxDepthInterval'), np.float_),
                  (native_str('maxInterpError'), np.float_),
                  (native_str('maxRangeInterval'), np.float_),
                  (native_str('minDeltaP'), np.float_),
                  (native_str('radiusOfEarth'), np.float_),
                  (native_str('slowness_tolerance'), np.float_)]
        slowness_model = np.empty(shape=(), dtype=dtypes)
        for dtype in dtypes:
            key = dtype[0]
            slowness_model[key] = getattr(self.sMod, key)
        arrays['sMod'] = slowness_model
        # d) handle complex contents of .sMod
        arrays['sMod.PLayers'] = self.sMod.PLayers
        arrays['sMod.SLayers'] = self.sMod.SLayers
        arrays['sMod.criticalDepths'] = self.sMod.criticalDepths
        for key in ['fluidLayerDepths', 'highSlownessLayerDepthsP',
                    'highSlownessLayerDepthsS']:
            data = getattr(self.sMod, key)
            if len(data) == 0:
                arr_ = np.array([])
            else:
                arr_ = np.vstack([data_._to_array() for data_ in data])
            arrays['sMod.' + key] = arr_
        # e) handle .sMod.vMod
        dtypes = [(native_str('cmbDepth'), np.float_),
                  (native_str('default_cmb'), np.float_),
                  (native_str('default_iocb'), np.float_),
                  (native_str('default_moho'), np.int_),
                  (native_str('iocbDepth'), np.float_),
                  (native_str('isSpherical'), np.bool_),
                  (native_str('maxRadius'), np.float_),
                  (native_str('minRadius'), np.int_),
                  (native_str('modelName'), np.str_,
                   len(self.sMod.vMod.modelName)),
                  (native_str('mohoDepth'), np.float_),
                  (native_str('radiusOfEarth'), np.float_)]
        velocity_model = np.empty(shape=(), dtype=dtypes)
        for dtype in dtypes:
            key = dtype[0]
            velocity_model[key] = getattr(self.sMod.vMod, key)
        arrays['vMod'] = velocity_model
        arrays['vMod.layers'] = self.sMod.vMod.layers
        # finally save the collection of (structured) arrays to binary file
        np.savez_compressed(filename, **arrays)

    @staticmethod
    def deserialize(filename):
        """
        Deserialize model from numpy npz binary file.
        """
        # XXX: Make this a with statement when old NumPy support is dropped.
        npz = np.load(filename)
        try:
            model = TauModel(sMod=None, skip_calc=True)
            complex_contents = [
                'tauBranches', 'sMod', 'vMod',
                'sMod.PLayers', 'sMod.SLayers', 'sMod.criticalDepths',
                'sMod.fluidLayerDepths', 'sMod.highSlownessLayerDepthsP',
                'sMod.highSlownessLayerDepthsS', 'vMod.layers']
            # a) handle simple contents
            for key in npz.keys():
                # we have multiple, dynamic key names for individual tau
                # branches now, skip them all
                if key in complex_contents or key.startswith('tauBranches'):
                    continue
                arr = npz[key]
                if arr.ndim == 0:
                    arr = arr[()]
                setattr(model, key, arr)
            # b) handle .tauBranches
            tau_branch_keys = [key for key in npz.keys()
                               if key.startswith('tauBranches_')]
            j, i = tau_branch_keys[0].split("_")[1:]
            i = int(i.split("/")[1])
            j = int(j.split("/")[1])
            branches = np.empty(shape=(i, j), dtype=np.object_)
            for key in tau_branch_keys:
                j_, i_ = key.split("_")[1:]
                i_ = int(i_.split("/")[0])
                j_ = int(j_.split("/")[0])
                branches[i_][j_] = TauBranch._from_array(npz[key])
            # no idea how numpy lays out empty arrays of object type,
            # make a copy just in case..
            branches = np.copy(branches)
            setattr(model, "tauBranches", branches)
            # c) handle simple contents of .sMod
            slowness_model = SlownessModel(vMod=None, skip_model_creation=True)
            setattr(model, "sMod", slowness_model)
            for key in npz['sMod'].dtype.names:
                # restore scalar types from 0d array
                arr = npz['sMod'][key]
                if arr.ndim == 0:
                    arr = arr.flatten()[0]
                setattr(slowness_model, key, arr)
            # d) handle complex contents of .sMod
            for key in ['PLayers', 'SLayers', 'criticalDepths']:
                setattr(slowness_model, key, npz['sMod.' + key])
            for key in ['fluidLayerDepths', 'highSlownessLayerDepthsP',
                        'highSlownessLayerDepthsS']:
                arr_ = npz['sMod.' + key]
                if len(arr_) == 0:
                    data = []
                else:
                    data = [DepthRange._from_array(x) for x in arr_]
                setattr(slowness_model, key, data)
            # e) handle .sMod.vMod
            velocity_model = VelocityModel()
            setattr(slowness_model, "vMod", velocity_model)
            for key in npz['vMod'].dtype.names:
                # restore scalar types from 0d array
                arr = npz['vMod'][key]
                if arr.ndim == 0:
                    arr = arr.flatten()[0]
                setattr(velocity_model, key, arr)
            setattr(velocity_model, 'layers', npz['vMod.layers'])
            setattr(velocity_model, 'modelName',
                    native_str(velocity_model.modelName))
        finally:
            if hasattr(npz, 'close'):
                npz.close()
            else:
                del npz
        return model

    @staticmethod
    def from_file(model_name):
        if os.path.exists(model_name):
            filename = model_name
        else:
            filename = os.path.join(os.path.dirname(__file__), "data",
                                    model_name.lower() + ".npz")
        return TauModel.deserialize(filename)
