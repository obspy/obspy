#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Velocity model class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os

import numpy as np

from .velocity_layer import (DEFAULT_QP, DEFAULT_QS, VelocityLayer,
                             evaluateVelocityAt)


class VelocityModel(object):
    # Some default values as class attributes [km]
    radiusOfEarth = 6371.0
    default_moho = 35
    default_cmb = 2889.0
    default_iocb = 5153.9

    def __init__(self, modelName="unknown",
                 radiusOfEarth=radiusOfEarth, mohoDepth=default_moho,
                 cmbDepth=default_cmb, iocbDepth=default_iocb,
                 minRadius=0.0, maxRadius=6371.0, isSpherical=True,
                 layers=None):
        """
        Create an object to store a seismic Earth model.

        :type modelName: str
        :param modelName: name of the velocity model.
        :type radiusOfEarth: float
        :param radiusOfEarth: reference radius (km), usually radius of the
            Earth.
        :type mohoDepth: float
        :param mohoDepth: Depth (km) of the Moho. It can be input from
            velocity model (``*.nd``) or should be explicitly set. By default
            it is 35 kilometers (from IASP91).  For phase naming, the tau model
            will choose the closest first order discontinuity. Thus for most
            simple Earth models these values are satisfactory. Take proper care
            if your model has a thicker crust and a discontinuity near 35 km
            depth.
        :type cmbDepth: float
        :param cmbDepth: Depth (km) of the CMB (core mantle boundary). It can
            be input from velocity model (``*.nd``) or should be explicitly
            set. By default it is 2889 kilometers (from IASP91). For phase
            naming, the tau model will choose the closest 1st order
            discontinuity. Thus for most simple Earth models these values are
            satisfactory.
        :type iocbDepth: float
        :param iocbDepth: Depth (km) of the IOCB (inner core-outer core
            boundary). It can be input from velocity model (``*.nd``) or should
            be explicitly set. By default it is 5153.9 kilometers (from
            IASP91). For phase naming, the tau model will choose the closest
            first order discontinuity. Thus for most simple Earth models these
            values are satisfactory.
        :type minRadius: float
        :param minRadius: Minimum radius of the model (km).
        :type maxRadius: float
        :param maxRadius: Maximum radius of the model (km).
        :type isSpherical: bool
        :param isSpherical: Is this a spherical model? Defaults to true.
        """
        self.modelName = modelName
        self.radiusOfEarth = radiusOfEarth
        self.mohoDepth = mohoDepth
        self.cmbDepth = cmbDepth
        self.iocbDepth = iocbDepth
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.isSpherical = isSpherical
        self.layers = np.array(layers if layers is not None else [],
                               dtype=VelocityLayer)

    def __len__(self):
        return len(self.layers)

    # @property  ?
    def getDisconDepths(self):
        """
        Return the depths of discontinuities within the velocity model.

        :rtype: :class:`~numpy.ndarray`
        """
        above = self.layers[:-1]
        below = self.layers[1:]
        mask = np.logical_or(above['botPVelocity'] != below['topPVelocity'],
                             above['botSVelocity'] != below['topSVelocity'])

        discontinuities = np.empty((mask != 0).sum() + 2)
        discontinuities[0] = self.layers[0]['topDepth']
        discontinuities[1:-1] = above[mask]['botDepth']
        discontinuities[-1] = self.layers[-1]['botDepth']

        return discontinuities

    def getNumLayers(self):
        """
        Return the number of layers in this velocity model.

        :rtype: int
        """
        return len(self.layers)

    def layerNumberAbove(self, depth):
        """
        Find the layer containing the given depth(s).

        Note this returns the upper layer if the depth happens to be at a layer
        boundary.

        .. seealso:: :meth:`layerNumberBelow`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`

        :returns: The layer number for the specified depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape equivalent to ``depth``)
        """
        depth = np.atleast_1d(depth)
        layer = np.logical_and(
            self.layers['topDepth'][np.newaxis, :] < depth[:, np.newaxis],
            depth[:, np.newaxis] <= self.layers['botDepth'][np.newaxis, :])
        layer = np.where(layer)[-1]
        if len(layer):
            return layer
        else:
            raise LookupError("No such layer.")

    def layerNumberBelow(self, depth):
        """
        Find the layer containing the given depth(s).

        Note this returns the lower layer if the depth happens to be at a layer
        boundary.

        .. seealso:: :meth:`layerNumberAbove`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`

        :returns: The layer number for the specified depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape equivalent to ``depth``)
        """
        depth = np.atleast_1d(depth)
        layer = np.logical_and(
            self.layers['topDepth'][np.newaxis, :] <= depth[:, np.newaxis],
            depth[:, np.newaxis] < self.layers['botDepth'][np.newaxis, :])
        layer = np.where(layer)[-1]
        if len(layer):
            return layer
        else:
            raise LookupError("No such layer.")

    def evaluateAbove(self, depth, prop):
        """
        Return the value of the given material property at the given depth(s).

        Note this returns the value at the bottom of the upper layer if the
        depth happens to be at a layer boundary.

        .. seealso:: :meth:`evaluateBelow`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`
        :param prop: The material property to evaluate. One of:

            * ``p``
                Compressional (P) velocity (km/s)
            * ``s``
                Shear (S) velocity (km/s)
            * ``r`` or ``d``
                Density (in g/cm^3)
        :type prop: str

        :returns: The value of the given material property
        :rtype: :class:`float` or :class:`~numpy.ndarray` (dtype =
            :class:`float`, shape equivalent to ``depth``)
        """
        layer = self.layers[self.layerNumberAbove(depth)]
        return evaluateVelocityAt(layer, depth, prop)

    def evaluateBelow(self, depth, prop):
        """
        Return the value of the given material property at the given depth(s).

        Note this returns the value at the top of the lower layer if the depth
        happens to be at a layer boundary.

        .. seealso:: :meth:`evaluateBelow`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`
        :param prop: The material property to evaluate. One of:

            * ``p``
                Compressional (P) velocity (km/s)
            * ``s``
                Shear (S) velocity (km/s)
            * ``r`` or ``d``
                Density (in g/cm^3)
        :type prop: str

        :returns: the value of the given material property
        :rtype: :class:`float` or :class:`~numpy.ndarray` (dtype =
            :class:`float`, shape equivalent to ``depth``)
        """
        layer = self.layers[self.layerNumberBelow(depth)]
        return evaluateVelocityAt(layer, depth, prop)

    def depthAtTop(self, layer):
        """
        Return the depth at the top of the given layer.

        .. seealso:: :meth:`depthAtBottom`

        :param layer: The layer number
        :type layer: :class:`int` or :class:`~numpy.ndarray`

        :returns: The depth of the top, in km.
        :rtype: :class:`float` or :class:`~numpy.ndarray` (dtype =
            :class:`float`, shape equivalent to ``layer``)
        """
        layer = self.layers[layer]
        return layer['topDepth']

    def depthAtBottom(self, layer):
        """
        Return the depth at the bottom of the given layer.

        .. seealso:: :meth:`depthAtTop`

        :param layer: The layer number
        :type layer: :class:`int` or :class:`~numpy.ndarray`

        :returns: The depth of the bottom, in km.
        :rtype: :class:`float` or :class:`~numpy.ndarray` (dtype =
            :class:`float`, shape equivalent to ``layer``)
        """
        layer = self.layers[layer]
        return layer['botDepth']

    def validate(self):
        """
        Perform internal consistency checks on the velocity model.

        :returns: True if the model is consistent.
        :raises ValueError: If the model is inconsistent.
        """
        # Is radiusOfEarth positive?
        if self.radiusOfEarth <= 0.0:
            raise ValueError("Radius of earth is not positive: %f" % (
                self.radiusOfEarth, ))

        # Is mohoDepth non-negative?
        if self.mohoDepth < 0.0:
            raise ValueError("mohoDepth is not non-negative: %f" % (
                self.mohoDepth, ))

        # Is cmbDepth >= mohoDepth?
        if self.cmbDepth < self.mohoDepth:
            raise ValueError("cmbDepth (%f) < mohoDepth (%f)" % (
                self.cmbDepth,
                self.mohoDepth))

        # Is cmbDepth positive?
        if self.cmbDepth <= 0.0:
            raise ValueError("cmbDepth is not positive: %f" % (
                self.cmbDepth, ))

        # Is iocbDepth >= cmbDepth?
        if self.iocbDepth < self.cmbDepth:
            raise ValueError("iocbDepth (%f) < cmbDepth (%f)" % (
                self.iocbDepth,
                self.cmbDepth))

        # Is iocbDepth positive?
        if self.iocbDepth <= 0.0:
            raise ValueError("iocbDepth is not positive: %f" % (
                self.iocbDepth, ))

        # Is minRadius non-negative?
        if self.minRadius < 0.0:
            raise ValueError("minRadius is not non-negative: %f " % (
                self.minRadius, ))

        # Is maxRadius non-negative?
        if self.maxRadius <= 0.0:
            raise ValueError("maxRadius is not positive: %f" % (
                self.maxRadius, ))

        # Is maxRadius > minRadius?
        if self.maxRadius <= self.minRadius:
            raise ValueError("maxRadius (%f) <= minRadius (%f)" % (
                self.maxRadius,
                self.minRadius))

        # Check for gaps
        gaps = self.layers[:-1]['botDepth'] != self.layers[1:]['topDepth']
        gaps = np.where(gaps)[0]
        if gaps:
            msg = ("There is a gap in the velocity model between layer(s) %s "
                   "and %s.\n%s" % (gaps, gaps + 1, self.layers[gaps]))
            raise ValueError(msg)

        # Check for zero thickness
        probs = self.layers['botDepth'] == self.layers['topDepth']
        probs = np.where(probs)[0]
        if probs:
            msg = ("There is a zero thickness layer in the velocity model at "
                   "layer(s) %s\n%s" % (probs, self.layers[probs]))
            raise ValueError(msg)

        # Check for negative P velocity
        probs = np.logical_or(self.layers['topPVelocity'] <= 0.0,
                              self.layers['botPVelocity'] <= 0.0)
        probs = np.where(probs)[0]
        if probs:
            msg = ("There is a negative P velocity layer in the velocity "
                   "model at layer(s) %s\n%s" % (probs, self.layers[probs]))
            raise ValueError(msg)

        # Check for negative S velocity
        probs = np.logical_or(self.layers['topSVelocity'] < 0.0,
                              self.layers['botSVelocity'] < 0.0)
        probs = np.where(probs)[0]
        if probs:
            msg = ("There is a negative S velocity layer in the velocity "
                   "model at layer(s) %s\n%s" % (probs, self.layers[probs]))
            raise ValueError(msg)

        # Check for zero P velocity
        probs = np.logical_or(
            np.logical_and(self.layers['topPVelocity'] != 0.0,
                           self.layers['botPVelocity'] == 0.0),
            np.logical_and(self.layers['topPVelocity'] == 0.0,
                           self.layers['botPVelocity'] != 0.0))
        probs = np.where(probs)[0]
        if probs:
            msg = ("There is a layer that goes to zero P velocity (top or "
                   "bottom) without a discontinuity in the velocity model at "
                   "layer(s) %s\nThis would cause a divide by zero within "
                   "this depth range. Try making the velocity small, followed "
                   "by a discontinuity to zero velocity.\n%s")
            raise ValueError(msg % (probs, self.layers[probs]))

        # Check for negative S velocity
        probs = np.logical_or(
            np.logical_and(self.layers['topSVelocity'] != 0.0,
                           self.layers['botSVelocity'] == 0.0),
            np.logical_and(self.layers['topSVelocity'] == 0.0,
                           self.layers['botSVelocity'] != 0.0))
        # This warning will always pop up for the top layer even
        #  in IASP91, therefore ignore it.
        probs = np.logical_and(probs, self.layers['topDepth'] != 0)
        probs = np.where(probs)[0]
        if probs:
            msg = ("There is a layer that goes to zero S velocity (top or "
                   "bottom) without a discontinuity in the velocity model at "
                   "layer(s) %s\nThis would cause a divide by zero within "
                   "this depth range. Try making the velocity small, followed "
                   "by a discontinuity to zero velocity.\n%s")
            raise ValueError(msg % (probs, self.layers[probs]))

        return True

    def __str__(self):
        desc = "modelName=" + str(self.modelName) + "\n" + \
               "\n radiusOfEarth=" + str(
            self.radiusOfEarth) + "\n mohoDepth=" + str(self.mohoDepth) + \
            "\n cmbDepth=" + str(self.cmbDepth) + "\n iocbDepth=" + \
            str(self.iocbDepth) + "\n minRadius=" + str(
            self.minRadius) + "\n maxRadius=" + str(self.maxRadius) + \
            "\n spherical=" + str(self.isSpherical)
        # desc += "\ngetNumLayers()=" + str(self.getNumLayers()) + "\n"
        return desc

    @classmethod
    def readVelocityFile(cls, filename):
        """
        Read in a velocity file.

        The type of file is determined from the file name (changed from the
        Java!).

        :param filename: The name of the file to read.
        :type filename: str

        :raises NotImplementedError: If the file extension is ``.nd``.
        :raises ValueError: If the file extension is not ``.tvel``.
        """
        if filename.endswith(".nd"):
            vMod = cls.read_nd_file(filename)
        elif filename.endswith(".tvel"):
            vMod = cls.readTVelFile(filename)
        else:
            raise ValueError("File type could not be determined, please "
                             "rename your file to end with .tvel or .nd")

        vMod.fixDisconDepths()
        return vMod

    @classmethod
    def readTVelFile(cls, filename):
        """
        Read in a velocity model from a "tvel" ASCII text file.

        The name of the model file for model "modelname" should be
        "modelname.tvel".  The format of the file is::

            comment line - generally info about the P velocity model
            comment line - generally info about the S velocity model
            depth pVel sVel Density
            depth pVel sVel Density

        The velocities are assumed to be linear between sample points. Because
        this type of model file doesn't give complete information we make the
        following assumptions:

        * ``modelname`` - from the filename, with ".tvel" dropped if present
        * ``radiusOfEarth`` - the largest depth in the model
        * ``meanDensity`` - 5517.0
        * ``G`` - 6.67e-11

        Comments using ``#`` are also allowed.

        :param filename: The name of the file to read.
        :type filename: str

        :raises ValueError: If model file is in error.
        """
        # Read all lines in the file. Each Layer needs top and bottom values,
        # i.e. info from two lines.
        data = np.genfromtxt(filename, skip_header=2, comments='#')

        # Check if density is present.
        if data.shape[1] < 4:
            raise ValueError("Top density not specified.")

        # Check that relative speed are sane.
        mask = data[:, 2] > data[:, 1]
        if np.any(mask):
            raise ValueError(
                "S velocity is greater than the P velocity\n" +
                str(data[mask]))

        layers = np.empty(data.shape[0] - 1, dtype=VelocityLayer)

        layers['topDepth'] = data[:-1, 0]
        layers['botDepth'] = data[1:, 0]

        layers['topPVelocity'] = data[:-1, 1]
        layers['botPVelocity'] = data[1:, 1]

        layers['topSVelocity'] = data[:-1, 2]
        layers['botSVelocity'] = data[1:, 2]

        layers['topDensity'] = data[:-1, 3]
        layers['botDensity'] = data[1:, 3]

        # We do not at present support varying attenuation
        layers['topQp'].fill(DEFAULT_QP)
        layers['botQp'].fill(DEFAULT_QP)
        layers['topQs'].fill(DEFAULT_QS)
        layers['botQs'].fill(DEFAULT_QS)

        # Don't use zero thickness layers; first order discontinuities are
        # taken care of by storing top and bottom depths.
        mask = layers['topDepth'] == layers['botDepth']
        layers = layers[~mask]

        radiusOfEarth = data[-1, 0]
        maxRadius = data[-1, 0]
        modelName = os.path.splitext(os.path.basename(filename))[0]
        # I assume that this is a whole earth model
        # so the maximum depth ==  maximum radius == earth radius.
        return VelocityModel(modelName, radiusOfEarth, cls.default_moho,
                             cls.default_cmb, cls.default_iocb, 0,
                             maxRadius, True, layers)

    @classmethod
    def read_nd_file(cls, filename):
        """
        Read in a velocity model from a "nd" ASCII text file.

        This method reads in a velocity model from a "nd" ASCII text file, the
        format used by Xgbm. The name of the model file for model "modelname"
        should be "modelname.nd".

        The format of the file is:

        depth pVel sVel Density Qp Qs

        depth pVel sVel Density Qp Qs

        . . . with each major boundary separated with a line with "mantle",
        "outer-core" or "inner-core". "moho", "cmb" and "icocb" are allowed
        as synonyms respectively.

        This feature makes phase interpretation much easier to
        code. Also, as they are not needed for travel time calculations, the
        density, Qp and Qs may be omitted.

        The velocities are assumed to be linear between sample points. Because
        this type of model file doesn't give complete information we make the
        following assumptions:

        modelname - from the filename, with ".nd" dropped, if present

        radiusOfEarth - the largest depth in the model

        Comments are allowed. # signifies that the rest of the
        line is a comment.  If # is the first character in a line, the line is
        ignored

        :param filename: The name of the file to read.
        :type filename: str

        :raises ValueError: If model file is in error.
        """
        moho_depth = cls.default_moho
        cmb_depth = cls.default_cmb
        iocb_depth = cls.default_iocb

        # Read all lines from file to enable identifying top and bottom values
        # for each layer and find named discontinuities if present
        with open(filename) as modfile:
            lines = modfile.readlines()

        # Loop through to fill data array and locate named discontinuities
        ii = 0
        for line in lines:
            # Strip off anything after '#'
            line = line.split('#')[0].split()
            if not line:  # Skip empty or comment lines
                continue
            if ii == 0:
                data = []
                for item in line:
                    data.append(float(item))
                data = np.array(data)
                ii = ii + 1
            else:
                if len(line) == 1:  # Named discontinuity
                    if ((line[0].lower() == 'mantle') or (line[0].lower() ==
                                                          'moho')):
                        moho_depth = data[ii - 1, 0]
                    elif ((line[0].lower() == 'outer-core') or
                          (line[0].lower() == 'cmb')):
                        cmb_depth = data[ii - 1, 0]
                    elif ((line[0].lower() == 'inner-core') or
                          (line[0].lower() == 'iocb')):
                        iocb_depth = data[ii - 1, 0]
                    else:
                        raise ValueError("Unrecognized discontinuity name: " +
                                         str(line[0]))
                else:
                    row = []
                    for item in line:
                        row.append(float(item))
                    data = np.vstack((data, np.array(row)))
                    ii = ii + 1

        # Check if density is present.
        if data.shape[1] < 4:
            raise ValueError("Top density not specified.")

        # Check that relative speed are sane.
        mask = data[:, 2] > data[:, 1]
        if np.any(mask):
            raise ValueError(
                "S velocity is greater than the P velocity\n" +
                str(data[mask]))

        layers = np.empty(data.shape[0] - 1, dtype=VelocityLayer)

        layers['topDepth'] = data[:-1, 0]
        layers['botDepth'] = data[1:, 0]

        layers['topPVelocity'] = data[:-1, 1]
        layers['botPVelocity'] = data[1:, 1]

        layers['topSVelocity'] = data[:-1, 2]
        layers['botSVelocity'] = data[1:, 2]

        layers['topDensity'] = data[:-1, 3]
        layers['botDensity'] = data[1:, 3]

        # We do not at present support varying attenuation
        layers['topQp'].fill(DEFAULT_QP)
        layers['botQp'].fill(DEFAULT_QP)
        layers['topQs'].fill(DEFAULT_QS)
        layers['botQs'].fill(DEFAULT_QS)

        # Don't use zero thickness layers; first order discontinuities are
        # taken care of by storing top and bottom depths.
        mask = layers['topDepth'] == layers['botDepth']
        layers = layers[~mask]

        radiusOfEarth = data[-1, 0]
        maxRadius = data[-1, 0]
        modelName = os.path.splitext(os.path.basename(filename))[0]
        # I assume that this is a whole earth model
        # so the maximum depth ==  maximum radius == earth radius.
        return VelocityModel(modelName, radiusOfEarth, moho_depth,
                             cmb_depth, iocb_depth, 0,
                             maxRadius, True, layers)

    def fixDisconDepths(self):
        """
        Reset depths of major discontinuities.

        The depths are set to match those existing in the input velocity model.
        The initial values are set such that if there is no discontinuity
        within the top 100 km then the Moho is set to 0.0. Similarly, if there
        are no discontinuities at all then the CMB is set to the radius of the
        Earth. Similarly for the IOCB, except it must be a fluid to solid
        boundary and deeper than 100 km to avoid problems with shallower fluid
        layers, e.g., oceans.
        """
        MOHO_MIN = 65.0
        CMB_MIN = self.radiusOfEarth
        IOCB_MIN = self.radiusOfEarth - 100.0

        changeMade = False
        tempMohoDepth = 0.0
        tempCmbDepth = self.radiusOfEarth
        tempIocbDepth = self.radiusOfEarth

        above = self.layers[:-1]
        below = self.layers[1:]
        # Only look for discontinuities:
        mask = np.logical_or(above['botPVelocity'] != below['topPVelocity'],
                             above['botSVelocity'] != below['topSVelocity'])

        # Find discontinuity closest to current Moho
        moho_diff = np.abs(self.mohoDepth - above['botDepth'])
        moho_diff[~mask] = MOHO_MIN
        moho = np.argmin(moho_diff)
        if moho_diff[moho] < MOHO_MIN:
            tempMohoDepth = above[moho]['botDepth']

        # Find discontinuity closest to current CMB
        cmb_diff = np.abs(self.cmbDepth - above['botDepth'])
        cmb_diff[~mask] = CMB_MIN
        cmb = np.argmin(cmb_diff)
        if cmb_diff[cmb] < CMB_MIN:
            tempCmbDepth = above[cmb]['botDepth']

        # Find discontinuity closest to current IOCB
        iocb_diff = self.iocbDepth - above['botDepth']
        iocb_diff[~mask] = IOCB_MIN
        # IOCB must transition from S==0 to S!=0
        iocb_diff[above['botSVelocity'] != 0.0] = IOCB_MIN
        iocb_diff[below['topSVelocity'] <= 0.0] = IOCB_MIN
        iocb = np.argmin(iocb_diff)
        if iocb_diff[iocb] < IOCB_MIN:
            tempIocbDepth = above[iocb]['botDepth']

        if self.mohoDepth != tempMohoDepth \
                or self.cmbDepth != tempCmbDepth \
                or self.iocbDepth != tempIocbDepth:
            changeMade = True
        self.mohoDepth = tempMohoDepth
        self.cmbDepth = tempCmbDepth
        self.iocbDepth = (tempIocbDepth
                          if tempCmbDepth != tempIocbDepth
                          else self.radiusOfEarth)
        return changeMade
