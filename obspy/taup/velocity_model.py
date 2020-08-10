# -*- coding: utf-8 -*-
"""
Velocity model class.
"""
import os

import numpy as np

from .velocity_layer import VelocityLayer, evaluate_velocity_at
from . import _DEFAULT_VALUES


class VelocityModel(object):
    def __init__(self, model_name, radius_of_planet, min_radius, max_radius,
                 moho_depth, cmb_depth, iocb_depth, is_spherical,
                 layers=None):
        """
        Object for storing a seismic planet model.

        :type model_name: str
        :param model_name: name of the velocity model.
        :type radius_of_planet: float
        :param radius_of_planet: reference radius (km), usually radius of the
            planet.
        :type min_radius: float
        :param min_radius: Minimum radius of the model (km).
        :type max_radius: float
        :param max_radius: Maximum radius of the model (km).
        :type moho_depth: float
        :param moho_depth: Depth (km) of the Moho. It can be input from
            velocity model (``*.nd``) or should be explicitly set. For phase
            naming, the tau model will choose the closest first order
            discontinuity.
        :type cmb_depth: float
        :param cmb_depth: Depth (km) of the CMB (core mantle boundary). It can
            be input from velocity model (``*.nd``) or should be explicitly
            set.
        :type iocb_depth: float
        :param iocb_depth: Depth (km) of the IOCB (inner core-outer core
            boundary). It can be input from velocity model (``*.nd``) or should
            be explicitly set.
        :type is_spherical: bool
        :param is_spherical: Is this a spherical model? Defaults to true.
        :type layers: list
        :param layers: The layers of the model.
        """
        self.model_name = model_name
        self.radius_of_planet = radius_of_planet
        self.moho_depth = moho_depth
        self.cmb_depth = cmb_depth
        self.iocb_depth = iocb_depth
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.is_spherical = is_spherical
        self.layers = np.array(layers if layers is not None else [],
                               dtype=VelocityLayer)

    def __len__(self):
        return len(self.layers)

    def is_discontinuity(self, depth):
        return np.any(self.get_discontinuity_depths() == depth)

    def get_discontinuity_depths(self):
        """
        Return the depths of discontinuities within the velocity model.

        :rtype: :class:`~numpy.ndarray`
        """
        above = self.layers[:-1]
        below = self.layers[1:]
        mask = np.logical_or(
            above['bot_p_velocity'] != below['top_p_velocity'],
            above['bot_s_velocity'] != below['top_s_velocity'])

        discontinuities = np.empty((mask != 0).sum() + 2)
        discontinuities[0] = self.layers[0]['top_depth']
        discontinuities[1:-1] = above[mask]['bot_depth']
        discontinuities[-1] = self.layers[-1]['bot_depth']

        return discontinuities

    def layer_number_above(self, depth):
        """
        Find the layer containing the given depth(s).

        Note this returns the upper layer if the depth happens to be at a layer
        boundary.

        .. seealso:: :meth:`layer_number_below`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`

        :returns: The layer number for the specified depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape equivalent to ``depth``)
        """
        depth = np.atleast_1d(depth)
        layer = np.logical_and(
            self.layers['top_depth'][np.newaxis, :] < depth[:, np.newaxis],
            depth[:, np.newaxis] <= self.layers['bot_depth'][np.newaxis, :])
        layer = np.where(layer)[-1]
        if len(layer):
            return layer
        else:
            raise LookupError("No such layer.")

    def layer_number_below(self, depth):
        """
        Find the layer containing the given depth(s).

        Note this returns the lower layer if the depth happens to be at a layer
        boundary.

        .. seealso:: :meth:`layer_number_above`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`

        :returns: The layer number for the specified depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape equivalent to ``depth``)
        """
        depth = np.atleast_1d(depth)
        layer = np.logical_and(
            self.layers['top_depth'][np.newaxis, :] <= depth[:, np.newaxis],
            depth[:, np.newaxis] < self.layers['bot_depth'][np.newaxis, :])
        layer = np.where(layer)[-1]
        if len(layer):
            return layer
        else:
            raise LookupError("No such layer.")

    def evaluate_above(self, depth, prop):
        """
        Return the value of the given material property at the given depth(s).

        Note this returns the value at the bottom of the upper layer if the
        depth happens to be at a layer boundary.

        .. seealso:: :meth:`evaluate_below`

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
        layer = self.layers[self.layer_number_above(depth)]
        return evaluate_velocity_at(layer, depth, prop)

    def evaluate_below(self, depth, prop):
        """
        Return the value of the given material property at the given depth(s).

        Note this returns the value at the top of the lower layer if the depth
        happens to be at a layer boundary.

        .. seealso:: :meth:`evaluate_below`

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
        layer = self.layers[self.layer_number_below(depth)]
        return evaluate_velocity_at(layer, depth, prop)

    def depth_at_top(self, layer):
        """
        Return the depth at the top of the given layer.

        .. seealso:: :meth:`depth_at_bottom`

        :param layer: The layer number
        :type layer: :class:`int` or :class:`~numpy.ndarray`

        :returns: The depth of the top, in km.
        :rtype: :class:`float` or :class:`~numpy.ndarray` (dtype =
            :class:`float`, shape equivalent to ``layer``)
        """
        layer = self.layers[layer]
        return layer['top_depth']

    def depth_at_bottom(self, layer):
        """
        Return the depth at the bottom of the given layer.

        .. seealso:: :meth:`depth_at_top`

        :param layer: The layer number
        :type layer: :class:`int` or :class:`~numpy.ndarray`

        :returns: The depth of the bottom, in km.
        :rtype: :class:`float` or :class:`~numpy.ndarray` (dtype =
            :class:`float`, shape equivalent to ``layer``)
        """
        layer = self.layers[layer]
        return layer['bot_depth']

    def validate(self):
        """
        Perform internal consistency checks on the velocity model.

        :returns: True if the model is consistent.
        :raises ValueError: If the model is inconsistent.
        """
        # Is radius_of_planet positive?
        if self.radius_of_planet <= 0.0:
            raise ValueError("Radius of the planet is not positive: %f" % (
                self.radius_of_planet, ))

        # Is moho_depth non-negative?
        if self.moho_depth < 0.0:
            raise ValueError("moho_depth is not non-negative: %f" % (
                self.moho_depth, ))

        # Is cmb_depth >= moho_depth?
        if self.cmb_depth < self.moho_depth:
            raise ValueError("cmb_depth (%f) < moho_depth (%f)" % (
                self.cmb_depth,
                self.moho_depth))

        # Is cmb_depth positive?
        if self.cmb_depth <= 0.0:
            raise ValueError("cmb_depth is not positive: %f" % (
                self.cmb_depth, ))

        # Is iocb_depth >= cmb_depth?
        if self.iocb_depth < self.cmb_depth:
            raise ValueError("iocb_depth (%f) < cmb_depth (%f)" % (
                self.iocb_depth,
                self.cmb_depth))

        # Is iocb_depth positive?
        if self.iocb_depth <= 0.0:
            raise ValueError("iocb_depth is not positive: %f" % (
                self.iocb_depth, ))

        # Is min_radius non-negative?
        if self.min_radius < 0.0:
            raise ValueError("min_radius is not non-negative: %f " % (
                self.min_radius, ))

        # Is max_radius non-negative?
        if self.max_radius <= 0.0:
            raise ValueError("max_radius is not positive: %f" % (
                self.max_radius, ))

        # Is max_radius > min_radius?
        if self.max_radius <= self.min_radius:
            raise ValueError("max_radius (%f) <= min_radius (%f)" % (
                self.max_radius,
                self.min_radius))

        # Check for gaps
        gaps = self.layers[:-1]['bot_depth'] != self.layers[1:]['top_depth']
        gaps = np.where(gaps)[0]
        if gaps.size:
            msg = ("There is a gap in the velocity model between layer(s) %s "
                   "and %s.\n%s" % (gaps, gaps + 1, self.layers[gaps]))
            raise ValueError(msg)

        # Check for zero thickness
        probs = self.layers['bot_depth'] == self.layers['top_depth']
        probs = np.where(probs)[0]
        if probs.size:
            msg = ("There is a zero thickness layer in the velocity model at "
                   "layer(s) %s\n%s" % (probs, self.layers[probs]))
            raise ValueError(msg)

        # Check for negative P velocity
        probs = np.logical_or(self.layers['top_p_velocity'] <= 0.0,
                              self.layers['bot_p_velocity'] <= 0.0)
        probs = np.where(probs)[0]
        if probs.size:
            msg = ("There is a negative P velocity layer in the velocity "
                   "model at layer(s) %s\n%s" % (probs, self.layers[probs]))
            raise ValueError(msg)

        # Check for negative S velocity
        probs = np.logical_or(self.layers['top_s_velocity'] < 0.0,
                              self.layers['bot_s_velocity'] < 0.0)
        probs = np.where(probs)[0]
        if probs.size:
            msg = ("There is a negative S velocity layer in the velocity "
                   "model at layer(s) %s\n%s" % (probs, self.layers[probs]))
            raise ValueError(msg)

        # Check for zero P velocity
        probs = np.logical_or(
            np.logical_and(self.layers['top_p_velocity'] != 0.0,
                           self.layers['bot_p_velocity'] == 0.0),
            np.logical_and(self.layers['top_p_velocity'] == 0.0,
                           self.layers['bot_p_velocity'] != 0.0))
        probs = np.where(probs)[0]
        if probs.size:
            msg = ("There is a layer that goes to zero P velocity (top or "
                   "bottom) without a discontinuity in the velocity model at "
                   "layer(s) %s\nThis would cause a divide by zero within "
                   "this depth range. Try making the velocity small, followed "
                   "by a discontinuity to zero velocity.\n%s")
            raise ValueError(msg % (probs, self.layers[probs]))

        # Check for negative S velocity
        probs = np.logical_or(
            np.logical_and(self.layers['top_s_velocity'] != 0.0,
                           self.layers['bot_s_velocity'] == 0.0),
            np.logical_and(self.layers['top_s_velocity'] == 0.0,
                           self.layers['bot_s_velocity'] != 0.0))
        # This warning will always pop up for the top layer even
        #  in IASP91, therefore ignore it.
        probs = np.logical_and(probs, self.layers['top_depth'] != 0)
        probs = np.where(probs)[0]
        if probs.size:
            msg = ("There is a layer that goes to zero S velocity (top or "
                   "bottom) without a discontinuity in the velocity model at "
                   "layer(s) %s\nThis would cause a divide by zero within "
                   "this depth range. Try making the velocity small, followed "
                   "by a discontinuity to zero velocity.\n%s")
            raise ValueError(msg % (probs, self.layers[probs]))

        return True

    def __str__(self):
        desc = "model_name=" + str(self.model_name) + "\n" + \
               "\n radius_of_planet=" + str(
            self.radius_of_planet) + "\n moho_depth=" + \
            str(self.moho_depth) + \
            "\n cmb_depth=" + str(self.cmb_depth) + "\n iocb_depth=" + \
            str(self.iocb_depth) + "\n min_radius=" + str(
            self.min_radius) + "\n max_radius=" + str(self.max_radius) + \
            "\n spherical=" + str(self.is_spherical)
        return desc

    @classmethod
    def read_velocity_file(cls, filename):
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
            v_mod = cls.read_nd_file(filename)
        elif filename.endswith(".tvel"):
            v_mod = cls.read_tvel_file(filename)
        else:
            raise ValueError("File type could not be determined, please "
                             "rename your file to end with .tvel or .nd")

        v_mod.fix_discontinuity_depths()
        return v_mod

    @classmethod
    def read_tvel_file(cls, filename):
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
        * ``radius_of_planet`` - the largest depth in the model
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

        layers['top_depth'] = data[:-1, 0]
        layers['bot_depth'] = data[1:, 0]

        layers['top_p_velocity'] = data[:-1, 1]
        layers['bot_p_velocity'] = data[1:, 1]

        layers['top_s_velocity'] = data[:-1, 2]
        layers['bot_s_velocity'] = data[1:, 2]

        layers['top_density'] = data[:-1, 3]
        layers['bot_density'] = data[1:, 3]

        # We do not at present support varying attenuation
        layers['top_qp'].fill(_DEFAULT_VALUES["qp"])
        layers['bot_qp'].fill(_DEFAULT_VALUES["qp"])
        layers['top_qs'].fill(_DEFAULT_VALUES["qs"])
        layers['bot_qs'].fill(_DEFAULT_VALUES["qs"])

        # Don't use zero thickness layers; first order discontinuities are
        # taken care of by storing top and bottom depths.
        mask = layers['top_depth'] == layers['bot_depth']
        layers = layers[~mask]

        # tvel files cannot have named discontinuities so it is really only
        # useful for Earth models. The exact radius is derived from the tvel
        # file, the depth of discontinuities are fixed.
        min_radius = 0
        max_radius = data[-1, 0]
        radius_of_planet = data[-1, 0]
        model_name = os.path.splitext(os.path.basename(filename))[0]

        return VelocityModel(
            model_name=model_name,
            radius_of_planet=radius_of_planet,
            min_radius=min_radius,
            max_radius=max_radius,
            moho_depth=_DEFAULT_VALUES["default_moho"],
            cmb_depth=_DEFAULT_VALUES["default_cmb"],
            iocb_depth=_DEFAULT_VALUES["default_iocb"],
            is_spherical=True,
            layers=layers)

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

        radius_of_planet - the largest depth in the model

        Comments are allowed. # signifies that the rest of the
        line is a comment.  If # is the first character in a line, the line is
        ignored

        :param filename: The name of the file to read.
        :type filename: str

        :raises ValueError: If model file is in error.
        """
        moho_depth = None
        cmb_depth = None
        iocb_depth = None

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
                    dc_name = line[0].lower()
                    if dc_name in ("mantle", "moho"):
                        moho_depth = data[ii - 1, 0]
                    elif dc_name in ("outer-core", "cmb"):
                        cmb_depth = data[ii - 1, 0]
                    elif dc_name in ("inner-core", "iocb"):
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

        if moho_depth is None:
            raise ValueError("Moho depth is not specified in model file!")
        if cmb_depth is None:
            raise ValueError("CMB depth is not specified in model file!")
        if iocb_depth is None:
            raise ValueError("IOCB depth is not specified in model file!")

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

        layers['top_depth'] = data[:-1, 0]
        layers['bot_depth'] = data[1:, 0]

        layers['top_p_velocity'] = data[:-1, 1]
        layers['bot_p_velocity'] = data[1:, 1]

        layers['top_s_velocity'] = data[:-1, 2]
        layers['bot_s_velocity'] = data[1:, 2]

        layers['top_density'] = data[:-1, 3]
        layers['bot_density'] = data[1:, 3]

        # We do not at present support varying attenuation
        layers['top_qp'].fill(_DEFAULT_VALUES["qp"])
        layers['bot_qp'].fill(_DEFAULT_VALUES["qp"])
        layers['top_qs'].fill(_DEFAULT_VALUES["qs"])
        layers['bot_qs'].fill(_DEFAULT_VALUES["qs"])

        # Don't use zero thickness layers; first order discontinuities are
        # taken care of by storing top and bottom depths.
        mask = layers['top_depth'] == layers['bot_depth']
        layers = layers[~mask]

        radius_of_planet = data[-1, 0]
        max_radius = data[-1, 0]
        model_name = os.path.splitext(os.path.basename(filename))[0]
        # I assume that this is a whole planet model
        # so the maximum depth ==  maximum radius == planet radius.
        return VelocityModel(
            model_name=model_name,
            radius_of_planet=radius_of_planet,
            min_radius=0, max_radius=max_radius,
            moho_depth=moho_depth, cmb_depth=cmb_depth, iocb_depth=iocb_depth,
            is_spherical=True, layers=layers)

    def fix_discontinuity_depths(self):
        """
        Reset depths of major discontinuities.

        The depths are set to match those existing in the input velocity model.
        The initial values are set such that if there is no discontinuity
        within the top 100 km then the Moho is set to 0.0. Similarly, if there
        are no discontinuities at all then the CMB is set to the radius of the
        planet. Similarly for the IOCB, except it must be a fluid to solid
        boundary and deeper than 100 km to avoid problems with shallower fluid
        layers, e.g., oceans.
        """
        moho_min = 65.0
        cmb_min = self.radius_of_planet
        iocb_min = self.radius_of_planet - 100.0

        change_made = False
        temp_moho_depth = 0.0
        temp_cmb_depth = self.radius_of_planet
        temp_iocb_depth = self.radius_of_planet

        above = self.layers[:-1]
        below = self.layers[1:]
        # Only look for discontinuities:
        mask = np.logical_or(
            above['bot_p_velocity'] != below['top_p_velocity'],
            above['bot_s_velocity'] != below['top_s_velocity'])

        # Find discontinuity closest to current Moho
        moho_diff = np.abs(self.moho_depth - above['bot_depth'])
        moho_diff[~mask] = moho_min
        moho = np.argmin(moho_diff)
        if moho_diff[moho] < moho_min:
            temp_moho_depth = above[moho]['bot_depth']

        # Find discontinuity closest to current CMB
        cmb_diff = np.abs(self.cmb_depth - above['bot_depth'])
        cmb_diff[~mask] = cmb_min
        cmb = np.argmin(cmb_diff)
        if cmb_diff[cmb] < cmb_min:
            temp_cmb_depth = above[cmb]['bot_depth']

        # Find discontinuity closest to current IOCB
        iocb_diff = self.iocb_depth - above['bot_depth']
        iocb_diff[~mask] = iocb_min
        # IOCB must transition from S==0 to S!=0
        iocb_diff[above['bot_s_velocity'] != 0.0] = iocb_min
        iocb_diff[below['top_s_velocity'] <= 0.0] = iocb_min
        iocb = np.argmin(iocb_diff)
        if iocb_diff[iocb] < iocb_min:
            temp_iocb_depth = above[iocb]['bot_depth']

        if self.moho_depth != temp_moho_depth \
                or self.cmb_depth != temp_cmb_depth \
                or self.iocb_depth != temp_iocb_depth:
            change_made = True
        self.moho_depth = temp_moho_depth
        self.cmb_depth = temp_cmb_depth
        self.iocb_depth = (temp_iocb_depth
                           if temp_cmb_depth != temp_iocb_depth
                           else self.radius_of_planet)
        return change_made
