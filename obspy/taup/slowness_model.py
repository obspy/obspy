# -*- coding: utf-8 -*-
"""
Slowness model class.
"""
from copy import deepcopy
import math

import numpy as np

from . import _DEFAULT_VALUES
from .helper_classes import (CriticalDepth, DepthRange, SlownessLayer,
                             SlownessModelError, SplitLayerInfo, TimeDist)
from .slowness_layer import (bullen_depth_for,
                             bullen_radial_slowness, create_from_vlayer,
                             evaluate_at_bullen)
from .velocity_layer import (VelocityLayer, evaluate_velocity_at_bottom,
                             evaluate_velocity_at_top)


def _fix_critical_depths(critical_depths, layer_num, is_p_wave):
    name = 'p_layer_num' if is_p_wave else 's_layer_num'

    mask = critical_depths[name] > layer_num
    critical_depths[name][mask] += 1


class SlownessModel(object):
    """
    Storage and methods for generating slowness-depth pairs.
    """
    def __init__(self, v_mod, min_delta_p=0.1, max_delta_p=11,
                 max_depth_interval=115,
                 max_range_interval=2.5 * math.pi / 180,
                 max_interp_error=0.05, allow_inner_core_s=True,
                 slowness_tolerance=_DEFAULT_VALUES["slowness_tolerance"],
                 skip_model_creation=False):
        self.debug = False
        # NB if the following are actually cleared (lists are mutable) every
        # time create_sample is called, maybe it would be better to just put
        # these initialisations into the relevant methods? They do have to be
        # persistent across method calls in create_sample though, so don't.

        # Stores the layer number for layers in the velocity model with a
        # critical point at their top. These form the "branches" of slowness
        # sampling.
        self.critical_depths = None  # will be list of CriticalDepth objects
        # Store depth ranges that contains a high slowness zone for P/S. Stored
        # as DepthRange objects, containing the top depth and bottom depth.
        self.high_slowness_layer_depths_p = []  # will be list of DepthRanges
        self.high_slowness_layer_depths_s = []
        # Stores depth ranges that are fluid, ie S velocity is zero. Stored as
        # DepthRange objects, containing the top depth and bottom depth.
        self.fluid_layer_depths = []
        self.p_layers = None
        self.s_layers = None
        # For methods that have an is_p_wave parameter
        self.s_wave = False
        self.p_wave = True

        self.v_mod = v_mod
        self.min_delta_p = min_delta_p
        self.max_delta_p = max_delta_p
        self.max_depth_interval = max_depth_interval
        self.max_range_interval = max_range_interval
        self.max_interp_error = max_interp_error
        self.allow_inner_core_s = allow_inner_core_s
        self.slowness_tolerance = slowness_tolerance
        if skip_model_creation:
            return
        self.create_sample()

    def __str__(self):
        desc = "".join([
            "radius_of_planet=", str(self.radius_of_planet), "\n max_delta_p=",
            str(self.max_delta_p),
            "\n min_delta_p=", str(self.min_delta_p), "\n max_depth_interval=",
            str(self.max_depth_interval), "\n max_range_interval=",
            str(self.max_range_interval),
            "\n allow_inner_core_s=", str(self.allow_inner_core_s),
            "\n slownessTolerance=", str(self.slowness_tolerance),
            "\n get_num_layers('P')=", str(self.get_num_layers(self.p_wave)),
            "\n get_num_layers('S')=", str(self.get_num_layers(self.s_wave)),
            "\n fluid_layer_depths.size()=", str(len(self.fluid_layer_depths)),
            "\n high_slowness_layer_depths_p.size()=",
            str(len(self.high_slowness_layer_depths_p)),
            "\n high_slowness_layer_depths_s.size()=",
            str(len(self.high_slowness_layer_depths_s)),
            "\n critical_depths.size()=",
            (str(len(self.critical_depths))
             if self.critical_depths is not None else 'N/A'),
            "\n"])
        desc += "**** Critical Depth Layers ************************\n"
        desc += str(self.critical_depths)
        desc += "\n"
        desc += "\n**** Fluid Layer Depths ************************\n"
        for fl in self.fluid_layer_depths:
            desc += str(fl.top_depth) + "," + str(fl.bot_depth) + " "
        desc += "\n"
        desc += "\n**** P High Slowness Layer Depths ****************\n"
        for fl in self.high_slowness_layer_depths_p:
            desc += str(fl.top_depth) + "," + str(fl.bot_depth) + " "
        desc += "\n"
        desc += "\n**** S High Slowness Layer Depths ****************\n"
        for fl in self.high_slowness_layer_depths_s:
            desc += str(fl.top_depth) + "," + str(fl.bot_depth) + " "
        desc += "\n"
        desc += "\n**** P Layers ****************\n"
        for l in self.p_layers:  # NOQA
            desc += str(l) + "\n"
        return desc

    def create_sample(self):
        """
        Create slowness-depth layers from a velocity model.

        This method takes a velocity model and creates a vector containing
        slowness-depth layers that, hopefully, adequately sample both slowness
        and depth so that the travel time as a function of distance can be
        reconstructed from the theta function.
        """
        # Some checks on the velocity model
        self.v_mod.validate()
        if len(self.v_mod) == 0:
            raise SlownessModelError("velModel.get_num_layers()==0")
        if self.v_mod.layers[0]['top_s_velocity'] == 0:
            raise SlownessModelError(
                "Unable to handle zero S velocity layers at surface. "
                "This should be fixed at some point, but is a limitation of "
                "TauP at this point.")
        if self.debug:
            print("start create_sample")

        self.radius_of_planet = self.v_mod.radius_of_planet

        if self.debug:
            print("find_critical_points")
        self.find_critical_points()
        if self.debug:
            print("coarse_sample")
        self.coarse_sample()
        if self.debug:
            self.validate()
        if self.debug:
            print("ray_paramCheck")
        self.ray_param_inc_check()
        if self.debug:
            print("depth_inc_check")
        self.depth_inc_check()
        if self.debug:
            print("distance_check")
        self.distance_check()
        if self.debug:
            print("fix_critical_points")
        self.fix_critical_points()

        self.validate()
        if self.debug:
            print("create_sample seems to be done successfully.")

    def find_critical_points(self):
        """
        Find all critical points within a velocity model.

        Critical points are first order discontinuities in velocity/slowness,
        local extrema in slowness. A high slowness zone is a low velocity zone,
        but it is possible to have a slightly low velocity zone within a
        spherical planet that is not a high slowness zone and thus does not
        exhibit any of the pathological behavior of a low velocity zone.
        """
        high_slowness_zone_p = DepthRange()
        high_slowness_zone_s = DepthRange()
        fluid_zone = DepthRange()
        in_fluid_zone = False
        below_outer_core = False
        in_high_slowness_zone_p = False
        in_high_slowness_zone_s = False
        # just some very big values (java had max possible of type,
        # but these should do)
        min_p_so_far = 1.1e300
        min_s_so_far = 1.1e300
        # First remove any critical points previously stored
        # so these are effectively re-initialised... it's probably silly
        self.critical_depths = np.zeros(len(self.v_mod.layers) + 1,
                                        dtype=CriticalDepth)
        cd_count = 0
        self.high_slowness_layer_depths_p = []  # lists of DepthRange
        self.high_slowness_layer_depths_s = []
        self.fluid_layer_depths = []

        # Initialize the current velocity layer
        # to be zero thickness layer with values at the surface
        curr_v_layer = self.v_mod.layers[0]
        curr_v_layer = np.array([(
            curr_v_layer['top_depth'], curr_v_layer['top_depth'],
            curr_v_layer['top_p_velocity'], curr_v_layer['top_p_velocity'],
            curr_v_layer['top_s_velocity'], curr_v_layer['top_s_velocity'],
            curr_v_layer['top_density'], curr_v_layer['top_density'],
            curr_v_layer['top_qp'], curr_v_layer['top_qp'],
            curr_v_layer['top_qs'], curr_v_layer['top_qs'])],
            dtype=VelocityLayer)

        curr_s_layer = create_from_vlayer(
            v_layer=curr_v_layer,
            is_p_wave=self.s_wave,
            radius_of_planet=self.v_mod.radius_of_planet,
            is_spherical=self.v_mod.is_spherical)
        curr_p_layer = create_from_vlayer(
            v_layer=curr_v_layer,
            is_p_wave=self.p_wave,
            radius_of_planet=self.v_mod.radius_of_planet,
            is_spherical=self.v_mod.is_spherical)

        # We know that the top is always a critical slowness so add 0
        self.critical_depths[cd_count] = (0, 0, 0, 0)
        cd_count += 1
        # Check to see if starting in fluid zone.
        if in_fluid_zone is False and curr_v_layer['top_s_velocity'] == 0:
            in_fluid_zone = True
            fluid_zone = DepthRange(top_depth=curr_v_layer['top_depth'])
            curr_s_layer = curr_p_layer
        if min_s_so_far > curr_s_layer['top_p']:
            min_s_so_far = curr_s_layer['top_p']
        # P is not a typo, it represents slowness, not P-wave speed.
        if min_p_so_far > curr_p_layer['top_p']:
            min_p_so_far = curr_p_layer['top_p']

        for layer_num, layer in enumerate(self.v_mod.layers):
            prev_v_layer = curr_v_layer
            prev_s_layer = curr_s_layer
            prev_p_layer = curr_p_layer
            # Could make the following a deep copy, but not necessary.
            # Also don't just replace layer here and in the loop
            # control with curr_v_layer, or the reference to the first
            # zero thickness layer that has been initialised above
            # will break.
            curr_v_layer = layer
            # Check again if in fluid zone
            if in_fluid_zone is False and curr_v_layer['top_s_velocity'] == 0:
                in_fluid_zone = True
                fluid_zone = DepthRange(top_depth=curr_v_layer['top_depth'])
            # If already in fluid zone, check if exited (java line 909)
            if in_fluid_zone is True and curr_v_layer['top_s_velocity'] != 0:
                if prev_v_layer['bot_depth'] > self.v_mod.iocb_depth:
                    below_outer_core = True
                in_fluid_zone = False
                fluid_zone.bot_depth = prev_v_layer['bot_depth']
                self.fluid_layer_depths.append(fluid_zone)

            curr_p_layer = create_from_vlayer(
                v_layer=curr_v_layer,
                is_p_wave=self.p_wave,
                radius_of_planet=self.v_mod.radius_of_planet,
                is_spherical=self.v_mod.is_spherical)

            # If we are in a fluid zone ( S velocity = 0.0 ) or if we are below
            # the outer core and allow_inner_core_s=false then use the P
            # velocity structure to look for critical points.
            if in_fluid_zone \
                    or (below_outer_core and self.allow_inner_core_s is False):
                curr_s_layer = curr_p_layer
            else:
                curr_s_layer = create_from_vlayer(
                    v_layer=curr_v_layer,
                    is_p_wave=self.s_wave,
                    radius_of_planet=self.v_mod.radius_of_planet,
                    is_spherical=self.v_mod.is_spherical)

            if prev_s_layer['bot_p'] != curr_s_layer['top_p'] \
                    or prev_p_layer['bot_p'] != curr_p_layer['top_p']:
                # a first order discontinuity
                self.critical_depths[cd_count] = (
                    curr_s_layer['top_depth'],
                    layer_num,
                    -1,
                    -1)
                cd_count += 1
                if self.debug:
                    print('First order discontinuity, depth =' +
                          str(curr_s_layer['top_depth']))
                    print('between' + str(prev_p_layer), str(curr_p_layer))
                if in_high_slowness_zone_s and \
                        curr_s_layer['top_p'] < min_s_so_far:
                    if self.debug:
                        print("Top of current layer is the bottom"
                              " of a high slowness zone.")
                    high_slowness_zone_s.bot_depth = curr_s_layer['top_depth']
                    self.high_slowness_layer_depths_s.append(
                        high_slowness_zone_s)
                    in_high_slowness_zone_s = False
                if in_high_slowness_zone_p and \
                        curr_p_layer['top_p'] < min_p_so_far:
                    if self.debug:
                        print("Top of current layer is the bottom"
                              " of a high slowness zone.")
                    high_slowness_zone_p.bot_depth = curr_s_layer['top_depth']
                    self.high_slowness_layer_depths_p.append(
                        high_slowness_zone_p)
                    in_high_slowness_zone_p = False
                # Update min_p_so_far and min_s_so_far as all total reflections
                # off of the top of the discontinuity are ok even though below
                # the discontinuity could be the start of a high slowness zone.
                if min_p_so_far > curr_p_layer['top_p']:
                    min_p_so_far = curr_p_layer['top_p']
                if min_s_so_far > curr_s_layer['top_p']:
                    min_s_so_far = curr_s_layer['top_p']

                if in_high_slowness_zone_s is False and (
                        prev_s_layer['bot_p'] < curr_s_layer['top_p'] or
                        curr_s_layer['top_p'] < curr_s_layer['bot_p']):
                    # start of a high slowness zone S
                    if self.debug:
                        print("Found S high slowness at first order " +
                              "discontinuity, layer = " + str(layer_num))
                    in_high_slowness_zone_s = True
                    high_slowness_zone_s = \
                        DepthRange(top_depth=curr_s_layer['top_depth'])
                    high_slowness_zone_s.ray_param = min_s_so_far
                if in_high_slowness_zone_p is False and (
                        prev_p_layer['bot_p'] < curr_p_layer['top_p'] or
                        curr_p_layer['top_p'] < curr_p_layer['bot_p']):
                    # start of a high slowness zone P
                    if self.debug:
                        print("Found P high slowness at first order " +
                              "discontinuity, layer = " + str(layer_num))
                    in_high_slowness_zone_p = True
                    high_slowness_zone_p = \
                        DepthRange(top_depth=curr_p_layer['top_depth'])
                    high_slowness_zone_p.ray_param = min_p_so_far

            elif ((prev_s_layer['top_p'] - prev_s_layer['bot_p']) *
                  (prev_s_layer['bot_p'] - curr_s_layer['bot_p']) < 0) or (
                      (prev_p_layer['top_p'] - prev_p_layer['bot_p']) *
                      (prev_p_layer['bot_p'] - curr_p_layer['bot_p'])) < 0:
                # local slowness extrema, java l 1005
                self.critical_depths[cd_count] = (
                    curr_s_layer['top_depth'],
                    layer_num,
                    -1,
                    -1)
                cd_count += 1
                if self.debug:
                    print("local slowness extrema, depth=" +
                          str(curr_s_layer['top_depth']))
                # here is line 1014 of the java src!
                if in_high_slowness_zone_p is False \
                        and curr_p_layer['top_p'] < curr_p_layer['bot_p']:
                    if self.debug:
                        print("start of a P high slowness zone, local "
                              "slowness extrema,min_p_so_far= " +
                              str(min_p_so_far))
                    in_high_slowness_zone_p = True
                    high_slowness_zone_p = \
                        DepthRange(top_depth=curr_p_layer['top_depth'])
                    high_slowness_zone_p.ray_param = min_p_so_far
                if in_high_slowness_zone_s is False \
                        and curr_s_layer['top_p'] < curr_s_layer['bot_p']:
                    if self.debug:
                        print("start of a S high slowness zone, local "
                              "slowness extrema, min_s_so_far= " +
                              str(min_s_so_far))
                    in_high_slowness_zone_s = True
                    high_slowness_zone_s = \
                        DepthRange(top_depth=curr_s_layer['top_depth'])
                    high_slowness_zone_s.ray_param = min_s_so_far

            if in_high_slowness_zone_p and \
                    curr_p_layer['bot_p'] < min_p_so_far:
                # P: layer contains the bottom of a high slowness zone. java
                #  l 1043
                if self.debug:
                    print("layer contains the bottom of a P " +
                          "high slowness zone. min_p_so_far=" +
                          str(min_p_so_far), curr_p_layer)
                high_slowness_zone_p.bot_depth = self.find_depth_from_layers(
                    min_p_so_far, layer_num, layer_num, self.p_wave)
                self.high_slowness_layer_depths_p.append(high_slowness_zone_p)
                in_high_slowness_zone_p = False

            if in_high_slowness_zone_s and \
                    curr_s_layer['bot_p'] < min_s_so_far:
                # S: layer contains the bottom of a high slowness zone. java
                #  l 1043
                if self.debug:
                    print("layer contains the bottom of a S " +
                          "high slowness zone. min_s_so_far=" +
                          str(min_s_so_far), curr_s_layer)
                # in fluid layers we want to check p_wave structure
                # when looking for S wave critical points
                por_s = (self.p_wave
                         if curr_s_layer == curr_p_layer else self.s_wave)
                high_slowness_zone_s.bot_depth = self.find_depth_from_layers(
                    min_s_so_far, layer_num, layer_num, por_s)
                self.high_slowness_layer_depths_s.append(high_slowness_zone_s)
                in_high_slowness_zone_s = False
            if min_p_so_far > curr_p_layer['bot_p']:
                min_p_so_far = curr_p_layer['bot_p']
            if min_p_so_far > curr_p_layer['top_p']:
                min_p_so_far = curr_p_layer['top_p']
            if min_s_so_far > curr_s_layer['bot_p']:
                min_s_so_far = curr_s_layer['bot_p']
            if min_s_so_far > curr_s_layer['top_p']:
                min_s_so_far = curr_s_layer['top_p']
            if self.debug and in_high_slowness_zone_s:
                print("In S high slowness zone, layer_num = " +
                      str(layer_num) + " min_s_so_far=" + str(min_s_so_far))
            if self.debug and in_high_slowness_zone_p:
                print("In P high slowness zone, layer_num = " +
                      str(layer_num) + " min_p_so_far=" + str(min_p_so_far))

        # We know that the bottommost depth is always a critical slowness,
        # so we add v_mod.get_num_layers()
        # java line 1094
        self.critical_depths[cd_count] = (
            self.radius_of_planet, len(self.v_mod), -1, -1)
        cd_count += 1

        # Check if the bottommost depth is contained within a high slowness
        # zone, might happen in a flat non-whole-planet model
        if in_high_slowness_zone_s:
            high_slowness_zone_s.bot_depth = curr_v_layer['bot_depth']
            self.high_slowness_layer_depths_s.append(high_slowness_zone_s)
        if in_high_slowness_zone_p:
            high_slowness_zone_p.bot_depth = curr_v_layer['bot_depth']
            self.high_slowness_layer_depths_p.append(high_slowness_zone_p)

        # Check if the bottommost depth is contained within a fluid zone, this
        # would be the case if we have a non whole planet model with the bottom
        # in the outer core or if allow_inner_core_s == false and we want to
        # use the P velocity structure in the inner core.
        if in_fluid_zone:
            fluid_zone.bot_depth = curr_v_layer['bot_depth']
            self.fluid_layer_depths.append(fluid_zone)

        self.critical_depths = self.critical_depths[:cd_count]

        self.validate()

    def get_num_layers(self, is_p_wave):
        """
        Number of slowness layers.

        This is meant to return the number of P or S layers.

        :param is_p_wave: Return P layer count (``True``) or S layer count
            (``False``).
        :type is_p_wave: bool
        :returns: Number of slowness layers.
        :rtype: int
        """
        if is_p_wave:
            return len(self.p_layers)
        else:
            return len(self.s_layers)

    def find_depth_from_depths(self, ray_param, top_depth, bot_depth,
                               is_p_wave):
        """
        Find depth corresponding to a slowness between two given depths.

        The given depths are converted to layer numbers before calling
        :meth:`find_depth_from_layers`.

        :param ray_param: Slowness (aka ray parameter) to find, in s/km.
        :type ray_param: float
        :param top_depth: Top depth to search, in km.
        :type top_depth: float
        :param bot_depth: Bottom depth to search, in km.
        :type bot_depth: float
        :param is_p_wave: ``True`` if P wave or ``False`` for S wave.
        :type is_p_wave: bool

        :returns: Depth (in km) corresponding to the desired slowness.
        :rtype: float

        :raises SlownessModelError:
            If ``top_critical_layer > bot_critical_layer``
            because there are no layers to search, or if there is an increase
            in slowness, i.e., a negative velocity gradient, that just balances
            the decrease in slowness due to the spherical planet, or if the ray
            parameter ``p`` is not contained within the specified layer range.
        """
        top_layer_num = self.v_mod.layer_number_below(top_depth)[0]
        if self.v_mod.layers[top_layer_num]['bot_depth'] == top_depth:
            top_layer_num += 1
        bot_layer_num = self.v_mod.layer_number_above(bot_depth)[0]
        return self.find_depth_from_layers(ray_param, top_layer_num,
                                           bot_layer_num, is_p_wave)

    def find_depth_from_layers(self, p, top_critical_layer, bot_critical_layer,
                               is_p_wave):
        """
        Find depth corresponding to a slowness p between two velocity layers.

        Here, slowness is defined as ``(radius_of_planet-depth) / velocity``,
        and sometimes called ray parameter. Both the top and the bottom
        velocity layers are included. We also check to see if the slowness is
        less than the bottom slowness of these layers but greater than the top
        slowness of the next deeper layer. This corresponds to a total
        reflection. In this case a check needs to be made to see if this is an
        S wave reflecting off of a fluid layer, use P velocity below in this
        case. We assume that slowness is monotonic within these layers and
        therefore there is only one depth with the given slowness. This means
        we return the first depth that we find.

        :param p: Slowness (aka ray parameter) to find, in s/km.
        :type p: float
        :param top_critical_layer: Top layer number to search.
        :type top_critical_layer: int
        :param bot_critical_layer: Bottom layer number to search.
        :type bot_critical_layer: int
        :param is_p_wave: ``True`` if P wave or ``False`` for S wave.
        :type is_p_wave: bool

        :returns: Depth (in km) corresponding to the desired slowness.
        :rtype: float

        :raises SlownessModelError: If
            ``top_critical_layer > bot_critical_layer``
            because there are no layers to search, or if there is an increase
            in slowness, i.e., a negative velocity gradient, that just balances
            the decrease in slowness due to the spherical planet, or if the ray
            parameter ``p`` is not contained within the specified layer range.
        """
        # top_p = 1.1e300  # dummy numbers
        # bot_p = 1.1e300
        wave_type = 'P' if is_p_wave else 'S'

        if top_critical_layer > bot_critical_layer:
            raise SlownessModelError(
                "findDepth: no layers to search (wrong layer num?)")
        for layer_num in range(top_critical_layer, bot_critical_layer + 1):
            vel_layer = self.v_mod.layers[layer_num]
            top_velocity = evaluate_velocity_at_top(vel_layer, wave_type)
            bot_velocity = evaluate_velocity_at_bottom(vel_layer, wave_type)
            top_p = self.to_slowness(top_velocity, vel_layer['top_depth'])
            bot_p = self.to_slowness(bot_velocity, vel_layer['bot_depth'])
            # Check to see if we are within 'chatter level' (numerical
            # error) of the top or bottom and if so then return that depth.
            if abs(top_p - p) < self.slowness_tolerance:
                return vel_layer['top_depth']
            if abs(p - bot_p) < self.slowness_tolerance:
                return vel_layer['bot_depth']

            if (top_p - p) * (p - bot_p) >= 0:
                # Found layer containing p.
                # We interpolate assuming that velocity is linear within
                # this interval. So slope is the slope for velocity versus
                # depth.
                slope = (bot_velocity - top_velocity) / (
                    vel_layer['bot_depth'] - vel_layer['top_depth'])
                depth = self.interpolate(p, top_velocity,
                                         vel_layer['top_depth'], slope)
                return depth
            elif layer_num == top_critical_layer \
                    and abs(p - top_p) < self.slowness_tolerance:
                # Check to see if p is just outside the topmost layer. If so
                # then return the top depth.
                return vel_layer['top_depth']

            # Is p a total reflection? bot_p is the slowness at the bottom
            # of the last velocity layer from the previous loop, set top_p
            # to be the slowness at the top of the next layer.
            if layer_num < len(self.v_mod) - 1:
                vel_layer = self.v_mod.layers[layer_num + 1]
                top_velocity = evaluate_velocity_at_top(vel_layer, wave_type)
                if (is_p_wave is False and
                        np.any(self.depth_in_fluid(vel_layer['top_depth']))):
                    # Special case for S waves above a fluid. If top next
                    # layer is in a fluid then we should set top_velocity to
                    # be the P velocity at the top of the layer.
                    top_velocity = evaluate_velocity_at_top(vel_layer, 'P')

                top_p = self.to_slowness(top_velocity, vel_layer['top_depth'])
                if bot_p >= p >= top_p:
                    return vel_layer['top_depth']

        # noinspection PyUnboundLocalVariable
        if abs(p - bot_p) < self.slowness_tolerance:
            # java line 1275
            # Check to see if p is just outside the bottommost layer. If so
            # than return the bottom depth.
            print(" p is just outside the bottommost layer. This probably "
                  "shouldn't be allowed to happen!")
            # noinspection PyUnboundLocalVariable
            return vel_layer.getBotDepth()
        raise SlownessModelError(
            "slowness p=" + str(p) +
            " is not contained within the specified layers." +
            " top_critical_layer=" + str(top_critical_layer) +
            " bot_critical_layer=" + str(bot_critical_layer))

    def to_slowness(self, velocity, depth):
        """
        Convert velocity at some depth to slowness.

        :param velocity: The velocity to convert, in km/s.
        :type velocity: float
        :param depth: The depth (in km) at which to perform the calculation.
            Must be less than the radius of the planet defined in this
            model, or the result is undefined.
        :type depth: float

        :returns: The slowness, in s/km.
        :rtype: float
        """
        if np.any(velocity == 0):
            raise SlownessModelError(
                "to_slowness: velocity can't be zero, at depth" +
                str(depth),
                "Maybe related to using S velocities in outer core?")
        return (self.radius_of_planet - depth) / velocity

    def interpolate(self, p, top_velocity, top_depth, slope):
        """
        Interpolate slowness to depth within a layer.

        We interpolate assuming that velocity is linear within
        this interval.

        All parameters must be of the same shape.

        :param p: The slowness to interpolate, in s/km.
        :type p: :class:`float` or :class:`~numpy.ndarray`
        :param top_velocity: The velocity (in km/s) at the top of the layer.
        :type top_velocity: :class:`float` or :class:`~numpy.ndarray`
        :param top_depth: The depth (in km) for the top of the layer.
        :type top_depth: :class:`float` or :class:`~numpy.ndarray`
        :param slope: The slope (in (km/s)/km)  for velocity versus depth.
        :type slope: :class:`float` or :class:`~numpy.ndarray`

        :returns: The depth (in km) of the slowness below the layer boundary.
        :rtype: :class:`float` or :class:`~numpy.ndarray`
        """
        denominator = p * slope + 1
        if np.any(denominator == 0):
            raise SlownessModelError(
                "Negative velocity gradient that just balances the slowness "
                "gradient of the spherical slowness, i.e. planet flattening. "
                "Instructions unclear; explode.")
        else:
            depth = (self.radius_of_planet +
                     p * (top_depth * slope - top_velocity)) / denominator
            return depth

    def depth_in_fluid(self, depth):
        """
        Determine if the given depth is contained within a fluid zone.

        The fluid zone includes its upper boundary but not its lower boundary.
        The top and bottom of the fluid zone are not returned as a DepthRange,
        just like in the Java code, despite its claims to the contrary.

        :param depth: The depth to check, in km.
        :type depth: :class:`~numpy.ndarray`, dtype = :class:`float`

        :returns: ``True`` if the depth is within a fluid zone, ``False``
            otherwise.
        :rtype: :class:`~numpy.ndarray` (dtype = :class:`bool`)
        """
        ret = np.zeros(shape=depth.shape, dtype=np.bool_)
        for elem in self.fluid_layer_depths:
            ret |= (elem.top_depth <= depth) & (depth < elem.bot_depth)
        return ret

    def coarse_sample(self):
        """
        Create a coarse slowness sampling of the velocity model (v_mod).

        The resultant slowness layers will satisfy the maximum depth increments
        as well as sampling each point specified within the VelocityModel. The
        P and S sampling will also be compatible.
        """
        self.p_layers = create_from_vlayer(
            v_layer=self.v_mod.layers,
            is_p_wave=self.p_wave,
            radius_of_planet=self.v_mod.radius_of_planet,
            is_spherical=self.v_mod.is_spherical)

        with np.errstate(divide='ignore'):
            self.s_layers = create_from_vlayer(
                v_layer=self.v_mod.layers,
                is_p_wave=self.s_wave,
                radius_of_planet=self.v_mod.radius_of_planet,
                is_spherical=self.v_mod.is_spherical)

        mask = self.depth_in_fluid(self.v_mod.layers['top_depth'])
        if not self.allow_inner_core_s:
            mask |= self.v_mod.layers['top_depth'] >= self.v_mod.iocb_depth
        self.s_layers[mask] = self.p_layers[mask]

        # Check for first order discontinuity. However, we only consider
        # S discontinuities in the inner core if allow_inner_core_s is true.
        above = self.v_mod.layers[:-1]
        below = self.v_mod.layers[1:]
        mask = np.logical_or(
            above['bot_p_velocity'] != below['top_p_velocity'],
            np.logical_and(
                above['bot_s_velocity'] != below['top_s_velocity'],
                np.logical_or(
                    self.allow_inner_core_s,
                    below['top_depth'] < self.v_mod.iocb_depth)))

        index = np.where(mask)[0] + 1
        above = above[mask]
        below = below[mask]

        # If we are going from a fluid to a solid or solid to fluid, e.g., core
        # mantle or outer core to inner core then we need to use the P velocity
        # for determining the S discontinuity.
        top_s_vel = np.where(above['bot_s_velocity'] == 0,
                             above['bot_p_velocity'],
                             above['bot_s_velocity'])
        bot_s_vel = np.where(below['top_s_velocity'] == 0,
                             below['top_p_velocity'],
                             below['top_s_velocity'])

        # Add the layer, with zero thickness but nonzero slowness step,
        # corresponding to the discontinuity.
        curr_v_layer = np.empty(shape=above.shape, dtype=VelocityLayer)
        curr_v_layer['top_depth'] = above['bot_depth']
        curr_v_layer['bot_depth'] = above['bot_depth']
        curr_v_layer['top_p_velocity'] = above['bot_p_velocity']
        curr_v_layer['bot_p_velocity'] = below['top_p_velocity']
        curr_v_layer['top_s_velocity'] = top_s_vel
        curr_v_layer['bot_s_velocity'] = bot_s_vel
        curr_v_layer['top_density'].fill(_DEFAULT_VALUES["density"])
        curr_v_layer['bot_density'].fill(_DEFAULT_VALUES["density"])
        curr_v_layer['top_qp'].fill(_DEFAULT_VALUES["qp"])
        curr_v_layer['bot_qp'].fill(_DEFAULT_VALUES["qp"])
        curr_v_layer['top_qs'].fill(_DEFAULT_VALUES["qs"])
        curr_v_layer['bot_qs'].fill(_DEFAULT_VALUES["qs"])

        curr_p_layer = create_from_vlayer(
            v_layer=curr_v_layer,
            is_p_wave=self.p_wave,
            radius_of_planet=self.v_mod.radius_of_planet,
            is_spherical=self.v_mod.is_spherical)

        self.p_layers = np.insert(self.p_layers, index, curr_p_layer)

        curr_s_layer = create_from_vlayer(
            v_layer=curr_v_layer,
            is_p_wave=self.s_wave,
            radius_of_planet=self.v_mod.radius_of_planet,
            is_spherical=self.v_mod.is_spherical)

        mask2 = (above['bot_s_velocity'] == 0) & (below['top_s_velocity'] == 0)
        if not self.allow_inner_core_s:
            mask2 |= curr_v_layer['top_depth'] >= self.v_mod.iocb_depth
        curr_s_layer = np.where(mask2, curr_p_layer, curr_s_layer)
        self.s_layers = np.insert(self.s_layers, index, curr_s_layer)

        # Make sure that all high slowness layers are sampled exactly
        # at their bottom
        for high_zone in self.high_slowness_layer_depths_s:
            s_layer_num = self.layer_number_above(high_zone.bot_depth,
                                                  self.s_wave)
            high_s_layer = self.s_layers[s_layer_num]
            while high_s_layer['top_depth'] == high_s_layer['bot_depth'] and (
                    (high_s_layer['top_p'] - high_zone.ray_param) *
                    (high_zone.ray_param - high_s_layer['bot_p']) < 0):
                s_layer_num += 1
                high_s_layer = self.s_layers[s_layer_num]
            if high_zone.ray_param != high_s_layer['bot_p']:
                self.add_slowness(high_zone.ray_param, self.s_wave)
        for high_zone in self.high_slowness_layer_depths_p:
            s_layer_num = self.layer_number_above(high_zone.bot_depth,
                                                  self.p_wave)
            high_s_layer = self.p_layers[s_layer_num]
            while high_s_layer['top_depth'] == high_s_layer['bot_depth'] and (
                    (high_s_layer['top_p'] - high_zone.ray_param) *
                    (high_zone.ray_param - high_s_layer['bot_p']) < 0):
                s_layer_num += 1
                high_s_layer = self.p_layers[s_layer_num]
            if high_zone.ray_param != high_s_layer['bot_p']:
                self.add_slowness(high_zone.ray_param, self.p_wave)

        # Make sure P and S are consistent by adding discontinuities in one to
        # the other.
        # Numpy 1.6 compatibility
        # _tb = self.p_layers[['top_p', 'bot_p']]
        _tb = np.vstack([self.p_layers['top_p'],
                         self.p_layers['bot_p']]).T.ravel()
        uniq = np.unique(_tb)
        for p in uniq:
            self.add_slowness(p, self.s_wave)

        # Numpy 1.6 compatibility
        # _tb = self.p_layers[['top_p', 'bot_p']]
        _tb = np.vstack([self.s_layers['top_p'],
                         self.s_layers['bot_p']]).T.ravel()
        uniq = np.unique(_tb)
        for p in uniq:
            self.add_slowness(p, self.p_wave)

    def layer_number_above(self, depth, is_p_wave):
        """
        Find the index of the slowness layer that contains the given depth.

        Note that if the depth is a layer boundary, it returns the shallower
        of the two or possibly more (since total reflections are zero
        thickness layers) layers.

        .. seealso:: :meth:`layer_number_below`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`
        :param is_p_wave: Whether to look at P (``True``) velocity or S
            (``False``) velocity.
        :type is_p_wave: bool

        :returns: The slowness layer containing the requested depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape = ``depth.shape``)

        :raises SlownessModelError: If no layer in the slowness model contains
            the given depth.
        """
        if is_p_wave:
            layers = self.p_layers
        else:
            layers = self.s_layers

        # Check to make sure depth is within the range available
        if np.any(depth < layers[0]['top_depth']) or \
                np.any(depth > layers[-1]['bot_depth']):
            raise SlownessModelError("No layer contains this depth")

        found_layer_num = np.searchsorted(layers['top_depth'], depth)

        mask = found_layer_num != 0
        if np.isscalar(found_layer_num):
            if mask:
                found_layer_num -= 1
        else:
            found_layer_num[mask] -= 1

        return found_layer_num

    def layer_number_below(self, depth, is_p_wave):
        """
        Find the index of the slowness layer that contains the given depth.

        Note that if the depth is a layer boundary, it returns the deeper of
        the two or possibly more (since total reflections are zero thickness
        layers) layers.

        .. seealso:: :meth:`layer_number_above`

        :param depth: The depth to find, in km.
        :type depth: :class:`float` or :class:`~numpy.ndarray`
        :param is_p_wave: Whether to look at P (``True``) velocity or S
            (``False``) velocity.
        :type is_p_wave: bool

        :returns: The slowness layer containing the requested depth.
        :rtype: :class:`int` or :class:`~numpy.ndarray` (dtype = :class:`int`,
            shape = ``depth.shape``)

        :raises SlownessModelError: If no layer in the slowness model contains
            the given depth.
        """
        if is_p_wave:
            layers = self.p_layers
        else:
            layers = self.s_layers

        # Check to make sure depth is within the range available
        if np.any(depth < layers[0]['top_depth']) or \
                np.any(depth > layers[-1]['bot_depth']):
            raise SlownessModelError("No layer contains this depth")

        found_layer_num = np.searchsorted(layers['bot_depth'], depth,
                                          side='right')

        mask = found_layer_num == layers.shape[0]
        if np.isscalar(found_layer_num):
            if mask:
                found_layer_num -= 1
        else:
            found_layer_num[mask] -= 1

        return found_layer_num

    def get_slowness_layer(self, layer, is_p_wave):
        """
        Return the Slowness_layer of the requested wave type.

        This is not meant to be a clone!

        :param layer: The number of the layer(s) to return.
        :type layer: :class:`int` or :class:`~numpy.ndarray` (dtype =
            :class:`int`)
        :param is_p_wave: Whether to return the P layer (``True``) or the S
            layer (``False``).
        :type is_p_wave: bool

        :returns: The slowness layer(s).
        :rtype: :class:`~numpy.ndarray`
            (dtype = :class:`obspy.taup.helper_classes.SlownessLayer`,
            shape = ``layer_num.shape``)
        """
        if is_p_wave:
            return self.p_layers[layer]
        else:
            return self.s_layers[layer]

    def add_slowness(self, p, is_p_wave):
        """
        Add a ray parameter to the slowness sampling for the given wave type.

        Slowness layers are split as needed and P and S sampling are kept
        consistent within fluid layers. Note, this makes use of the velocity
        model, so all interpolation is linear in velocity, not in slowness!

        :param p: The slowness value to add, in s/km.
        :type p: float
        :param is_p_wave: Whether to add to the P wave (``True``) or the S wave
            (``False``) sampling.
        :type is_p_wave: bool
        """
        if is_p_wave:
            # NB Unlike Java (unfortunately) these are not modified in place!
            # NumPy arrays cannot have values inserted in place.
            layers = self.p_layers
            other_layers = self.s_layers
            wave = 'P'
        else:
            layers = self.s_layers
            other_layers = self.p_layers
            wave = 'S'

        # If depths are the same only need top_velocity, and just to verify we
        # are not in a fluid.
        nonzero = layers['top_depth'] != layers['bot_depth']
        above = self.v_mod.evaluate_above(layers['bot_depth'], wave)
        below = self.v_mod.evaluate_below(layers['top_depth'], wave)
        top_velocity = np.where(nonzero, below, above)
        bot_velocity = np.where(nonzero, above, below)

        mask = ((layers['top_p'] - p) * (p - layers['bot_p'])) > 0
        # Don't need to check for S waves in a fluid or in inner core if
        # allow_inner_core_s is False.
        if not is_p_wave:
            mask &= top_velocity != 0
            if not self.allow_inner_core_s:
                iocb_mask = layers['bot_depth'] > self.v_mod.iocb_depth
                mask &= ~iocb_mask

        index = np.where(mask)[0]

        bot_depth = np.copy(layers['bot_depth'])
        # Not a zero thickness layer, so calculate the depth for
        # the ray parameter.
        slope = ((bot_velocity[nonzero] - top_velocity[nonzero]) /
                 (layers['bot_depth'][nonzero] - layers['top_depth'][nonzero]))
        bot_depth[nonzero] = self.interpolate(p, top_velocity[nonzero],
                                              layers['top_depth'][nonzero],
                                              slope)

        bot_layer = np.empty(shape=index.shape, dtype=SlownessLayer)
        bot_layer['top_p'].fill(p)
        bot_layer['top_depth'] = bot_depth[mask]
        bot_layer['bot_p'] = layers['bot_p'][mask]
        bot_layer['bot_depth'] = layers['bot_depth'][mask]

        top_layer = np.empty(shape=index.shape, dtype=SlownessLayer)
        top_layer['top_p'] = layers['top_p'][mask]
        top_layer['top_depth'] = layers['top_depth'][mask]
        top_layer['bot_p'].fill(p)
        top_layer['bot_depth'] = bot_depth[mask]

        # numpy 1.6 compatibility
        other_index = np.where(other_layers.reshape(1, -1) ==
                               layers[mask].reshape(-1, 1))
        layers[index] = bot_layer
        layers = np.insert(layers, index, top_layer)
        if len(other_index[0]):
            other_layers[other_index[1]] = bot_layer[other_index[0]]
            other_layers = np.insert(other_layers, other_index[1],
                                     top_layer[other_index[0]])

        if is_p_wave:
            self.p_layers = layers
            self.s_layers = other_layers
        else:
            self.s_layers = layers
            self.p_layers = other_layers

    def ray_param_inc_check(self):
        """
        Check that no slowness layer's ray parameter interval is too large.

        The limit is determined by ``self.max_delta_p``.
        """
        for wave in [self.s_wave, self.p_wave]:
            # These might change with calls to add_slowness, so be sure we have
            # the correct copy.
            if wave == self.p_wave:
                layers = self.p_layers
            else:
                layers = self.s_layers

            diff = layers['top_p'] - layers['bot_p']
            absdiff = np.abs(diff)

            mask = absdiff > self.max_delta_p
            diff = diff[mask]
            absdiff = absdiff[mask]
            top_p = layers['top_p'][mask]

            new_count = np.ceil(absdiff / self.max_delta_p).astype(np.int_)
            steps = diff / new_count

            for start, n, delta in zip(top_p, new_count, steps):
                for j in range(1, n):
                    newp = start + j * delta
                    self.add_slowness(newp, self.p_wave)
                    self.add_slowness(newp, self.s_wave)

    def depth_inc_check(self):
        """
        Check that no slowness layer is too thick.

        The maximum is determined by ``self.max_depth_interval``.
        """
        for wave in [self.s_wave, self.p_wave]:
            # These might change with calls to add_slowness, so be sure we
            # have the correct copy.
            if wave == self.p_wave:
                layers = self.p_layers
            else:
                layers = self.s_layers

            diff = layers['bot_depth'] - layers['top_depth']

            mask = diff > self.max_depth_interval
            diff = diff[mask]
            top_depth = layers['top_depth'][mask]

            new_count = np.ceil(diff / self.max_depth_interval).astype(np.int_)
            steps = diff / new_count

            for start, nd, delta in zip(top_depth, new_count, steps):
                new_depth = start + np.arange(1, nd) * delta
                if wave == self.s_wave:
                    velocity = self.v_mod.evaluate_above(new_depth, 'S')

                    smask = velocity == 0
                    if not self.allow_inner_core_s:
                        smask |= new_depth >= self.v_mod.iocb_depth
                    if np.any(smask):
                        velocity[smask] = self.v_mod.evaluate_above(
                            new_depth[smask], 'P')
                    slowness = self.to_slowness(velocity, new_depth)
                else:
                    slowness = self.to_slowness(
                        self.v_mod.evaluate_above(new_depth, 'P'),
                        new_depth)

                for p in slowness:
                    self.add_slowness(p, self.p_wave)
                    self.add_slowness(p, self.s_wave)

    def distance_check(self):
        """
        Check that no slowness layer is too wide or undersampled.

        The width must be less than ``self.max_range_interval`` and the
        (estimated) error due to linear interpolation must be less than
        ``self.max_interp_error``.
        """
        for curr_wave_type in [self.s_wave, self.p_wave]:
            is_curr_ok = False
            is_prev_ok = False
            prev_prev_id = None
            prev_td = None
            curr_td = None
            j = 0
            s_layer = self.get_slowness_layer(j, curr_wave_type)
            while j < self.get_num_layers(curr_wave_type):
                prev_s_layer = s_layer
                s_layer = self.get_slowness_layer(j, curr_wave_type)
                if (self.depth_in_high_slowness(s_layer['bot_depth'],
                                                s_layer['bot_p'],
                                                curr_wave_type) is False and
                    self.depth_in_high_slowness(s_layer['top_depth'],
                                                s_layer['top_p'],
                                                curr_wave_type) is False):
                    # Don't calculate prevTD if we can avoid it
                    if is_curr_ok and curr_td is not None:
                        if is_prev_ok:
                            prev_prev_id = prev_td
                        else:
                            prev_prev_id = None
                        prev_td = curr_td
                        is_prev_ok = True
                    else:
                        prev_td = self.approx_distance(j - 1, s_layer['top_p'],
                                                       curr_wave_type)
                        is_prev_ok = True
                    curr_td = self.approx_distance(j, s_layer['bot_p'],
                                                   curr_wave_type)
                    is_curr_ok = True
                    # Check for jump of too great distance
                    if (abs(prev_td['dist'] - curr_td['dist']) >
                            self.max_range_interval and
                            abs(s_layer['top_p'] - s_layer['bot_p']) >
                            2 * self.min_delta_p):
                        if self.debug:
                            print("At " + str(j) + " Distance jump too great ("
                                  ">max_range_interval " +
                                  str(self.max_range_interval) + "), adding "
                                  "slowness. ")
                        p = (s_layer['top_p'] + s_layer['bot_p']) / 2
                        self.add_slowness(p, self.p_wave)
                        self.add_slowness(p, self.s_wave)
                        curr_td = prev_td
                        is_curr_ok = is_prev_ok
                        prev_td = prev_prev_id
                        prev_prev_id = None
                        is_prev_ok = prev_td is not None
                    else:
                        # Make guess as to error estimate due to linear
                        # interpolation if it is not ok, then we split both
                        # the previous and current slowness layers, this has
                        # the nice, if unintended, consequence of adding
                        # extra samples in the neighborhood of poorly
                        # sampled caustics.
                        split_ray_param = \
                            (s_layer['top_p'] + s_layer['bot_p']) / 2
                        all_but_layer = self.approx_distance(
                            j - 1, split_ray_param, curr_wave_type)
                        split_layer = np.array([(
                            s_layer['top_p'], s_layer['top_depth'],
                            split_ray_param,
                            bullen_depth_for(s_layer, split_ray_param,
                                             self.radius_of_planet))],
                            dtype=SlownessLayer)
                        just_layer_time, just_layer_dist = \
                            bullen_radial_slowness(
                                split_layer, split_ray_param,
                                self.radius_of_planet)
                        split_time = \
                            all_but_layer['time'] + 2 * just_layer_time
                        split_dist = \
                            all_but_layer['dist'] + 2 * just_layer_dist
                        # Python standard division is not IEEE compliant,
                        # as “The IEEE 754 standard specifies that every
                        # floating point arithmetic operation, including
                        # division by zero, has a well-defined result”.
                        # Use numpy's division instead by using np.array:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            diff = (curr_td['time'] -
                                    ((split_time - prev_td['time']) *
                                     ((curr_td['dist'] - prev_td['dist']) /
                                      (split_dist - prev_td['dist'])) +
                                     prev_td['time']))
                        if abs(diff) > self.max_interp_error:
                            p1 = (prev_s_layer['top_p'] +
                                  prev_s_layer['bot_p']) / 2
                            p2 = (s_layer['top_p'] + s_layer['bot_p']) / 2
                            self.add_slowness(p1, self.p_wave)
                            self.add_slowness(p1, self.s_wave)
                            self.add_slowness(p2, self.p_wave)
                            self.add_slowness(p2, self.s_wave)
                            curr_td = prev_prev_id
                            is_curr_ok = curr_td is not None
                            is_prev_ok = False
                            if j > 0:
                                # Back up one step unless we are at the
                                # beginning, then stay put.
                                j -= 1
                                s_layer = self.get_slowness_layer(
                                    j - 1 if j - 1 >= 0 else 0, curr_wave_type)
                                # This s_layer will become prevSLayer in the
                                # next loop.
                            else:
                                is_prev_ok = False
                                is_curr_ok = False
                        else:
                            j += 1
                            if self.debug and j % 10 == 0:
                                print(j)
                else:
                    prev_prev_id = None
                    prev_td = None
                    curr_td = None
                    is_curr_ok = False
                    is_prev_ok = False
                    j += 1
                    if self.debug and j % 100 == 0:
                        print(j)
            if self.debug:
                print("Number of " + ("P" if curr_wave_type else "S") +
                      " slowness layers: " + str(j))

    def depth_in_high_slowness(self, depth, ray_param, is_p_wave,
                               return_depth_range=False):
        """
        Determine if depth and slowness are within a high slowness zone.

        Whether the high slowness zone includes its upper boundary and its
        lower boundaries depends upon the ray parameter. The slowness at the
        depth is needed because if depth happens to correspond to a
        discontinuity that marks the bottom of the high slowness zone but the
        ray is actually a total reflection then it is not part of the high
        slowness zone. The ray parameter that delimits the zone, i.e., it can
        turn at the top and the bottom, is in the zone at the top, but out of
        the zone at the bottom. (?)

        :param depth: The depth to check, in km.
        :type depth: float
        :param ray_param: The slowness to check, in s/km.
        :type ray_param: float
        :param is_p_wave: Whether to check the P wave (``True``) or the S wave
            (``False``).
        :type is_p_wave: bool
        :param return_depth_range: Whether to also return the DepthRange of
            the high slowness zone.

        :returns: ``True`` if within a high slowness zone, ``False`` otherwise.
            If return_depth_range is ``True``, also returns a DepthRange object
        :rtype: bool, or (bool, DepthRange)
        """
        if is_p_wave:
            high_slowness_layer_depths = self.high_slowness_layer_depths_p
        else:
            high_slowness_layer_depths = self.high_slowness_layer_depths_s
        for temp_range in high_slowness_layer_depths:
            if temp_range.top_depth <= depth <= temp_range.bot_depth:
                if ray_param > temp_range.ray_param \
                        or (ray_param == temp_range.ray_param and
                            depth == temp_range.top_depth):
                    if return_depth_range:
                        return True, temp_range
                    return True
        if return_depth_range:
            return False, None
        return False

    def approx_distance(self, slowness_turn_layer, p, is_p_wave):
        """
        Approximate distance for ray turning at the bottom of a layer.

        Generates approximate distance, in radians, for a ray from a surface
        source that turns at the bottom of the given slowness layer.

        :param slowness_turn_layer: The number of the layer at which the ray
            should turn.
        :type slowness_turn_layer: int
        :param p: The slowness to calculate, in s/km.
        :type p: float
        :param is_p_wave: Whether to use the P (``True``) or S (``False``)
            wave.
        :type is_p_wave: bool

        :returns: The time (in s) and distance (in rad) the ray travels.
        :rtype: :class:`~numpy.ndarray`
            (dtype = :class:`obspy.taup.helper_classes.TimeDist`, shape =
            (``slowness_turn_layer``, ))
        """
        # First, if the slowness model contains less than slowness_turn_layer
        # elements we can't calculate a distance.
        if slowness_turn_layer >= self.get_num_layers(is_p_wave):
            raise SlownessModelError(
                "Can't calculate a distance when get_num_layers() is smaller "
                "than the given slowness_turn_layer!")
        if p < 0:
            raise SlownessModelError("Ray parameter must not be negative!")
        td = np.zeros(1, dtype=TimeDist)
        td['p'] = p
        layer_num = np.arange(0, slowness_turn_layer + 1)
        if len(layer_num):
            time, dist = self.layer_time_dist(p, layer_num, is_p_wave)
            # Return 2* distance and time because there is a downgoing as well
            # as an upgoing leg, which are equal since this is for a surface
            # source.
            td['time'] = 2 * np.sum(time)
            td['dist'] = 2 * np.sum(dist)
        return td

    def layer_time_dist(self, spherical_ray_param, layer_num, is_p_wave,
                        check=True, allow_turn=False):
        """
        Calculate time and distance for a ray passing through a layer.

        Calculates the time and distance increments accumulated by a ray of
        spherical ray parameter ``p`` when passing through layer ``layer_num``.
        Note that this gives half of the true range and time increments since
        there will be both an upgoing and a downgoing path. It also only does
        the calculation for the simple cases of the centre of the planet, where
        the ray parameter is zero, or for constant velocity layers. Otherwise,
        it calls :func:`~.bullen_radial_slowness`.

        Either ``spherical_ray_param`` or ``layer_num`` must be 0-D, or they
        must have the same shape.

        :param spherical_ray_param: The spherical ray parameter of the
            ray(s), in s/km.
        :type spherical_ray_param: :class:`float` or :class:`~numpy.ndarray`
        :param layer_num: The layer(s) in which the calculation should be done.
        :type layer_num: :class:`float` or :class:`~numpy.ndarray`
        :param is_p_wave: Whether to look at the P (``True``) or S (``False``)
            wave.
        :type is_p_wave: bool
        :param check: Whether to perform checks of input consistency.
        :type check: bool
        :param allow_turn: Whether to allow the ray to turn in the middle of a
            layer.
        :type allow_turn: bool

        :returns: The time (in s) and distance (in rad) increments for the
            specified ray(s) and layer(s).
        :rtype: :class:`~numpy.ndarray`
            (dtype = :class:`obspy.taup.helper_classes.TimeDist`, shape =
            ``spherical_ray_param.shape`` or ``layer_num.shape``)

        :raises SlownessModelError: If the ray with the given spherical ray
            parameter cannot propagate within this layer, or if the ray turns
            within this layer but not at the bottom. These checks may be
            bypassed by specifying ``check=False``.
        """
        spherical_layer = self.get_slowness_layer(layer_num, is_p_wave)
        pdim = np.ndim(spherical_ray_param)
        ldim = np.ndim(layer_num)

        if ldim == 1 and pdim == 0:
            time = np.empty(shape=layer_num.shape, dtype=np.float_)
            dist = np.empty(shape=layer_num.shape, dtype=np.float_)
        elif ldim == 0 and pdim == 1:
            time = np.empty(shape=spherical_ray_param.shape, dtype=np.float_)
            dist = np.empty(shape=spherical_ray_param.shape, dtype=np.float_)
        elif ldim == pdim and (ldim == 0 or
                               layer_num.shape == spherical_ray_param.shape):
            time = np.empty(shape=layer_num.shape, dtype=np.float_)
            dist = np.empty(shape=layer_num.shape, dtype=np.float_)
        else:
            raise TypeError('Either spherical_ray_param or layer_num must be '
                            '0D, or they must have the same shape.')

        # First make sure that a ray with this ray param can propagate
        # within this layer and doesn't turn in the middle of the layer. If
        # not, raise error.
        if check:
            if not allow_turn:
                minp = np.minimum(spherical_layer['top_p'],
                                  spherical_layer['bot_p'])
                if np.any(spherical_ray_param > minp):
                    raise SlownessModelError(
                        'Ray turns in the middle of this layer! '
                        'layer_num = %d' % (layer_num, ))

            if np.any(spherical_ray_param > spherical_layer['top_p']):
                raise SlownessModelError('Ray cannot propagate within this '
                                         ' layer, given ray param too large.')
        if np.any(spherical_ray_param < 0):
            raise SlownessModelError("Ray parameter must not be negative!")

        turning_layers = spherical_ray_param > spherical_layer['bot_p']
        if np.any(turning_layers):
            if ldim == 1 and pdim == 0:
                # Turn in a layer, create temp layers with p at bottom.
                tmp_layers = spherical_layer[turning_layers]
                turn_depth = bullen_depth_for(
                    tmp_layers, spherical_ray_param, self.radius_of_planet,
                    check=False)
                spherical_layer['bot_p'][turning_layers] = spherical_ray_param
                spherical_layer['bot_depth'][turning_layers] = turn_depth

            elif ldim == 0 and pdim == 1:
                # Turn in layer, create temp layers with each p at bottom.
                # Expand layer array so that each one turns at correct depth.
                ldim = 1
                new_layers = np.repeat(spherical_layer,
                                       len(spherical_ray_param))
                turn_depth = bullen_depth_for(
                    new_layers, spherical_ray_param, self.radius_of_planet,
                    check=False)
                new_layers['bot_p'][turning_layers] = \
                    spherical_ray_param[turning_layers]
                new_layers['bot_depth'][turning_layers] = \
                    turn_depth[turning_layers]
                spherical_layer = new_layers

            elif ldim == pdim == 0:
                # Turn in layer, create temp layer with p at bottom.
                try:
                    turn_depth = bullen_depth_for(spherical_layer,
                                                  spherical_ray_param,
                                                  self.radius_of_planet)
                except SlownessModelError:
                    if check:
                        raise
                    else:
                        turn_depth = np.nan
                spherical_layer['bot_p'] = spherical_ray_param
                spherical_layer['bot_depth'] = turn_depth

            else:
                # Turn in layer, create temp layers with each p at bottom.
                turn_depth = bullen_depth_for(
                    spherical_layer, spherical_ray_param,
                    self.radius_of_planet, check=False)
                turning_layers = np.where(turning_layers)
                spherical_layer['bot_p'][turning_layers] = \
                    spherical_ray_param[turning_layers]
                spherical_layer['bot_depth'][turning_layers] = \
                    turn_depth[turning_layers]

        if check and np.any(
                spherical_ray_param > np.maximum(spherical_layer['top_p'],
                                                 spherical_layer['bot_p'])):
            raise SlownessModelError("Ray cannot propagate within this layer, "
                                     "given ray param too large.")

        # Check to see if this layer has zero thickness, if so then it is
        # from a critically reflected slowness sample. That means just
        # return 0 for time and distance increments.
        zero_thick = \
            spherical_layer['top_depth'] == spherical_layer['bot_depth']
        if ldim == 0:
            if zero_thick:
                time.fill(0)
                dist.fill(0)
                return time, dist
            else:
                zero_thick = np.zeros(shape=time.shape, dtype=np.bool_)

        leftover = ~zero_thick
        time[zero_thick] = 0
        dist[zero_thick] = 0

        # Check to see if this layer contains the centre of the planet. If so
        # then the spherical ray parameter should be 0.0 and we calculate the
        # range and time increments using a constant velocity layer (sphere).
        # See eqns. 43 and 44 of [Buland1983]_, although we implement them
        # slightly differently. Note that the distance and time increments are
        # for just downgoing or just upgoing, i.e. from the top of the layer
        # to the centre of the planet or vice versa but not both. This is in
        # keeping with the convention that these are one way distance and time
        # increments. We will multiply the result by 2 at the end, or if we are
        # doing a 1.5D model, the other direction may be different. The time
        # increment for a ray of zero ray parameter passing half way through a
        # sphere of constant velocity is just the spherical slowness at the top
        # of the sphere. An amazingly simple result!
        centre_layer = np.logical_and(leftover, np.logical_and(
            spherical_ray_param == 0,
            spherical_layer['bot_depth'] == self.radius_of_planet))
        leftover &= ~centre_layer
        if np.any(layer_num[centre_layer] !=
                  self.get_num_layers(is_p_wave) - 1):
            raise SlownessModelError("There are layers deeper than the "
                                     "centre of the planet!")
        time[centre_layer] = spherical_layer['top_p'][centre_layer]
        dist[centre_layer] = math.pi / 2

        # Now we check to see if this is a constant velocity layer and if so
        # than we can do a simple triangle calculation to get the range and
        # time increments. To get the time increment we first calculate the
        # path length through the layer using the law of cosines, noting
        # that the angle at the top of the layer can be obtained from the
        # spherical Snell's Law. The time increment is just the path length
        # divided by the velocity. To get the distance we first find the
        # angular distance traveled, using the law of sines.
        top_radius = self.radius_of_planet - spherical_layer['top_depth']
        bot_radius = self.radius_of_planet - spherical_layer['bot_depth']
        with np.errstate(invalid='ignore'):
            vel = bot_radius / spherical_layer['bot_p']
            constant_velocity = np.logical_and(
                leftover,
                np.abs(top_radius / spherical_layer['top_p'] -
                       vel) < self.slowness_tolerance)
        leftover &= ~constant_velocity
        top_radius = top_radius[constant_velocity]
        bot_radius = bot_radius[constant_velocity]
        vel = vel[constant_velocity]
        if pdim:
            ray_param_const_velocity = spherical_ray_param[constant_velocity]
        else:
            ray_param_const_velocity = spherical_ray_param

        # In cases of a ray turning at the bottom of the layer numerical
        # round-off can cause bot_term to be very small (1e-9) but
        # negative which causes the sqrt to raise an error. We check for
        # values that are within the numerical chatter of zero and just
        # set them to zero.
        top_term = top_radius ** 2 - (ray_param_const_velocity * vel) ** 2
        top_term[np.abs(top_term) < self.slowness_tolerance] = 0

        # In this case the ray turns at the bottom of this layer so
        # spherical_ray_param*vel == bot_radius, and bot_term should be
        # zero. We check for this case specifically because
        # numerical chatter can cause small round-off errors that
        # lead to bot_term being negative, causing a sqrt error.
        bot_term = np.zeros(shape=top_term.shape)
        mask = (ray_param_const_velocity !=
                spherical_layer['bot_p'][constant_velocity])
        if pdim:
            bot_term[mask] = bot_radius[mask] ** 2 - (
                ray_param_const_velocity[mask] * vel[mask]) ** 2
        else:
            bot_term[mask] = bot_radius[mask] ** 2 - (
                ray_param_const_velocity * vel[mask]) ** 2

        b = np.sqrt(top_term) - np.sqrt(bot_term)
        time[constant_velocity] = b / vel
        dist[constant_velocity] = np.arcsin(
            b * ray_param_const_velocity * vel / (top_radius * bot_radius))

        # If the layer is not a constant velocity layer or the centre of the
        # planet and p is not zero we have to do it the hard way:
        time[leftover], dist[leftover] = bullen_radial_slowness(
            spherical_layer[leftover] if ldim else spherical_layer,
            spherical_ray_param[leftover] if pdim else spherical_ray_param,
            self.radius_of_planet,
            check=check)

        if check and (np.any(time < 0) or np.any(np.isnan(time)) or
                      np.any(dist < 0) or np.any(np.isnan(dist))):
            raise SlownessModelError(
                "layer time|dist < 0 or NaN.")

        return time, dist

    def fix_critical_points(self):
        """
        Reset the slowness layers that correspond to critical points.
        """
        self.critical_depths['p_layer_num'] = self.layer_number_below(
            self.critical_depths['depth'],
            self.p_wave)
        s_layer = self.get_slowness_layer(self.critical_depths['p_layer_num'],
                                          self.p_wave)

        # We want the last critical point to be the bottom of the last layer.
        mask = (
            (self.critical_depths['p_layer_num'] == len(self.p_layers) - 1) &
            (s_layer['bot_depth'] == self.critical_depths['depth']))
        self.critical_depths['p_layer_num'][mask] += 1

        self.critical_depths['s_layer_num'] = self.layer_number_below(
            self.critical_depths['depth'],
            self.s_wave)
        s_layer = self.get_slowness_layer(self.critical_depths['s_layer_num'],
                                          self.s_wave)

        # We want the last critical point to be the bottom of the last layer.
        mask = (
            (self.critical_depths['s_layer_num'] == len(self.s_layers) - 1) &
            (s_layer['bot_depth'] == self.critical_depths['depth']))
        self.critical_depths['s_layer_num'][mask] += 1

    def validate(self):
        """
        Perform consistency check on the slowness model.

        In Java, there is a separate validate method defined in the
        SphericalSModel subclass and as such overrides the validate in
        SlownessModel, but it itself calls the super method (by
        super.validate()), i.e. the code above. Both are merged here (in
        fact, it only contained one test).
        """
        if self.radius_of_planet <= 0:
            raise SlownessModelError("Radius of planet must be positive.")
        if self.max_depth_interval <= 0:
            raise SlownessModelError(
                "max_depth_interval must be positive and non-zero.")
        # Check for inconsistencies in high slowness zones.
        for is_p_wave in [self.p_wave, self.s_wave]:
            if is_p_wave:
                high_slowness_layer_depths = self.high_slowness_layer_depths_p
            else:
                high_slowness_layer_depths = self.high_slowness_layer_depths_s
            prev_depth = -1e300
            for high_s_zone_depth in high_slowness_layer_depths:
                if high_s_zone_depth.top_depth >= high_s_zone_depth.bot_depth:
                    raise SlownessModelError(
                        "High Slowness zone has zero or negative thickness!")
                if (high_s_zone_depth.top_depth < prev_depth or (
                        high_s_zone_depth.top_depth == prev_depth and not
                        self.v_mod.is_discontinuity(
                            high_s_zone_depth.top_depth))):
                    raise SlownessModelError(
                        "High Slowness zone overlaps previous zone.")
                prev_depth = high_s_zone_depth.bot_depth
        # Check for inconsistencies in fluid zones.
        prev_depth = -1e300
        for fluid_zone in self.fluid_layer_depths:
            if fluid_zone.top_depth >= fluid_zone.bot_depth:
                raise SlownessModelError(
                    "Fluid zone has zero or negative thickness!")
            if fluid_zone.top_depth <= prev_depth:
                raise SlownessModelError("Fluid zone overlaps previous zone.")
            prev_depth = fluid_zone.bot_depth
        # Check for inconsistencies in slowness layers.
        for layers in [self.p_layers, self.s_layers]:
            if layers is None:
                continue

            if np.any(np.isnan(layers['top_p']) | np.isnan(layers['bot_p'])):
                raise SlownessModelError("Slowness layer has NaN values.")
            if np.any((layers['top_p'] < 0) | (layers['bot_p'] < 0)):
                raise SlownessModelError(
                    "Slowness layer has negative slowness.")
            if np.any(layers['top_p'][1:] != layers['bot_p'][:-1]):
                raise SlownessModelError(
                    "Slowness layer slowness does not agree with "
                    "previous layer (at same depth)!")

            if np.any(layers['top_depth'] > layers['bot_depth']):
                raise SlownessModelError(
                    "Slowness layer has negative thickness.")

            if layers['top_depth'][0] > 0:
                raise SlownessModelError("Gap between slowness layers!")
            if np.any(layers['top_depth'][1:] > layers['bot_depth'][:-1]):
                raise SlownessModelError("Gap between slowness layers!")

            if layers['top_depth'][0] < 0:
                raise SlownessModelError("Slowness layer overlaps previous!")
            if np.any(layers['top_depth'][1:] < layers['bot_depth'][:-1]):
                raise SlownessModelError("Slowness layer overlaps previous!")

            if np.any(np.isnan(layers['top_depth']) |
                      np.isnan(layers['bot_depth'])):
                raise SlownessModelError(
                    "Slowness layer depth (top or bottom) is NaN!")

            if np.any(layers['bot_depth'] > self.radius_of_planet):
                raise SlownessModelError(
                    "Slowness layer has a depth larger than radius of the "
                    "planet.")

        # Everything seems OK.
        return True

    def get_min_turn_ray_param(self, depth, is_p_wave):
        """
        Find minimum slowness, turning but not reflected, at or above a depth.

        Normally this is the slowness sample at the given depth, but if the
        depth is within a high slowness zone, then it may be smaller.

        :param depth: The depth to search for, in km.
        :type depth: float
        :param is_p_wave: Whether to search the P (``True``) or S (``False``)
            wave.
        :type is_p_wave: bool

        :returns: The minimum ray parameter, in s/km.
        :rtype: float
        """
        min_p_so_far = 1e300
        if self.depth_in_high_slowness(depth, 1e300, is_p_wave):
            for s_layer in (self.p_layers if is_p_wave else self.s_layers):
                if s_layer['bot_depth'] == depth:
                    min_p_so_far = min(min_p_so_far, s_layer['bot_p'])
                    return min_p_so_far
                elif s_layer['bot_depth'] > depth:
                    min_p_so_far = min(
                        min_p_so_far,
                        evaluate_at_bullen(s_layer, depth,
                                           self.radius_of_planet))
                    return min_p_so_far
                else:
                    min_p_so_far = min(min_p_so_far, s_layer['bot_p'])
        else:
            s_layer = self.get_slowness_layer(
                self.layer_number_above(depth, is_p_wave), is_p_wave)
            if depth == s_layer['bot_depth']:
                min_p_so_far = s_layer['bot_p']
            else:
                min_p_so_far = evaluate_at_bullen(s_layer, depth,
                                                  self.radius_of_planet)
        return min_p_so_far

    def get_min_ray_param(self, depth, is_p_wave):
        """
        Find minimum slowness, turning or reflected, at or above a depth.

        Normally this is the slowness sample at the given depth, but if the
        depth is within a high slowness zone, then it may be smaller. Also, at
        first order discontinuities, there may be many slowness samples at the
        same depth.

        :param depth: The depth to search for, in km.
        :type depth: float
        :param is_p_wave: Whether to search the P (``True``) or S (``False``)
            wave.
        :type is_p_wave: bool

        :returns: The minimum ray parameter, in s/km.
        :rtype: float
        """
        min_p_so_far = self.get_min_turn_ray_param(depth, is_p_wave)
        s_layer_above = self.get_slowness_layer(
            self.layer_number_above(depth, is_p_wave), is_p_wave)
        s_layer_below = self.get_slowness_layer(
            self.layer_number_below(depth, is_p_wave), is_p_wave)
        if s_layer_above['bot_depth'] == depth:
            min_p_so_far = min(min_p_so_far, s_layer_above['bot_p'],
                               s_layer_below['top_p'])
        return min_p_so_far

    def split_layer(self, depth, is_p_wave):
        """
        Split a slowness layer into two slowness layers.

        The interpolation for splitting a layer is a Bullen p=Ar^B and so does
        not directly use information from the VelocityModel.

        :param depth: The depth at which attempt a split, in km.
        :type depth: float
        :param is_p_wave: Whether to split based on P (``True``) or S
            (``False``) wave.
        :type is_p_wave: bool

        :returns: Information about the split as (or if) it was performed, such
            that:

            * ``needed_split=True`` if a layer was actually split;
            * ``moved_sample=True`` if a layer was very close, and so moving
              the layer's depth is better than making a very thin layer;
            * ``ray_param=...``, the new ray parameter (in s/km), if the layer
              was split.

        :rtype: :class:`~.SplitLayerInfo`
        """
        layer_num = self.layer_number_above(depth, is_p_wave)
        s_layer = self.get_slowness_layer(layer_num, is_p_wave)
        if s_layer['top_depth'] == depth or s_layer['bot_depth'] == depth:
            # Depth is already on a slowness layer boundary so no need to
            # split any slowness layers.
            return SplitLayerInfo(self, False, False, 0)
        elif abs(s_layer['top_depth'] - depth) < 0.000001:
            # Check for very thin layers, just move the layer to hit the
            # boundary.
            out = deepcopy(self)
            out_layers = out.p_layers if is_p_wave else out.s_layers
            out_layers[layer_num] = (s_layer['top_p'], depth,
                                     s_layer['bot_p'], s_layer['bot_depth'])
            s_layer = self.get_slowness_layer(layer_num - 1, is_p_wave)
            out_layers[layer_num - 1] = (s_layer['top_p'],
                                         s_layer['top_depth'],
                                         s_layer['bot_p'], depth)
            return SplitLayerInfo(out, False, True, s_layer['bot_p'])
        elif abs(depth - s_layer['bot_depth']) < 0.000001:
            # As above.
            out = deepcopy(self)
            out_layers = out.p_layers if is_p_wave else out.s_layers
            out_layers[layer_num] = (s_layer['top_p'], s_layer['top_depth'],
                                     s_layer['bot_p'], depth)
            s_layer = self.get_slowness_layer(layer_num + 1, is_p_wave)
            out_layers[layer_num + 1] = (s_layer['top_p'], depth,
                                         s_layer['bot_p'],
                                         s_layer['bot_depth'])
            return SplitLayerInfo(out, False, True, s_layer['bot_p'])
        else:
            # Must split properly.
            out = deepcopy(self)
            p = evaluate_at_bullen(s_layer, depth, self.radius_of_planet)
            top_layer = np.array([(s_layer['top_p'], s_layer['top_depth'],
                                   p, depth)],
                                 dtype=SlownessLayer)
            bot_layer = (p, depth, s_layer['bot_p'], s_layer['bot_depth'])
            out_layers = out.p_layers if is_p_wave else out.s_layers
            out_layers[layer_num] = bot_layer
            out_layers = np.insert(out_layers, layer_num, top_layer)
            # Fix critical layers since we added a slowness layer.
            out_critical_depths = self.critical_depths
            _fix_critical_depths(out_critical_depths, layer_num, is_p_wave)
            if is_p_wave:
                out_p_layers = out_layers
                out_s_layers = self._fix_other_layers(out.s_layers, p, s_layer,
                                                      top_layer, bot_layer,
                                                      out_critical_depths,
                                                      False)
            else:
                out_p_layers = self._fix_other_layers(out.p_layers, p, s_layer,
                                                      top_layer, bot_layer,
                                                      out_critical_depths,
                                                      True)
                out_s_layers = out_layers
            out.critical_depths = out_critical_depths
            out.p_layers = out_p_layers
            out.s_layers = out_s_layers
            return SplitLayerInfo(out, True, False, p)

    def _fix_other_layers(self, other_layers, p, changed_layer, new_top_layer,
                          new_bot_layer, critical_depths, is_p_wave):
        """
        Fix other wave layers when a split is made.

        This performs the second split of the *other* wave type when a split is
        made by :meth:`split_layer`.
        """
        out = other_layers
        # Make sure to keep sampling consistent. If in a fluid, both wave
        # types will share a single slowness layer.

        other_index = np.where(other_layers == changed_layer)
        if len(other_index[0]):
            out[other_index[0]] = new_bot_layer
            out = np.insert(out, other_index[0], new_top_layer)

        number_added = 0
        for other_layer_num, s_layer in enumerate(out.copy()):
            if (s_layer['top_p'] - p) * (p - s_layer['bot_p']) > 0:
                # Found a slowness layer with the other wave type that
                # contains the new slowness sample.
                top_layer = np.array([(
                    s_layer['top_p'], s_layer['top_depth'], p,
                    bullen_depth_for(s_layer, p, self.radius_of_planet))],
                    dtype=SlownessLayer)
                bot_layer = (p, top_layer['bot_depth'],
                             s_layer['bot_p'], s_layer['bot_depth'])
                out[other_layer_num+number_added] = bot_layer
                out = np.insert(out, other_layer_num+number_added, top_layer)
                # Fix critical layers since we have added a slowness layer.
                _fix_critical_depths(critical_depths,
                                     other_layer_num, not is_p_wave)
                # Skip next layer as it was just added: achieved by slicing
                # the list iterator.
                number_added += 1

        return out
