# -*- coding: utf-8 -*-
"""
Object dealing with branches in the model.
"""
import warnings

import numpy as np

from .c_wrappers import clibtau
from .helper_classes import (SlownessLayer, SlownessModelError,
                             TauModelError, TimeDist)
from .slowness_layer import bullen_depth_for, bullen_radial_slowness


class TauBranch(object):
    """
    Provides storage and methods for distance, time and tau increments for a
    branch. A branch is a group of layers bounded by discontinuities or
    reversals in slowness gradient.
    """
    def __init__(self, top_depth=0, bot_depth=0, is_p_wave=False):
        self.top_depth = top_depth
        self.bot_depth = bot_depth
        self.is_p_wave = is_p_wave
        self.debug = False

    def __str__(self):
        desc = "Tau Branch\n"
        desc += " top_depth = " + str(self.top_depth) + "\n"
        desc += " bot_depth = " + str(self.bot_depth) + "\n"
        desc += " max_ray_param=" + str(self.max_ray_param) + \
            " min_turn_ray_param=" + str(self.min_turn_ray_param)
        desc += " min_ray_param=" + str(self.min_ray_param) + "\n"
        return desc

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def create_branch(self, s_mod, min_p_so_far, ray_params):
        """
        Calculates tau for this branch, between slowness layers top_layer_num
        and bot_layer_num, inclusive.
        """
        top_layer_num = s_mod.layer_number_below(self.top_depth,
                                                 self.is_p_wave)
        bot_layer_num = s_mod.layer_number_above(self.bot_depth,
                                                 self.is_p_wave)
        top_s_layer = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave)
        bot_s_layer = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave)
        if top_s_layer['top_depth'] != self.top_depth \
                or bot_s_layer['bot_depth'] != self.bot_depth:
            if top_s_layer['top_depth'] != self.top_depth \
                    and abs(top_s_layer['top_depth'] -
                            self.top_depth) < 0.000001:
                # Really close, so just move the top.
                print("Changing top_depth" + str(self.top_depth) + "-->" +
                      str(top_s_layer.top_depth))
                self.top_depth = top_s_layer['top_depth']
            elif bot_s_layer['bot_depth'] != self.bot_depth and \
                    abs(bot_s_layer['bot_depth'] - self.bot_depth) < 0.000001:
                # Really close, so just move the bottom.
                print("Changing bot_depth" + str(self.bot_depth) + "-->" +
                      str(bot_s_layer['bot_depth']))
                self.bot_depth = bot_s_layer['bot_depth']
            else:
                raise TauModelError("create_branch: TauBranch not compatible "
                                    "with slowness sampling at top_depth" +
                                    str(self.top_depth))
        # Here we set min_turn_ray_param to be the ray parameter that turns
        # within the layer, not including total reflections off of the bottom.
        # max_ray_param is the largest ray parameter that can penetrate this
        # branch. min_ray_param is the minimum ray parameter that turns or is
        # totally reflected in this branch.
        self.max_ray_param = min_p_so_far
        self.min_turn_ray_param = s_mod.get_min_turn_ray_param(
            self.bot_depth, self.is_p_wave)
        self.min_ray_param = s_mod.get_min_ray_param(self.bot_depth,
                                                     self.is_p_wave)

        time_dist = self.calc_time_dist(s_mod, top_layer_num, bot_layer_num,
                                        ray_params)
        self.time = time_dist['time']
        self.dist = time_dist['dist']
        self.tau = self.time - ray_params * self.dist

    def calc_time_dist(self, s_mod, top_layer_num, bot_layer_num, ray_params,
                       allow_turn_in_layer=False):
        time_dist = np.zeros(shape=ray_params.shape, dtype=TimeDist)
        time_dist['p'] = ray_params

        layer_num = np.arange(top_layer_num, bot_layer_num + 1)
        layer = s_mod.get_slowness_layer(layer_num, self.is_p_wave)

        plen = len(ray_params)
        llen = len(layer_num)
        ray_params = np.repeat(ray_params, llen).reshape((plen, llen))
        layer_num = np.tile(layer_num, plen).reshape((plen, llen))

        # Ignore some errors because we pass in a few invalid combinations that
        # are masked out later.
        with np.errstate(divide='ignore', invalid='ignore'):
            time, dist = s_mod.layer_time_dist(
                ray_params, layer_num, self.is_p_wave, check=False,
                allow_turn=True)

        clibtau.tau_branch_calc_time_dist_inner_loop(
            ray_params, time, dist, layer, time_dist, ray_params.shape[0],
            ray_params.shape[1], self.max_ray_param, allow_turn_in_layer)

        return time_dist

    def insert(self, ray_param, s_mod, index):
        """
        Inserts the distance, time, and tau increment for the slowness sample
        given to the branch. This is used for making the depth correction to a
        tau model for a non-surface source.
        """
        top_layer_num = s_mod.layer_number_below(self.top_depth,
                                                 self.is_p_wave)
        bot_layer_num = s_mod.layer_number_above(self.bot_depth,
                                                 self.is_p_wave)
        top_s_layer = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave)
        bot_s_layer = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave)
        if top_s_layer['top_depth'] != self.top_depth \
                or bot_s_layer['bot_depth'] != self.bot_depth:
            raise TauModelError(
                "TauBranch depths not compatible with slowness sampling.")

        new_time = 0.0
        new_dist = 0.0
        if top_s_layer['bot_p'] >= ray_param and \
                top_s_layer['top_p'] >= ray_param:
            layer_num = np.arange(top_layer_num, bot_layer_num + 1)
            layers = s_mod.get_slowness_layer(layer_num, self.is_p_wave)
            # So we don't sum below the turning depth.
            mask = np.cumprod(layers['bot_p'] >= ray_param).astype(np.bool_)
            layer_num = layer_num[mask]
            if len(layer_num):
                time, dist = s_mod.layer_time_dist(ray_param, layer_num,
                                                   self.is_p_wave)
                new_time = np.sum(time)
                new_dist = np.sum(dist)

        self.shift_branch(index)
        self.time[index] = new_time
        self.dist[index] = new_dist
        self.tau[index] = new_time - ray_param * new_dist

    def shift_branch(self, index):
        new_size = len(self.dist) + 1

        self._robust_resize('time', new_size)
        self.time[index + 1:] = self.time[index:-1]
        self.time[index] = 0

        self._robust_resize('dist', new_size)
        self.dist[index + 1:] = self.dist[index:-1]
        self.dist[index] = 0

        self._robust_resize('tau', new_size)
        self.tau[index + 1:] = self.tau[index:-1]
        self.tau[index] = 0

    def difference(self, top_branch, index_p, index_s, s_mod, min_p_so_far,
                   ray_params):
        """
        Generates a new tau branch by "subtracting" the given tau branch from
        this tau branch (self). The given tau branch is assumed to by the
        upper part of this branch. index_p specifies where a new ray
        corresponding to a P wave sample has been added; it is -1 if no ray
        parameter has been added to top_branch. index_s  is similar to index_p
        except for a S wave sample. Note that although the ray parameters
        for index_p and index_s were for the P and S waves that turned at the
        source depth, both ray parameters need to be added to both P and S
        branches.
        """
        if top_branch.top_depth != self.top_depth \
                or top_branch.bot_depth > self.bot_depth:
            if top_branch.top_depth != self.top_depth \
                    and abs(top_branch.top_depth - self.top_depth) < 0.000001:
                # Really close, just move top.
                self.top_depth = top_branch.top_depth
            else:
                raise TauModelError(
                    "TauBranch not compatible with slowness sampling.")
        if top_branch.is_p_wave != self.is_p_wave:
            raise TauModelError(
                "Can't subtract branches is is_p_wave doesn't agree.")
        # Find the top and bottom slowness layers of the bottom half.
        top_layer_num = s_mod.layer_number_below(top_branch.bot_depth,
                                                 self.is_p_wave)
        bot_layer_num = s_mod.layer_number_below(self.bot_depth,
                                                 self.is_p_wave)
        top_s_layer = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave)
        bot_s_layer = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave)
        if bot_s_layer['top_depth'] == self.bot_depth \
                and bot_s_layer['bot_depth'] > self.bot_depth:
            # Gone one too far.
            bot_layer_num -= 1
            bot_s_layer = s_mod.get_slowness_layer(bot_layer_num,
                                                   self.is_p_wave)
        if top_s_layer['top_depth'] != top_branch.bot_depth \
                or bot_s_layer['bot_depth'] != self.bot_depth:
            raise TauModelError(
                "TauBranch not compatible with slowness sampling.")
        # Make sure index_p and index_s really correspond to new ray
        # parameters at the top of this branch.
        s_layer = s_mod.get_slowness_layer(s_mod.layer_number_below(
            top_branch.bot_depth, True), True)
        if index_p >= 0 and s_layer['top_p'] != ray_params[index_p]:
            raise TauModelError("P wave index doesn't match top layer.")
        s_layer = s_mod.get_slowness_layer(s_mod.layer_number_below(
            top_branch.bot_depth, False), False)
        if index_s >= 0 and s_layer['top_p'] != ray_params[index_s]:
            raise TauModelError("S wave index doesn't match top layer.")
        del s_layer
        # Construct the new TauBranch, going from the bottom of the top half
        # to the bottom of the whole branch.
        bot_branch = TauBranch(top_branch.bot_depth, self.bot_depth,
                               self.is_p_wave)
        bot_branch.max_ray_param = top_branch.min_ray_param
        bot_branch.min_turn_ray_param = self.min_turn_ray_param
        bot_branch.min_ray_param = self.min_ray_param
        p_ray_param = -1
        s_ray_param = -1
        array_length = len(self.dist)
        if index_p != -1:
            array_length += 1
            p_ray_param = ray_params[index_p:index_p + 1]
            time_dist_p = bot_branch.calc_time_dist(
                s_mod, top_layer_num, bot_layer_num, p_ray_param)
        if index_s != -1 and index_s != index_p:
            array_length += 1
            s_ray_param = ray_params[index_s:index_s + 1]
            time_dist_s = bot_branch.calc_time_dist(s_mod, top_layer_num,
                                                    bot_layer_num, s_ray_param)
        else:
            # In case index_s==P then only need one.
            index_s = -1

        if index_p == -1:
            # Then both indices are -1 so no new ray parameters are added.
            bot_branch.time = self.time - top_branch.time
            bot_branch.dist = self.dist - top_branch.dist
            bot_branch.tau = self.tau - top_branch.tau
        else:
            bot_branch.time = np.empty(array_length)
            bot_branch.dist = np.empty(array_length)
            bot_branch.tau = np.empty(array_length)

            if index_s == -1:
                # Only index_p != -1.
                bot_branch.time[:index_p] = (self.time[:index_p] -
                                             top_branch.time[:index_p])
                bot_branch.dist[:index_p] = (self.dist[:index_p] -
                                             top_branch.dist[:index_p])
                bot_branch.tau[:index_p] = (self.tau[:index_p] -
                                            top_branch.tau[:index_p])

                bot_branch.time[index_p] = time_dist_p['time']
                bot_branch.dist[index_p] = time_dist_p['dist']
                bot_branch.tau[index_p] = (time_dist_p['time'] -
                                           p_ray_param * time_dist_p['dist'])

                bot_branch.time[index_p + 1:] = (self.time[index_p:] -
                                                 top_branch.time[index_p + 1:])
                bot_branch.dist[index_p + 1:] = (self.dist[index_p:] -
                                                 top_branch.dist[index_p + 1:])
                bot_branch.tau[index_p + 1:] = (self.tau[index_p:] -
                                                top_branch.tau[index_p + 1:])

            else:
                # Both index_p and S are != -1 so have two new samples
                bot_branch.time[:index_s] = (self.time[:index_s] -
                                             top_branch.time[:index_s])
                bot_branch.dist[:index_s] = (self.dist[:index_s] -
                                             top_branch.dist[:index_s])
                bot_branch.tau[:index_s] = (self.tau[:index_s] -
                                            top_branch.tau[:index_s])

                bot_branch.time[index_s] = time_dist_s['time']
                bot_branch.dist[index_s] = time_dist_s['dist']
                bot_branch.tau[index_s] = (time_dist_s['time'] -
                                           s_ray_param * time_dist_s['dist'])

                bot_branch.time[index_s + 1:index_p] = (
                    self.time[index_s:index_p - 1] -
                    top_branch.time[index_s + 1:index_p])
                bot_branch.dist[index_s + 1:index_p] = (
                    self.dist[index_s:index_p - 1] -
                    top_branch.dist[index_s + 1:index_p])
                bot_branch.tau[index_s + 1:index_p] = (
                    self.tau[index_s:index_p - 1] -
                    top_branch.tau[index_s + 1:index_p])

                bot_branch.time[index_p] = time_dist_p['time']
                bot_branch.dist[index_p] = time_dist_p['dist']
                bot_branch.tau[index_p] = (time_dist_p['time'] -
                                           p_ray_param * time_dist_p['dist'])

                bot_branch.time[index_p + 1:] = (self.time[index_p - 1:] -
                                                 top_branch.time[index_p + 1:])
                bot_branch.dist[index_p + 1:] = (self.dist[index_p - 1:] -
                                                 top_branch.dist[index_p + 1:])
                bot_branch.tau[index_p + 1:] = (self.tau[index_p - 1:] -
                                                top_branch.tau[index_p + 1:])

        return bot_branch

    def path(self, ray_param, downgoing, s_mod):
        """
        Called from TauPPath to calculate ray paths.
        :param ray_param:
        :param downgoing:
        :param s_mod:
        :return:
        """
        if ray_param > self.max_ray_param:
            return np.empty(0, dtype=TimeDist)
        assert ray_param >= 0

        try:
            top_layer_num = s_mod.layer_number_below(self.top_depth,
                                                     self.is_p_wave)
            bot_layer_num = s_mod.layer_number_above(self.bot_depth,
                                                     self.is_p_wave)
        # except NoSuchLayerError as e:
        except SlownessModelError:
            raise SlownessModelError("SlownessModel and TauModel are likely"
                                     "out of sync.")

        the_path = np.empty(bot_layer_num - top_layer_num + 1, dtype=TimeDist)
        path_index = 0

        # Check to make sure layers and branches are compatible.
        s_layer = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave)
        if s_layer['top_depth'] != self.top_depth:
            raise SlownessModelError("Branch and slowness model are not in "
                                     "agreement.")
        s_layer = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave)
        if s_layer['bot_depth'] != self.bot_depth:
            raise SlownessModelError("Branch and slowness model are not in "
                                     "agreement.")

        # Downgoing branches:
        if downgoing:
            s_layer_num = np.arange(top_layer_num, bot_layer_num + 1)
            s_layer = s_mod.get_slowness_layer(s_layer_num, self.is_p_wave)

            mask = np.cumprod(s_layer['bot_p'] >= ray_param).astype(np.bool_)
            mask &= s_layer['top_depth'] != s_layer['bot_depth']
            s_layer_num = s_layer_num[mask]
            s_layer = s_layer[mask]

            if len(s_layer):
                path_index_end = path_index + len(s_layer)
                time, dist = s_mod.layer_time_dist(
                    ray_param,
                    s_layer_num,
                    self.is_p_wave)
                the_path[path_index:path_index_end]['p'] = ray_param
                the_path[path_index:path_index_end]['time'] = time
                the_path[path_index:path_index_end]['dist'] = dist
                the_path[path_index:path_index_end]['depth'] = \
                    s_layer['bot_depth']
                path_index = path_index_end

            # Apply Bullen laws on last element, if available.
            if len(s_layer_num):
                s_layer_num = s_layer_num[-1] + 1
            else:
                s_layer_num = top_layer_num
            if s_layer_num <= bot_layer_num:
                s_layer = s_mod.get_slowness_layer(s_layer_num, self.is_p_wave)
                if s_layer['top_depth'] != s_layer['bot_depth']:
                    turn_depth = bullen_depth_for(s_layer, ray_param,
                                                  s_mod.radius_of_planet)
                    turn_s_layer = np.array([
                        (s_layer['top_p'], s_layer['top_depth'], ray_param,
                         turn_depth)], dtype=SlownessLayer)
                    time, dist = bullen_radial_slowness(
                        turn_s_layer,
                        ray_param,
                        s_mod.radius_of_planet)
                    the_path[path_index]['p'] = ray_param
                    the_path[path_index]['time'] = time
                    the_path[path_index]['dist'] = dist
                    the_path[path_index]['depth'] = turn_s_layer['bot_depth']
                    path_index += 1

        # Upgoing branches:
        else:
            s_layer_num = np.arange(bot_layer_num, top_layer_num - 1, -1)
            s_layer = s_mod.get_slowness_layer(s_layer_num, self.is_p_wave)

            mask = np.logical_or(s_layer['top_p'] <= ray_param,
                                 s_layer['top_depth'] == s_layer['bot_depth'])
            mask = np.cumprod(mask).astype(np.bool_)
            mask[-1] = False  # Always leave one element for Bullen.

            # Apply Bullen laws on first available element, if possible.
            first_unmasked = np.sum(mask)
            s_layer_2 = s_layer[first_unmasked]
            if s_layer_2['bot_p'] < ray_param:
                turn_depth = bullen_depth_for(s_layer_2, ray_param,
                                              s_mod.radius_of_planet)
                turn_s_layer = np.array([(
                    s_layer_2['top_p'], s_layer_2['top_depth'], ray_param,
                    turn_depth)], dtype=SlownessLayer)
                time, dist = bullen_radial_slowness(
                    turn_s_layer,
                    ray_param,
                    s_mod.radius_of_planet)
                the_path[path_index]['p'] = ray_param
                the_path[path_index]['time'] = time
                the_path[path_index]['dist'] = dist
                the_path[path_index]['depth'] = turn_s_layer['top_depth']
                path_index += 1
                mask[first_unmasked] = True

            # Apply regular time/distance calculation on all unmasked and
            # non-zero thickness layers.
            mask = (~mask) & (s_layer['top_depth'] != s_layer['bot_depth'])
            s_layer = s_layer[mask]
            s_layer_num = s_layer_num[mask]

            if len(s_layer):
                path_index_end = path_index + len(s_layer)
                time, dist = s_mod.layer_time_dist(
                    ray_param,
                    s_layer_num,
                    self.is_p_wave)
                the_path[path_index:path_index_end]['p'] = ray_param
                the_path[path_index:path_index_end]['time'] = time
                the_path[path_index:path_index_end]['dist'] = dist
                the_path[path_index:path_index_end]['depth'] = \
                    s_layer['top_depth']
                path_index = path_index_end

        temp_path = the_path[:path_index]
        return temp_path

    def _to_array(self):
        """
        Store all attributes for serialization in a structured array.
        """
        dtypes = [('debug', np.bool_),
                  ('bot_depth', np.float_),
                  ('dist', np.float_, self.dist.shape),
                  ('is_p_wave', np.bool_),
                  ('max_ray_param', np.float_),
                  ('min_ray_param', np.float_),
                  ('min_turn_ray_param', np.float_),
                  ('tau', np.float_, self.tau.shape),
                  ('time', np.float_, self.time.shape),
                  ('top_depth', np.float_)]
        arr = np.empty(shape=(), dtype=dtypes)
        for dtype in dtypes:
            key = dtype[0]
            arr[key] = getattr(self, key)
        return arr

    @staticmethod
    def _from_array(arr):
        """
        Create instance object from a structured array used in serialization.
        """
        branch = TauBranch()
        for key in arr.dtype.names:
            # restore scalar types from 0d array
            arr_ = arr[key]
            if arr_.ndim == 0:
                arr_ = arr_[()]
            setattr(branch, key, arr_)
        return branch

    def _robust_resize(self, attr, new_size):
        """
        Try to resize an array inplace. If an error is raised use numpy
        resize function to create a new array. Assign the array to self as
        attribute listed in attr.
        """
        try:
            getattr(self, attr).resize(new_size)
        except ValueError:
            msg = ('Resizing a TauP array inplace failed due to the '
                   'existence of other references to the array, creating '
                   'a new array. See Obspy #2280.')
            warnings.warn(msg)
            setattr(self, attr, np.resize(getattr(self, attr), new_size))
