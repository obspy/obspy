# -*- coding: utf-8 -*-
"""
Internal TauModel class.
"""
from collections import OrderedDict
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
    def __init__(self, s_mod, radius_of_planet, is_spherical=True, cache=None,
                 debug=False, skip_calc=False):
        self.debug = debug
        # Depth for which tau model as constructed.
        self.source_depth = 0.0
        self.radius_of_planet = radius_of_planet
        # True if this is a spherical slowness model. False if flat.
        self.is_spherical = is_spherical
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
        self.tau_branches = None

        self.s_mod = s_mod

        # Branch with the source at its top.
        self.source_branch = 0
        # Depths that should not have reflections or phase conversions. For
        # instance, if the source is not at a branch boundary then
        # no_discon_depths contains source depth and reflections and phase
        # conversions are not allowed at this branch boundary. If the source
        # happens to fall on a real discontinuity then it is not included.
        self.no_discon_depths = []

        if cache is None:
            self._depth_cache = OrderedDict()
        elif cache is not False:
            self._depth_cache = cache
        else:
            self._depth_cache = None

        if not skip_calc:
            self.calc_tau_inc_from()

    def calc_tau_inc_from(self):
        """
        Calculates tau for each branch within a slowness model.
        """
        # First, we must have at least 1 slowness layer to calculate a
        #  distance. Otherwise we must signal an exception.
        if self.s_mod.get_num_layers(True) == 0 \
                or self.s_mod.get_num_layers(False) == 0:
            raise SlownessModelError(
                "Can't calculate tauInc when get_num_layers() = 0. "
                "I need more slowness samples.")
        self.s_mod.validate()
        # Create an array holding the ray parameter that we will use for
        # constructing the tau splines. Only store ray parameters that are
        # not in a high slowness zone, i.e. they are smaller than the
        # minimum ray parameter encountered so far.
        num_branches = len(self.s_mod.critical_depths) - 1
        self.tau_branches = np.empty((2, num_branches), dtype=TauBranch)
        # Here we find the list of ray parameters to be used for the tau
        # model. We only need to find ray parameters for S waves since P
        # waves have been constructed to be a subset of the S samples.
        ray_num = 0
        min_p_so_far = self.s_mod.s_layers[0]['top_p']
        temp_ray_params = np.empty(2 * self.s_mod.get_num_layers(False) +
                                   len(self.s_mod.critical_depths))
        # Make sure we get the top slowness of the very top layer
        temp_ray_params[ray_num] = min_p_so_far
        ray_num += 1
        for curr_s_layer in self.s_mod.s_layers:
            # Add the top if it is strictly less than the last sample added.
            # Note that this will not be added if the slowness is continuous
            #  across the layer boundary.
            if curr_s_layer['top_p'] < min_p_so_far:
                temp_ray_params[ray_num] = curr_s_layer['top_p']
                ray_num += 1
                min_p_so_far = curr_s_layer['top_p']
            if curr_s_layer['bot_p'] < min_p_so_far:
                # Add the bottom if it is strictly less than the last sample
                # added. This will always happen unless we are
                # within a high slowness zone.
                temp_ray_params[ray_num] = curr_s_layer['bot_p']
                ray_num += 1
                min_p_so_far = curr_s_layer['bot_p']
        # Copy tempRayParams to ray_param while chopping off trailing zeros
        # (from the initialisation), so the size is exactly right. NB
        # slicing doesn't really mean deep copy, but it works for a list of
        # doubles like this
        self.ray_params = temp_ray_params[:ray_num]
        if self.debug:
            print("Number of slowness samples for tau:" + str(ray_num))
        for wave_num, is_p_wave in enumerate([True, False]):
            # The minimum slowness seen so far.
            min_p_so_far = self.s_mod.get_slowness_layer(0, is_p_wave)['top_p']
            # for critNum, (topCritDepth, botCritDepth) in enumerate(zip(
            # self.s_mod.critical_depths[:-1],
            # self.s_mod.critical_depths[1:])):
            # Faster:
            for crit_num, top_crit_depth, bot_crit_depth in zip(
                    count(), self.s_mod.critical_depths[:-1],
                    self.s_mod.critical_depths[1:]):
                top_crit_layer_num = top_crit_depth['p_layer_num'] \
                    if is_p_wave else top_crit_depth['s_layer_num']
                bot_crit_layer_num = (
                    bot_crit_depth['p_layer_num']
                    if is_p_wave else bot_crit_depth['s_layer_num']) - 1
                self.tau_branches[wave_num, crit_num] = \
                    TauBranch(top_crit_depth['depth'], bot_crit_depth['depth'],
                              is_p_wave)
                self.tau_branches[wave_num, crit_num].debug = self.debug
                self.tau_branches[wave_num, crit_num].create_branch(
                    self.s_mod, min_p_so_far, self.ray_params)
                # Update minPSoFar. Note that the new minPSoFar could be at
                # the start of a discontinuity over a high slowness zone,
                # so we need to check the top, bottom and the layer just
                # above the discontinuity.
                top_s_layer = self.s_mod.get_slowness_layer(top_crit_layer_num,
                                                            is_p_wave)
                bot_s_layer = self.s_mod.get_slowness_layer(bot_crit_layer_num,
                                                            is_p_wave)
                min_p_so_far = min(
                    min_p_so_far, min(top_s_layer['top_p'],
                                      bot_s_layer['bot_p']))
                bot_s_layer = self.s_mod.get_slowness_layer(
                    self.s_mod.layer_number_above(bot_crit_depth['depth'],
                                                  is_p_wave), is_p_wave)
                min_p_so_far = min(min_p_so_far, bot_s_layer['bot_p'])
        # Here we decide which branches are the closest to the Moho, CMB,
        # and IOCB by comparing the depth of the top of the branch with the
        # depths in the Velocity Model.
        best_moho = 1e300
        best_cmb = 1e300
        best_iocb = 1e300
        for branch_num, t_branch in enumerate(self.tau_branches[0]):
            if abs(t_branch.top_depth - self.s_mod.v_mod.moho_depth) <= \
                    best_moho:
                # Branch with Moho at its top.
                self.moho_branch = branch_num
                best_moho = abs(t_branch.top_depth -
                                self.s_mod.v_mod.moho_depth)
            if abs(t_branch.top_depth - self.s_mod.v_mod.cmb_depth) < best_cmb:
                self.cmb_branch = branch_num
                best_cmb = abs(t_branch.top_depth - self.s_mod.v_mod.cmb_depth)
            if abs(t_branch.top_depth - self.s_mod.v_mod.iocb_depth) < \
                    best_iocb:
                self.iocb_branch = branch_num
                best_iocb = abs(t_branch.top_depth -
                                self.s_mod.v_mod.iocb_depth)
        # Now set moho_depth etc. to the top of the branches we have decided
        # on.
        self.moho_depth = self.tau_branches[0, self.moho_branch].top_depth
        self.cmb_depth = self.tau_branches[0, self.cmb_branch].top_depth
        self.iocb_depth = self.tau_branches[0, self.iocb_branch].top_depth
        self.validate()

    def __str__(self):
        desc = "Delta tau for each slowness sample and layer.\n"
        for j, ray_param in enumerate(self.ray_params):
            for i, tb in enumerate(self.tau_branches[0]):
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
        Called in TauPTime. Computes a new tau model for a source at depth
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
        if depth > self.radius_of_planet:
            raise TauModelError("Can't depth correct to a source deeper than "
                                "the radius of the planet.")
        return self.load_from_depth_cache(depth)

    def load_from_depth_cache(self, depth):
        # Very simple and straightforward LRU cache implementation.
        if self._depth_cache is not None:
            # Retrieve and later insert again to get LRU cache behaviour.
            try:
                value = self._depth_cache.pop(depth)
            except KeyError:
                value = self._load_from_depth_cache(depth)
            self._depth_cache[depth] = value
            # Pop first key-value pairs until at most 128 elements are still
            # in the cache.
            while len(self._depth_cache) > 128:
                self._depth_cache.popitem(last=False)
            return value
        else:
            return self._load_from_depth_cache(depth)

    def _load_from_depth_cache(self, depth):
        depth_corrected = self.split_branch(depth)
        depth_corrected.source_depth = depth
        depth_corrected.source_branch = depth_corrected.find_branch(depth)
        depth_corrected.validate()
        return depth_corrected

    def split_branch(self, depth):
        """
        Returns a new TauModel with the branches containing depth split at
        depth. Used for putting a source at depth since a source can only be
        located on a branch boundary.
         """
        # First check to see if depth happens to already be a branch
        # boundary, then just return original model.
        for tb in self.tau_branches[0]:
            if tb.top_depth == depth or tb.bot_depth == depth:
                return deepcopy(self)
        # Depth is not a branch boundary, so must modify the tau model.
        index_p = -1
        p_wave_ray_param = -1
        index_s = -1
        s_wave_ray_param = -1
        out_s_mod = self.s_mod
        out_ray_params = self.ray_params
        # Do S wave first since the S ray param is > P ray param.
        for is_p_wave in [False, True]:
            split_info = out_s_mod.split_layer(depth, is_p_wave)
            out_s_mod = split_info.s_mod
            if split_info.needed_split and not split_info.moved_sample:
                # Split the slowness layers containing depth into two layers
                # each.
                new_ray_param = split_info.ray_param
                # Insert the new ray parameters into the ray_param array.
                above = out_ray_params[:-1]
                below = out_ray_params[1:]
                index = (above > new_ray_param) & (new_ray_param > below)
                if np.any(index):
                    index = np.where(index)[0][0] + 1
                    out_ray_params = np.insert(out_ray_params, index,
                                               new_ray_param)

                    if is_p_wave:
                        index_p = index
                        p_wave_ray_param = new_ray_param
                    else:
                        index_s = index
                        s_wave_ray_param = new_ray_param

        # Now add a sample to each branch above the depth, split the branch
        # containing the depth, and add a sample to each deeper branch.
        branch_to_split = self.find_branch(depth)
        new_tau_branches = np.empty((2, self.tau_branches.shape[1] + 1),
                                    dtype=TauBranch)
        for i in range(branch_to_split):
            new_tau_branches[0, i] = deepcopy(self.tau_branches[0, i])
            new_tau_branches[1, i] = deepcopy(self.tau_branches[1, i])
            # Add the new ray parameter(s) from splitting the S and/or P
            # wave slowness layer to both the P and S wave tau branches (if
            # splitting occurred).
            if index_s != -1:
                new_tau_branches[0, i].insert(s_wave_ray_param, out_s_mod,
                                              index_s)
                new_tau_branches[1, i].insert(s_wave_ray_param, out_s_mod,
                                              index_s)
            if index_p != -1:
                new_tau_branches[0, i].insert(p_wave_ray_param, out_s_mod,
                                              index_p)
                new_tau_branches[1, i].insert(p_wave_ray_param, out_s_mod,
                                              index_p)
        for p_or_s in range(2):
            new_tau_branches[p_or_s, branch_to_split] = TauBranch(
                self.tau_branches[p_or_s, branch_to_split].top_depth, depth,
                p_or_s == 0)
            new_tau_branches[p_or_s, branch_to_split].create_branch(
                out_s_mod,
                self.tau_branches[p_or_s, branch_to_split].max_ray_param,
                out_ray_params)
            new_tau_branches[p_or_s, branch_to_split + 1] = \
                self.tau_branches[p_or_s, branch_to_split].difference(
                    new_tau_branches[p_or_s, branch_to_split],
                    index_p, index_s, out_s_mod,
                    new_tau_branches[p_or_s, branch_to_split].min_ray_param,
                    out_ray_params)
        for i in range(branch_to_split + 1, len(self.tau_branches[0])):
            for p_or_s in range(2):
                new_tau_branches[p_or_s, i + 1] =  \
                    deepcopy(self.tau_branches[p_or_s, i])
            if index_s != -1:
                # Add the new ray parameter from splitting the S wave
                # slownes layer to both the P and S wave tau branches.
                for p_or_s in range(2):
                    new_tau_branches[p_or_s, i + 1].insert(
                        s_wave_ray_param, out_s_mod, index_s)
            if index_p != -1:
                # Add the new ray parameter from splitting the P wave
                # slownes layer to both the P and S wave tau branches.
                for p_or_s in range(2):
                    new_tau_branches[p_or_s, i + 1].insert(
                        p_wave_ray_param, out_s_mod, index_p)
        # We have split a branch so possibly source_branch, moho_branch,
        # cmb_branch and iocb_branch are off by 1.
        out_source_branch = self.source_branch
        if self.source_depth > depth:
            out_source_branch += 1
        out_moho_branch = self.moho_branch
        if self.moho_depth > depth:
            out_moho_branch += 1
        out_cmb_branch = self.cmb_branch
        if self.cmb_depth > depth:
            out_cmb_branch += 1
        out_iocb_branch = self.iocb_branch
        if self.iocb_depth > depth:
            out_iocb_branch += 1
        # No overloaded constructors - so do it this way to bypass the
        # calc_tau_inc_from in the __init__.
        tau_model = TauModel(
            out_s_mod,
            radius_of_planet=out_s_mod.v_mod.radius_of_planet,
            is_spherical=self.is_spherical, cache=False,
            debug=self.debug, skip_calc=True)
        tau_model.source_depth = self.source_depth
        tau_model.source_branch = out_source_branch
        tau_model.moho_branch = out_moho_branch
        tau_model.moho_depth = self.moho_depth
        tau_model.cmb_branch = out_cmb_branch
        tau_model.cmb_depth = self.cmb_depth
        tau_model.iocb_branch = out_iocb_branch
        tau_model.iocb_depth = self.iocb_depth
        tau_model.ray_params = out_ray_params
        tau_model.tau_branches = new_tau_branches
        tau_model.no_discon_depths = self.no_discon_depths + [depth]
        tau_model.validate()
        return tau_model

    def find_branch(self, depth):
        """Finds the branch that either has the depth as its top boundary, or
        strictly contains the depth. Also, we allow the bottom-most branch to
        contain its bottom depth, so that the center of the planet is contained
        within the bottom branch."""
        for i, tb in enumerate(self.tau_branches[0]):
            if tb.top_depth <= depth < tb.bot_depth:
                return i
        # Check to see if depth is centre of the planet.
        if self.tau_branches[0, -1].bot_depth == depth:
            return len(self.tau_branches) - 1
        else:
            raise TauModelError("No TauBranch contains this depth.")

    def get_tau_branch(self, branch_nu, is_p_wave):
        if is_p_wave:
            return self.tau_branches[0, branch_nu]
        else:
            return self.tau_branches[1, branch_nu]

    def get_branch_depths(self):
        """
        Return an array of the depths that are boundaries between branches.
        :return:
        """
        branch_depths = [self.get_tau_branch(0, True).top_depth]
        branch_depths += [
            self.get_tau_branch(i - 1, True).bot_depth
            for i in range(1, len(self.tau_branches[0]))]
        return branch_depths

    def serialize(self, filename):
        """
        Serialize model to numpy npz binary file.

        Summary of contents that have to be handled during serialization::

            TauModel
            ========
            cmb_branch <type 'int'>
            cmb_depth <type 'float'>
            debug <type 'bool'>
            iocb_branch <type 'int'>
            iocb_depth <type 'float'>
            moho_branch <type 'int'>
            moho_depth <type 'float'>
            no_discon_depths <type 'list'> (of float!?)
            radius_of_planet <type 'float'>
            ray_params <type 'numpy.ndarray'> (1D, float)
            s_mod <class 'obspy.taup.slowness_model.SlownessModel'>
            source_branch <type 'int'>
            source_depth <type 'float'>
            is_spherical <type 'bool'>
            tau_branches <type 'numpy.ndarray'> (2D, type TauBranch)

            TauBranch
            =========
            debug <type 'bool'>
            bot_depth <type 'float'>
            dist <type 'numpy.ndarray'>
            is_p_wave <type 'bool'>
            max_ray_param <type 'float'>
            min_ray_param <type 'float'>
            min_turn_ray_param <type 'float'>
            tau <type 'numpy.ndarray'>
            time <type 'numpy.ndarray'>
            top_depth <type 'float'>

            SlownessModel
            =============
            debug <type 'bool'>
            p_layers <type 'numpy.ndarray'>
            p_wave <type 'bool'>
            s_layers <type 'numpy.ndarray'>
            s_wave <type 'bool'>
            allow_inner_core_s <type 'bool'>
            critical_depths <type 'numpy.ndarray'>
            fluid_layer_depths <type 'list'> (of DepthRange)
            high_slowness_layer_depths_p <type 'list'> (of DepthRange)
            high_slowness_layer_depths_s <type 'list'> (of DepthRange)
            max_delta_p <type 'float'>
            max_depth_interval <type 'float'>
            max_interp_error <type 'float'>
            max_range_interval <type 'float'>
            min_delta_p <type 'float'>
            radius_of_planet <type 'float'>
            slowness_tolerance <type 'float'>
            v_mod <class 'obspy.taup.velocity_model.VelocityModel'>

            VelocityModel
            =============
            cmb_depth <type 'float'>
            iocb_depth <type 'float'>
            is_spherical <type 'bool'>
            layers <type 'numpy.ndarray'>
            max_radius <type 'float'>
            min_radius <type 'int'>
            model_name <type 'unicode'>
            moho_depth <type 'float'>
            radius_of_planet <type 'float'>
        """
        # a) handle simple contents
        keys = ['cmb_branch', 'cmb_depth', 'debug', 'iocb_branch',
                'iocb_depth', 'moho_branch', 'moho_depth', 'no_discon_depths',
                'radius_of_planet', 'ray_params', 'source_branch',
                'source_depth', 'is_spherical']
        arrays = {k: getattr(self, k) for k in keys}

        # b) handle .tau_branches
        i, j = self.tau_branches.shape
        for j_ in range(j):
            for i_ in range(i):
                # just store the shape of self.tau_branches in the key names
                # for later reconstruction of array in deserialization.
                key = 'tau_branches__%i/%i__%i/%i' % (j_, j, i_, i)
                arrays[key] = self.tau_branches[i_][j_]._to_array()

        # c) handle simple contents of .s_mod
        dtypes = [('debug', np.bool_),
                  ('p_wave', np.bool_),
                  ('s_wave', np.bool_),
                  ('allow_inner_core_s', np.bool_),
                  ('max_delta_p', np.float_),
                  ('max_depth_interval', np.float_),
                  ('max_interp_error', np.float_),
                  ('max_range_interval', np.float_),
                  ('min_delta_p', np.float_),
                  ('radius_of_planet', np.float_),
                  ('slowness_tolerance', np.float_)]
        slowness_model = np.empty(shape=(), dtype=dtypes)
        for dtype in dtypes:
            key = dtype[0]
            slowness_model[key] = getattr(self.s_mod, key)
        arrays['s_mod'] = slowness_model

        # d) handle complex contents of .s_mod
        arrays['s_mod.p_layers'] = self.s_mod.p_layers
        arrays['s_mod.s_layers'] = self.s_mod.s_layers
        arrays['s_mod.critical_depths'] = self.s_mod.critical_depths
        for key in ['fluid_layer_depths', 'high_slowness_layer_depths_p',
                    'high_slowness_layer_depths_s']:
            data = getattr(self.s_mod, key)
            if len(data) == 0:
                arr_ = np.array([])
            else:
                arr_ = np.vstack([data_._to_array() for data_ in data])
            arrays['s_mod.' + key] = arr_

        # e) handle .s_mod.v_mod
        dtypes = [('cmb_depth', np.float_),
                  ('iocb_depth', np.float_),
                  ('is_spherical', np.bool_),
                  ('max_radius', np.float_),
                  ('min_radius', np.int_),
                  ('model_name', np.str_,
                   len(self.s_mod.v_mod.model_name)),
                  ('moho_depth', np.float_),
                  ('radius_of_planet', np.float_)]
        velocity_model = np.empty(shape=(), dtype=dtypes)
        for dtype in dtypes:
            key = dtype[0]
            velocity_model[key] = getattr(self.s_mod.v_mod, key)
        arrays['v_mod'] = velocity_model
        arrays['v_mod.layers'] = self.s_mod.v_mod.layers

        # finally save the collection of (structured) arrays to a binary file
        np.savez_compressed(filename, **arrays)

    @staticmethod
    def deserialize(filename, cache=None):
        """
        Deserialize model from numpy npz binary file.
        """
        # XXX: Make this a with statement when old NumPy support is dropped.
        npz = np.load(filename)
        try:
            model = TauModel(s_mod=None,
                             radius_of_planet=float(npz["radius_of_planet"]),
                             cache=cache, skip_calc=True)
            complex_contents = [
                'tau_branches', 's_mod', 'v_mod',
                's_mod.p_layers', 's_mod.s_layers', 's_mod.critical_depths',
                's_mod.fluid_layer_depths',
                's_mod.high_slowness_layer_depths_p',
                's_mod.high_slowness_layer_depths_s', 'v_mod.layers']

            # a) handle simple contents
            for key in npz.keys():
                # we have multiple, dynamic key names for individual tau
                # branches now, skip them all
                if key in complex_contents or key.startswith('tau_branches'):
                    continue
                arr = npz[key]
                if arr.ndim == 0:
                    arr = arr[()]
                setattr(model, key, arr)

            # b) handle .tau_branches
            tau_branch_keys = [key for key in npz.keys()
                               if key.startswith('tau_branches_')]
            j, i = tau_branch_keys[0].split("__")[1:]
            i = int(i.split("/")[1])
            j = int(j.split("/")[1])
            branches = np.empty(shape=(i, j), dtype=np.object_)
            for key in tau_branch_keys:
                j_, i_ = key.split("__")[1:]
                i_ = int(i_.split("/")[0])
                j_ = int(j_.split("/")[0])
                branches[i_][j_] = TauBranch._from_array(npz[key])
            # no idea how numpy lays out empty arrays of object type,
            # make a copy just in case..
            branches = np.copy(branches)
            setattr(model, "tau_branches", branches)

            # c) handle simple contents of .s_mod
            slowness_model = SlownessModel(v_mod=None,
                                           skip_model_creation=True)
            setattr(model, "s_mod", slowness_model)
            for key in npz['s_mod'].dtype.names:
                # restore scalar types from 0d array
                arr = npz['s_mod'][key]
                if arr.ndim == 0:
                    arr = arr.flatten()[0]
                setattr(slowness_model, key, arr)

            # d) handle complex contents of .s_mod
            for key in ['p_layers', 's_layers', 'critical_depths']:
                setattr(slowness_model, key, npz['s_mod.' + key])
            for key in ['fluid_layer_depths', 'high_slowness_layer_depths_p',
                        'high_slowness_layer_depths_s']:
                arr_ = npz['s_mod.' + key]
                if len(arr_) == 0:
                    data = []
                else:
                    data = [DepthRange._from_array(x) for x in arr_]
                setattr(slowness_model, key, data)

            # e) handle .s_mod.v_mod
            velocity_model = VelocityModel(
                model_name=npz["v_mod"]["model_name"],
                radius_of_planet=float(npz["v_mod"]["radius_of_planet"]),
                min_radius=float(npz["v_mod"]["min_radius"]),
                max_radius=float(npz["v_mod"]["max_radius"]),
                moho_depth=float(npz["v_mod"]["moho_depth"]),
                cmb_depth=float(npz["v_mod"]["cmb_depth"]),
                iocb_depth=float(npz["v_mod"]["iocb_depth"]),
                is_spherical=bool(npz["v_mod"]["is_spherical"]),
                layers=None
            )
            setattr(slowness_model, "v_mod", velocity_model)
            setattr(velocity_model, 'layers', npz['v_mod.layers'])
        finally:
            if hasattr(npz, 'close'):
                npz.close()
            else:
                del npz
        return model

    @staticmethod
    def from_file(model_name, cache=None):
        if os.path.exists(model_name):
            filename = model_name
        else:
            filename = os.path.join(os.path.dirname(__file__), "data",
                                    model_name.lower() + ".npz")
        return TauModel.deserialize(filename, cache=cache)
