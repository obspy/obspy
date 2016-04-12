#!/usr/bin/env python
"""
Module for handling array analysis.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import copy
import math
import tempfile
import os
import shutil
import numpy as np
import scipy as sp
from scipy import interpolate
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from obspy.core import UTCDateTime
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.signal.util import util_geo_km, next_pow_2
from obspy.core import Trace
from obspy.core.inventory import Inventory
from scipy.integrate import cumtrapz
from obspy.core import Stream
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper
from obspy.core.util import AttribDict
import warnings
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


class SeismicArray(object):
    """
    Class representing a seismic array.

    The SeismicArray class is a named container for an
    :class:`~obspy.core.inventory.Inventory` containing the components
    making up the array along with methods for array processing. It does not
    contain any seismic data. The locations of the array components
    (stations or channels) must be set in the respective objects (for an
    overview of the inventory system, see :mod:`~obspy.station`). While the
    inventory must be composed of :class:`~obspy.station.station.Station`
    and/or :class:`~obspy.station.channel.Channel` objects, they do not need
    to represent actual seismic stations; their only required attribute is
    the location information.

    :param name: Array name.
    :type name: str
    :param inventory: Inventory of stations making up the array.
    :type inventory: :class:`~obspy.core.inventory.Inventory`

    .. rubric:: Basic Usage

    >>> from obspy.core.inventory import read_inventory
    >>> inv = read_inventory('http://examples.obspy.org/agfainventory.xml')
    >>> array = SeismicArray('AGFA', inv)
    >>> print(array)
    Seismic Array 'AGFA' with 5 Stations, aperture: 0.06 km.


    .. rubric:: Coordinate conventions:
    * Right handed
    * X positive to east
    * Y positive to north
    * Z positive up

    """

    def __init__(self, name, inventory):
        self.name = name
        self.inventory = inventory
        if not isinstance(inventory, Inventory):
            raise TypeError("Inventory must be an ObsPy Inventory.")
        # Must use deepcopy, otherwise even temporary changes to the array
        # inventory will affect the 'original' inventory.
        self.inventory = copy.deepcopy(inventory)

    def __str__(self):
        """
        Pretty representation of the array.
        """
        if self.inventory is None:
            return "Empty seismic array '{}'".format(self.name)
        ret_str = "Seismic Array '{name}' with ".format(name=self.name)
        ret_str += "{count} Stations, ".format(count=len(self.geometry))
        ret_str += "aperture: {aperture:.2f} km.".format(
            aperture=self.aperture)
        return ret_str

    def inventory_cull(self, st):
        """
        Shrink array inventory to channels present in the given stream.

        Permanently remove from the array inventory all entries for stations or
        channels that do not have traces in the given
        :class:`~obspy.core.stream.Stream` st. This may be useful e.g. if data
        is not consistently available from every channel in an array, or if a
        web service has returned many more inventory objects than required. The
        method selects channels based on matching network, station, location
        and channel codes to the ones given in the trace headers. Furthermore,
        if a time range is specified for a channel, it will only be kept if it
        matches the time span of its corresponding trace.

        If you wish to keep the original inventory, make a copy first:

        >>> from copy import deepcopy #doctest: +SKIP
        >>> original_inventory = deepcopy(array.inventory) #doctest: +SKIP

        :param st: :class:`~obspy.core.stream.Stream` to which the array
            inventory should correspond.
        """
        inv = self.inventory
        # check what station/channel IDs are in the data
        stations_present = list(set(tr.id for tr in st))
        # delete all channels that are not represented
        for k, netw in reversed(list(enumerate(inv.networks))):
            for j, stn in reversed(list(enumerate(netw.stations))):
                for i, cha in reversed(list(enumerate(stn.channels))):
                    if ("{}.{}.{}.{}".format(netw.code, stn.code,
                                             cha.location_code, cha.code)
                       not in stations_present):
                        del stn.channels[i]
                        continue
                        # Also remove if it doesn't cover the time of the
                        # trace:
                    for tr in st.select(network=netw.code, station=stn.code,
                                        location=cha.location_code,
                                        channel=cha.code):
                        if not cha.is_active(starttime=tr.stats.starttime,
                                             endtime=tr.stats.endtime):
                            del stn.channels[i]
                stn.total_number_of_channels = len(stn.channels)
                # no point keeping stations with all channels removed
                if len(stn.channels) == 0:
                    del netw.stations[j]
            # no point keeping networks with no stations in them:
            if len(netw.stations) == 0:
                del inv.networks[k]
        # check total number of channels now:
        contents = inv.get_contents()
        if len(contents['channels']) < len(stations_present):
            # Inventory is altered anyway in this case.
            warnings.warn('Inventory does not contain information for all '
                          'traces in stream.')
        self.inventory = inv

    def plot(self):
        """
        Plot the array's station/channel locations as well as it's geometric
        and centre of gravity.
        """
        import matplotlib.pyplot as plt
        if self.inventory:
            self.inventory.plot(projection="local", show=False)
            bmap = plt.gca().basemap

            grav = self.center_of_gravity
            x, y = bmap(grav["longitude"], grav["latitude"])
            bmap.scatter(x, y, marker="x", c="red", s=40, zorder=20)
            plt.text(x, y, "Center of Gravity", color="red")

            geo = self.geometrical_center
            x, y = bmap(geo["longitude"], geo["latitude"])
            bmap.scatter(x, y, marker="x", c="green", s=40, zorder=20)
            plt.text(x, y, "Geometrical Center", color="green")

            plt.show()

    def _get_geometry(self):
        """
        Return a dictionary of latitude, longitude and absolute height
        [km] for each component in the array inventory.

        For every component in the array inventory (channels if available,
        stations otherwise), a SEED ID string with the format
        'network.station.location.channel', leaving any unknown parts blank, is
        assembled. This is one key for the returned dictionary, while the value
        is a dictionary of the component's coordinates.

        :return: A dictionary with keys: SEED IDs and values: dictionaries of
            'latitude', 'longitude' and 'absolute_height_in_km'.
        """
        if not self.inventory:
            return {}
        geo = {}

        # Using core.inventory.inventory.Inventory.get_coordinates() is not
        # really satisfactory: It doesn't return coordinates for inventories
        # that have stations but no channels defined.
        # Might be the case e.g. if using the array class for inventory of
        # sources.
        for network in self.inventory:
            for station in network:
                if len(station.channels) == 0:
                    # Using the full Seed ID string allows retrieving
                    # coordinates with geometry[trace.id] for other methods.
                    item_code = "{n}.{s}..".format(n=network.code,
                                                   s=station.code)
                    this_coordinates = \
                        {"latitude": float(station.latitude),
                         "longitude": float(station.longitude),
                         "absolute_height_in_km":
                         float(station.elevation) / 1000.0}
                    geo[item_code] = this_coordinates
                else:
                    for channel in station:
                        item_code = "{}.{}.{}.{}".format(network.code,
                                                         station.code,
                                                         channel.location_code,
                                                         channel.code)
                        this_coordinates = \
                            {"latitude": float(channel.latitude),
                             "longitude": float(channel.longitude),
                             "absolute_height_in_km":
                             float(channel.elevation - channel.depth) / 1000.0}
                        geo[item_code] = this_coordinates
        return geo

    @property
    def geometrical_center(self):
        """
        Return the geometrical centre as dictionary.

        The geometrical centre is the mid-point of the maximum array extent in
        each direction.
        """
        extent = self.extent
        return {
            "latitude": (extent["max_latitude"] +
                         extent["min_latitude"]) / 2.0,
            "longitude": (extent["max_longitude"] +
                          extent["min_longitude"]) / 2.0,
            "absolute_height_in_km":
            (extent["min_absolute_height_in_km"] +
             extent["max_absolute_height_in_km"]) / 2.0
        }

    @property
    def center_of_gravity(self):
        """
        Return the centre of gravity as a dictionary.

        The centre of gravity is calculated as the mean of the array stations'
        locations in each direction.
        """
        lats, lngs, hgts = self._coordinate_values()
        return {
            "latitude": np.mean(lats),
            "longitude": np.mean(lngs),
            "absolute_height_in_km": np.mean(hgts)}

    @property
    def geometry(self):
        """
        A dictionary of latitude, longitude and absolute height [km] values
        for each item in the array inventory.

        For every component in the array inventory (channels if available,
        stations otherwise), a SEED ID string with the format
        'network.station.location.channel', leaving any unknown parts blank, is
        assembled. This is one key for the returned dictionary, while the value
        is a dictionary of the component's coordinates.

        :return A dictionary with keys: SEED IDs and values: dictionaries of
            'latitude', 'longitude' and 'absolute_height_in_km'.
        """
        return self._get_geometry()

    @property
    def aperture(self):
        """
        Return the array aperture in kilometers.

        The array aperture is the maximum distance between any two stations in
        the array.
        """
        distances = []
        geo = self.geometry
        for location, coordinates in geo.items():
            for other_location, other_coordinates in list(geo.items()):
                if location == other_location:
                    continue
                distances.append(gps2dist_azimuth(
                    coordinates["latitude"], coordinates["longitude"],
                    other_coordinates["latitude"],
                    other_coordinates["longitude"])[0] / 1000.0)

        return max(distances)

    @property
    def extent(self):
        """
        Dictionary of the array's minimum and maximum lat/long and elevation
        values.
        """
        lats, lngs, hgt = self._coordinate_values()

        return {
            "min_latitude": min(lats),
            "max_latitude": max(lats),
            "min_longitude": min(lngs),
            "max_longitude": max(lngs),
            "min_absolute_height_in_km": min(hgt),
            "max_absolute_height_in_km": max(hgt)}

    def _coordinate_values(self):
        """
        Return the array geometry as simple lists of lat, long and elevation.
        """
        geo = self.geometry
        lats, lngs, hgt = [], [], []
        for coordinates in list(geo.values()):
            lats.append(coordinates["latitude"]),
            lngs.append(coordinates["longitude"]),
            hgt.append(coordinates["absolute_height_in_km"])
        return lats, lngs, hgt

    def get_geometry_xyz(self, latitude, longitude, absolute_height_in_km,
                         correct_3dplane=False):
        """
        Return the array geometry as each components's offset
        relative to a given reference point, in km.

        The returned geometry is a nested dictionary with each component's SEED
        ID as key and a dictionary of its coordinates as value, similar to that
        given by :attr:`~obspy.signal.array_analysis.SeismicArray.geometry`,
        but with different coordinate values and keys.

        To obtain the x-y-z geometry in relation to, for example, the center of
        gravity, use:

        >>> array = SeismicArray('', inv) # doctest: +SKIP
        >>> array.get_geometry_xyz(**array.center_of_gravity) # doctest: +SKIP

        :param latitude: Latitude of reference origin.
        :param longitude: Longitude of reference origin.
        :param absolute_height_in_km: Elevation of reference origin.
        :param correct_3dplane: Correct the returned geometry by a
            best-fitting 3D plane.
            This might be important if the array is located on an inclined
            slope.
        :return: The geometry of the components as dictionary, with coordinate
            keys of 'x', 'y' and 'z'.
        """
        geometry = {}
        for key, value in list(self.geometry.items()):
            x, y = util_geo_km(longitude, latitude, value["longitude"],
                               value["latitude"])
            geometry[key] = {
                "x": x,
                "y": y,
                "z": absolute_height_in_km - value["absolute_height_in_km"]
            }
        if correct_3dplane:
            self.correct_with_3dplane(geometry)
        return geometry

    def find_closest_station(self, latitude, longitude,
                             absolute_height_in_km=0.0):
        """
        Return the array station closest to a given reference point.

        :param latitude: Latitude of reference origin.
        :param longitude: Longitude of reference origin.
        :param absolute_height_in_km: Elevation of reference origin.
        """
        min_distance = None
        min_distance_station = None

        for key, value in list(self.geometry.items()):
            # output in [km]
            x, y = util_geo_km(longitude, latitude, value["longitude"],
                               value["latitude"])
            x *= 1000.0
            y *= 1000.0

            z = absolute_height_in_km

            distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_distance_station = key

        return min_distance_station

    def get_timeshift_baz(self, sll, slm, sls, baz, latitude=None,
                          longitude=None, absolute_height=None,
                          static3d=False, vel_cor=4.0):
        """
        Returns timeshift table for the geometry of the current array, in
        kilometres relative to a given centre (uses geometric centre if not
        specified), and a pre-defined backazimuth. Returns nested dict of
        timeshifts at each slowness between sll and slm, with sls increment.

        :param sll: slowness x min (lower)
        :param slm: slowness x max (lower)
        :param sls: slowness step
        :param baz:  backazimuth applied
        :param latitude: latitude of reference origin
        :param longitude: longitude of reference origin
        :param absolute_height: elevation of reference origin, in km
        :param vel_cor: correction velocity (upper layer) in km/s. May be given
            at each station as a dictionary with the station/channel IDs as
            keys (same as in self.geometry).
        :param static3d: a correction of the station height is applied using
            vel_cor the correction is done according to the formula:
            t = rxy*s - rz*cos(inc)/vel_cor
            where inc is defined by inv = asin(vel_cor*slow)
        """
        if any([_i is None for _i in [latitude, longitude, absolute_height]]):
            latitude = self.geometrical_center["latitude"]
            longitude = self.geometrical_center["longitude"]
            absolute_height = self.geometrical_center["absolute_height_in_km"]
        geom = self.get_geometry_xyz(latitude, longitude,
                                     absolute_height)

        baz = math.pi * baz / 180.0

        time_shift_tbl = {}
        sx = sll
        while sx < slm:
            try:
                inc = math.asin(vel_cor * sx)
            except ValueError:
                # if vel_cor given as dict:
                inc = np.pi / 2.0

            time_shifts = {}
            for key, value in list(geom.items()):
                time_shifts[key] = sx * (value["x"] * math.sin(baz) +
                                         value["y"] * math.cos(baz))

                if static3d:
                    try:
                        v = vel_cor[key]
                    except TypeError:
                        # if vel_cor is a constant:
                        v = vel_cor
                    time_shifts[key] += value["z"] * math.cos(inc) / v

                time_shift_tbl[sx] = time_shifts
            sx += sls

        return time_shift_tbl

    def get_timeshift(self, sllx, slly, sls, grdpts_x, grdpts_y,
                      latitude=None, longitude=None, absolute_height=None,
                      vel_cor=4., static3d=False):
        """
        Returns timeshift table for the geometry of the current array, in
        kilometres relative to a given centre (uses geometric centre if not
        specified).

        :param sllx: slowness x min (lower)
        :param slly: slowness y min (lower)
        :param sls: slowness step
        :param grdpts_x: number of grid points in x direction
        :param grdpts_y: number of grid points in y direction
        :param latitude: latitude of reference origin
        :param longitude: longitude of reference origin
        :param absolute_height: elevation of reference origin, in km
        :param vel_cor: correction velocity (upper layer) in km/s
        :param static3d: a correction of the station height is applied using
            vel_cor the correction is done according to the formula:
            t = rxy*s - rz*cos(inc)/vel_cor
            where inc is defined by inv = asin(vel_cor*slow)
        """
        if any([_i is None for _i in [latitude, longitude, absolute_height]]):
            latitude = self.geometrical_center["latitude"]
            longitude = self.geometrical_center["longitude"]
            absolute_height = self.geometrical_center["absolute_height_in_km"]
        geom = self.get_geometry_xyz(latitude, longitude,
                                     absolute_height)

        geometry = self._geometry_dict_to_array(geom)

        if static3d:
            nstat = len(geometry)
            time_shift_tbl = np.empty((nstat, grdpts_x, grdpts_y),
                                      dtype="float32")
            for k in range(grdpts_x):
                sx = sllx + k * sls
                for l in range(grdpts_y):
                    sy = slly + l * sls
                    slow = np.sqrt(sx * sx + sy * sy)
                    if vel_cor * slow <= 1.:
                        inc = np.arcsin(vel_cor * slow)
                    else:
                        warnings.warn(
                            "Correction velocity smaller than apparent"
                            " velocity")
                        inc = np.pi / 2.
                    time_shift_tbl[:, k, l] = sx * geometry[:, 0] + sy * \
                        geometry[:, 1] + geometry[:, 2] * np.cos(inc) / vel_cor
            return time_shift_tbl
        # optimized version
        else:
            mx = np.outer(geometry[:, 0], sllx + np.arange(grdpts_x) * sls)
            my = np.outer(geometry[:, 1], slly + np.arange(grdpts_y) * sls)
            return np.require(
                mx[:, :, np.newaxis].repeat(grdpts_y, axis=2) +
                my[:, np.newaxis, :].repeat(grdpts_x, axis=1),
                dtype='float32')

    def vespagram(self, stream, event_or_baz, sll, slm, sls, starttime,
                  endtime, latitude=None, longitude=None,
                  absolute_height_in_km=None, method="DLS", nthroot=1,
                  static3d=False, vel_cor=4.0):
        baz = float(event_or_baz)
        if(latitude is None or longitude is None or
           absolute_height_in_km is None):
            time_shift_table = self.get_timeshift_baz(sll, slm, sls, baz,
                                                      static3d=static3d,
                                                      vel_cor=vel_cor)
        else:
            time_shift_table = self.get_timeshift_baz(sll, slm, sls, baz,
                                                      latitude, longitude,
                                                      absolute_height_in_km,
                                                      static3d=static3d,
                                                      vel_cor=vel_cor)

        return self._vespagram_baz(stream, time_shift_table,
                                   starttime=starttime,
                                   endtime=endtime, method=method,
                                   nthroot=nthroot)

    def derive_rotation_from_array(self, stream, vp, vs, sigmau, latitude,
                                   longitude, absolute_height_in_km=0.0):
        # todo: what does this do??
        geo = self.geometry

        components = collections.defaultdict(list)
        for tr in stream:
            components[tr.stats.channel[-1].upper()].append(tr)

        # Sanity checks.
        if sorted(components.keys()) != ["E", "N", "Z"]:
            raise ValueError("Three components necessary.")

        for value in list(components.values()):
            value.sort(key=lambda x: "%s.%s" % (x.stats.network,
                                                x.stats.station))

        ids = [tuple([_i.id[:-1] for _i in traces]) for traces in
               list(components.values())]
        if len(set(ids)) != 1:
            raise ValueError("All stations need to have three components.")

        stats = [[(_i.stats.starttime.timestamp, _i.stats.npts,
                   _i.stats.sampling_rate)
                  for _i in traces] for traces in list(components.values())]
        s = []
        for st in stats:
            s.extend(st)

        if len(set(s)) != 1:
            raise ValueError("starttime, npts, and sampling rate must be "
                             "identical for all traces.")

        stations = ["%s.%s" % (_i.stats.network, _i.stats.station)
                    for _i in list(components.values())[0]]
        for station in stations:
            if station not in geo:
                raise ValueError("No coordinates known for station '%s'" %
                                 station)

        array_coords = np.ndarray(shape=(len(geo), 3))
        for _i, tr in enumerate(list(components.values())[0]):
            station = "%s.%s" % (tr.stats.network, tr.stats.station)
            # todo: This is the same as self.geometry_xyz, isn't it?

            x, y = util_geo_km(longitude, latitude,
                               geo[station]["longitude"],
                               geo[station]["latitude"])
            z = absolute_height_in_km
            array_coords[_i][0] = x * 1000.0
            array_coords[_i][1] = y * 1000.0
            array_coords[_i][2] = z * 1000.0

        subarray = np.arange(len(geo))

        tr = []
        for _i, component in enumerate(["Z", "N", "E"]):
            comp = components[component]
            tr.append(np.empty((len(comp[0]), len(comp))))
            for _j, trace in enumerate(comp):
                tr[_i][:, _j][:] = np.require(trace.data, np.float64)

        sp = self.array_rotation_strain(subarray, tr[0], tr[1], tr[2], vp=vp,
                                        vs=vs, array_coords=array_coords,
                                        sigmau=sigmau)

        d1 = sp.pop("ts_w1")
        d2 = sp.pop("ts_w2")
        d3 = sp.pop("ts_w3")

        header = {"network": "XX", "station": "YY", "location": "99",
                  "starttime": list(components.values())[0][0].stats.starttime,
                  "sampling_rate":
                  list(components.values())[0][0].stats.sampling_rate,
                  "channel": "ROZ",
                  "npts": len(d1)}

        tr1 = Trace(data=d1, header=copy.copy(header))
        header["channel"] = "RON"
        header["npts"] = len(d2)
        tr2 = Trace(data=d2, header=copy.copy(header))
        header["channel"] = "ROE"
        header["npts"] = len(d3)
        tr3 = Trace(data=d3, header=copy.copy(header))

        return Stream(traces=[tr1, tr2, tr3]), sp

    def slowness_whitened_power(self, stream, frqlow, frqhigh,
                                prefilter=True, plots=(),
                                static3d=False, vel_corr=4.8, wlen=-1,
                                slx=(-10, 10), sly=(-10, 10), sls=0.5):
        """
        Slowness whitened power analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param prefilter: Whether to bandpass data to selected frequency range
        :type prefilter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3d: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3d: bool
        :param vel_corr: Correction velocity for static topography correction
         in km/s.
        :type vel_corr: float
        :param wlen: sliding window for analysis in seconds, use -1 to use the
         whole trace without windowing.
        :type wlen: float
        :param slx: Min/Max slowness for analysis in x direction [s/km].
        :type slx: (float, float)
        :param sly: Min/Max slowness for analysis in y direction [s/km].
        :type sly: (float, float)
        :param sls: step width of slowness grid [s/km].
        :type sls: float
        :param plots: List or tuple of desired plots that should be plotted for
         each beamforming window.
         Supported options:
         "baz_slow_map" for backazimuth-slowness maps for each window,
         "slowness_xy" for slowness_xy maps for each window.
         Further plotting otions are attached to the returned object.
        :rtype: :class:`~obspy.signal.array_analysis.BeamformerResult`
        """
        return self._array_analysis_helper(stream=stream, method="SWP",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           prefilter=prefilter, plots=plots,
                                           static3d=static3d,
                                           vel_corr=vel_corr, wlen=wlen,
                                           slx=slx, sly=sly, sls=sls)

    def phase_weighted_stack(self, stream, frqlow, frqhigh,
                             prefilter=True, plots=(),
                             static3d=False,
                             vel_corr=4.8, wlen=-1, slx=(-10, 10),
                             sly=(-10, 10), sls=0.5):
        """
        Phase weighted stack analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param prefilter: Whether to bandpass data to selected frequency range
        :type prefilter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3d: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3d: bool
        :param vel_corr: Correction velocity for static topography correction
         in km/s.
        :type vel_corr: float
        :param wlen: sliding window for analysis in seconds, use -1 to use the
         whole trace without windowing.
        :type wlen: float
        :param slx: Min/Max slowness for analysis in x direction [s/km].
        :type slx: (float, float)
        :param sly: Min/Max slowness for analysis in y direction [s/km].
        :type sly: (float, float)
        :param sls: step width of slowness grid [s/km].
        :type sls: float
        :param plots: List or tuple of desired plots that should be plotted for
         each beamforming window.
         Supported options:
         "baz_slow_map" for backazimuth-slowness maps for each window,
         "slowness_xy" for slowness_xy maps for each window.
         Further plotting otions are attached to the returned object.
        :rtype: :class:`~obspy.signal.array_analysis.BeamformerResult`
        """
        return self._array_analysis_helper(stream=stream, method="PWS",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           prefilter=prefilter, plots=plots,
                                           static3d=static3d,
                                           vel_corr=vel_corr, wlen=wlen,
                                           slx=slx, sly=sly, sls=sls)

    def delay_and_sum(self, stream, frqlow, frqhigh,
                      prefilter=True, plots=(), static3d=False,
                      vel_corr=4.8, wlen=-1, slx=(-10, 10),
                      sly=(-10, 10), sls=0.5):
        """
        Delay and sum analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param prefilter: Whether to bandpass data to selected frequency range
        :type prefilter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3d: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3d: bool
        :param vel_corr: Correction velocity for static topography correction
         in km/s.
        :type vel_corr: float
        :param wlen: sliding window for analysis in seconds, use -1 to use the
         whole trace without windowing.
        :type wlen: float
        :param slx: Min/Max slowness for analysis in x direction [s/km].
        :type slx: (float, float)
        :param sly: Min/Max slowness for analysis in y direction [s/km].
        :type sly: (float, float)
        :param sls: step width of slowness grid [s/km].
        :type sls: float
        :param plots: List or tuple of desired plots that should be plotted for
         each beamforming window.
         Supported options:
         "baz_slow_map" for backazimuth-slowness maps for each window,
         "slowness_xy" for slowness_xy maps for each window.
         Further plotting otions are attached to the returned object.
        :rtype: :class:`~obspy.signal.array_analysis.BeamformerResult`
        """
        return self._array_analysis_helper(stream=stream, method="DLS",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           prefilter=prefilter, plots=plots,
                                           static3d=static3d,
                                           vel_corr=vel_corr, wlen=wlen,
                                           slx=slx, sly=sly, sls=sls)

    def fk_analysis(self, stream, frqlow, frqhigh,
                    prefilter=True, plots=(),
                    static3d=False, vel_corr=4.8, wlen=-1, wfrac=0.8,
                    slx=(-10, 10), sly=(-10, 10), sls=0.5):
        """
        FK analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param prefilter: Whether to bandpass data to selected frequency range
        :type prefilter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3d: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3d: bool
        :param vel_corr: Correction velocity for static topography correction
        in km/s.
        :type vel_corr: float
        :param wlen: sliding window for analysis in seconds, use -1 to use the
         whole trace without windowing.
        :type wlen: float
        :param wfrac: fraction of sliding window to use for step.
        :type wfrac: float
        :param slx: Min/Max slowness for analysis in x direction [s/km].
        :type slx: (float, float)
        :param sly: Min/Max slowness for analysis in y direction [s/km].
        :type sly: (float, float)
        :param sls: step width of slowness grid [s/km].
        :type sls: float
        :param plots: List or tuple of desired plots that should be plotted for
         each beamforming window.
         Supported options:
         "baz_slow_map" for backazimuth-slowness maps for each window,
         "slowness_xy" for slowness_xy maps for each window.
         Further plotting otions are attached to the returned object.
        :rtype: :class:`~obspy.signal.array_analysis.BeamformerResult`

        """
        return self._array_analysis_helper(stream=stream, method="FK",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           prefilter=prefilter, plots=plots,
                                           static3d=static3d,
                                           vel_corr=vel_corr,
                                           wlen=wlen, wfrac=wfrac,
                                           slx=slx, sly=sly, sls=sls)

    def _array_analysis_helper(self, stream, method, frqlow, frqhigh,
                               prefilter=True, static3d=False,
                               vel_corr=4.8, wlen=-1, wfrac=0.8, slx=(-10, 10),
                               sly=(-10, 10), sls=0.5,
                               plots=()):
        """
        Array analysis wrapper routine.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param method: Method used for the array analysis
            (one of "FK": Frequency Wavenumber, "DLS": Delay and Sum,
            "PWS": Phase Weighted Stack, "SWP": Slowness Whitened Power).
        :type method: str
        :param prefilter: Whether to bandpass data to selected frequency range
        :type prefilter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3d: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3d: bool
        :param vel_corr: Correction velocity for static topography correction
         in km/s.
        :type vel_corr: float
        :param wlen: sliding window for analysis in seconds, use -1 to use the
         whole trace without windowing.
        :type wlen: float
        :param wfrac: fraction of sliding window to use for step.
        :type wfrac: float
        :param slx: Min/Max slowness for analysis in x direction [s/km].
        :type slx: (float, float)
        :param sly: Min/Max slowness for analysis in y direction [s/km].
        :type sly: (float, float)
        :param sls: step width of slowness grid [s/km].
        :type sls: float
        :param plots: List or tuple of desired plots that should be plotted for
         each beamforming window.
         Supported options:
         "baz_slow_map" for backazimuth-slowness maps for each window,
         "slowness_xy" for slowness_xy maps for each window.
         Further plotting otions are attached to the returned object.
        :rtype: :class:`~obspy.signal.array_analysis.BeamformerResult`
        """
        import matplotlib.pyplot as plt
        if method not in ("FK", "DLS", "PWS", "SWP"):
            raise ValueError("Invalid method: ''" % method)

        if "baz_slow_map" in plots:
            make_slow_map = True
        else:
            make_slow_map = False
        if "slowness_xy" in plots:
            make_slowness_xy = True
        else:
            make_slowness_xy = False

        sllx, slmx = slx
        slly, slmy = sly
        # In terms of a single 'radial' slowness (used e.g. for plotting):
        if sllx < 0 or slly < 0:
            sll = 0
        else:
            sll = np.sqrt(sllx ** 2 + slly ** 2)
        slm = np.sqrt(slmx ** 2 + slmy ** 2)

        # Do not modify the given stream in place.
        st_workon = stream.copy()
        # Trim the stream so all traces are present.
        starttime = max([tr.stats.starttime for tr in st_workon])
        endtime = min([tr.stats.endtime for tr in st_workon])
        st_workon.trim(starttime, endtime)

        self._attach_coords_to_stream(st_workon)

        if prefilter:
            st_workon.filter('bandpass', freqmin=frqlow, freqmax=frqhigh,
                             zerophase=True)
        else:
            if frqlow is not None or frqhigh is not None:
                warnings.warn("No filtering done. Param 'prefilter' is False.")
        # Making the map plots is efficiently done by saving the power maps to
        # a temporary directory.
        tmpdir = tempfile.mkdtemp(prefix="obspy-")
        filename_patterns = (os.path.join(tmpdir, 'pow_map_%03d.npy'),
                             os.path.join(tmpdir, 'apow_map_%03d.npy'))
        if make_slow_map or make_slowness_xy:
            def dump(pow_map, apow_map, i):
                np.save(filename_patterns[0] % i, pow_map)
                np.save(filename_patterns[1] % i, apow_map)

        else:
            dump = None

        # Temporarily trim self.inventory so only stations/channels which are
        # actually represented in the traces are kept in the inventory.
        # Otherwise self.geometry and the xyz geometry arrays will have more
        # entries than the stream.
        # Todo: if inventory_cull becomes an inventory method, change this to
        # inv_workon = copy.deepcopy(self.inventory)
        # and use that instead to avoid the try ... finally.
        invbkp = copy.deepcopy(self.inventory)
        self.inventory_cull(st_workon)
        try:
            if method == 'FK':
                kwargs = dict(
                    # slowness grid: X min, X max, Y min, Y max, Slow Step
                    sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                    # sliding window properties
                    win_len=wlen, win_frac=wfrac,
                    # frequency properties
                    frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
                    # restrict output
                    store=dump,
                    semb_thres=-1e9, vel_thres=-1e9, verbose=False,
                    # use mlabday to be compatible with matplotlib
                    timestamp='julsec', stime=starttime, etime=endtime,
                    method=0, correct_3dplane=False, vel_cor=vel_corr,
                    static3d=static3d)

                # here we do the array processing
                start = UTCDateTime()
                outarr = self._covariance_array_processing(st_workon, **kwargs)
                print("Total time in routine: %f\n" % (UTCDateTime() - start))
                t, rel_power, abs_power, baz, slow = outarr.T

            else:
                kwargs = dict(
                    # slowness grid: X min, X max, Y min, Y max, Slow Step
                    sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                    # sliding window properties
                    # frequency properties
                    frqlow=frqlow, frqhigh=frqhigh,
                    # restrict output
                    store=dump,
                    win_len=wlen, win_frac=0.5,
                    nthroot=4, method=method,
                    verbose=False, timestamp='julsec',
                    stime=starttime, etime=endtime, vel_cor=vel_corr,
                    static3d=False)

                # here we do the array processing
                start = UTCDateTime()
                outarr = self._beamforming(st_workon, **kwargs)
                print("Total time in routine: %f\n" % (UTCDateTime() - start))
                t, rel_power, baz, slow_x, slow_y, slow = outarr.T
                abs_power = None

            baz[baz < 0.0] += 360
            if wlen < 0:
                # Need to explicitly specify the timestep of the analysis.
                out = BeamformerResult(inventory=self.inventory,
                                       win_starttimes=t,
                                       slowness_range=np.arange(sll, slm, sls),
                                       max_rel_power=rel_power,
                                       max_abs_power=abs_power,
                                       max_pow_baz=baz, max_pow_slow=slow,
                                       method=method,
                                       timestep=endtime - starttime)
            else:
                out = BeamformerResult(inventory=self.inventory,
                                       win_starttimes=t,
                                       slowness_range=np.arange(sll, slm, sls),
                                       max_rel_power=rel_power,
                                       max_abs_power=abs_power,
                                       max_pow_baz=baz, max_pow_slow=slow,
                                       method=method)

            # now let's do the plotting
            if "baz_slow_map" in plots:
                self._plot_array_analysis(outarr, sllx, slmx, slly, slmy, sls,
                                          filename_patterns, True, method,
                                          st_workon, starttime, wlen, endtime)
            if "slowness_xy" in plots:
                self._plot_array_analysis(outarr, sllx, slmx, slly, slmy, sls,
                                          filename_patterns, False, method,
                                          st_workon, starttime, wlen, endtime)
            plt.show()
            # Return the beamforming results to allow working more on them,
            # make other plots etc.
            return out
        finally:
            self.inventory = invbkp
            shutil.rmtree(tmpdir)

    def _attach_coords_to_stream(self, stream):
        """
        Attaches dictionary with latitude, longitude and elevation to each
        trace in stream as `trace.stats.coords`. Takes into account local
        depth of sensor.
        """
        geo = self.geometry

        for tr in stream:
            coords = geo[tr.id]
            tr.stats.coordinates = \
                AttribDict(dict(latitude=coords["latitude"],
                                longitude=coords["longitude"],
                                elevation=coords["absolute_height_in_km"]))

    def _covariance_array_processing(self, stream, win_len, win_frac, sll_x,
                                     slm_x, sll_y, slm_y, sl_s, semb_thres,
                                     vel_thres, frqlow, frqhigh, stime, etime,
                                     prewhiten, verbose=False,
                                     timestamp='mlabday', method=0,
                                     correct_3dplane=False, vel_cor=4.,
                                     static3d=False, store=None):
        """
        Method for FK-Analysis/Capon

        :param stream: Stream object, the trace.stats dict like class must
            contain an :class:`~obspy.core.util.attribdict.AttribDict` with
            'latitude', 'longitude' (in degrees) and 'elevation' (in km), or
            'x', 'y', 'elevation' (in km) items/attributes, as attached in
            `self._array_analysis_helper`/
        :type win_len: float
        :param win_len: Sliding window length in seconds
        :type win_frac: float
        :param win_frac: Fraction of sliding window to use for step
        :type sll_x: float
        :param sll_x: slowness x min (lower)
        :type slm_x: float
        :param slm_x: slowness x max
        :type sll_y: float
        :param sll_y: slowness y min (lower)
        :type slm_y: float
        :param slm_y: slowness y max
        :type sl_s: float
        :param sl_s: slowness step
        :type semb_thres: float
        :param semb_thres: Threshold for semblance
        :type vel_thres: float
        :param vel_thres: Threshold for velocity
        :type frqlow: float
        :param frqlow: lower frequency for fk/capon
        :type frqhigh: float
        :param frqhigh: higher frequency for fk/capon
        :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param stime: Start time of interest
        :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param etime: End time of interest
        :type prewhiten: int
        :param prewhiten: Do prewhitening, values: 1 or 0
        :type timestamp: str
        :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec'
            returns the timestamp in seconds since 1970-01-01T00:00:00,
            'mlabday' returns the timestamp in days (decimals represent hours,
            minutes and seconds) since '0001-01-01T00:00:00' as needed for
            matplotlib date plotting (see e.g. matplotlib's num2date)
        :type method: int
        :param method: the method to use 0 == bf, 1 == capon
        :param vel_cor: correction velocity (upper layer) in km/s
        :param static3d: a correction of the station height is applied using
            vel_cor the correction is done according to the formula:
            t = rxy*s - rz*cos(inc)/vel_cor
            where inc is defined by inv = asin(vel_cor*slow)
        :type store: function
        :param store: A custom function which gets called on each iteration.
            It is called with the relative power map and the time offset as
            first and second arguments and the iteration number as third
            argument. Useful for storing or plotting the map for each
            iteration.
        :return: :class:`numpy.ndarray` of timestamp, relative relpow, absolute
            relpow, backazimuth, slowness
        """
        res = []
        eotr = True

        # check that sampling rates do not vary
        fs = stream[0].stats.sampling_rate
        if len(stream) != len(stream.select(sampling_rate=fs)):
            msg = ('in array-processing sampling rates of traces in stream are'
                   ' not equal')
            raise ValueError(msg)

        grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
        grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

        if correct_3dplane:
            self.correct_with_3dplane(self.geometry)

        if verbose:
            print("geometry:")
            print(self.geometry)
            print("stream contains following traces:")
            print(stream)
            print("stime = " + str(stime) + ", etime = " + str(etime))

        time_shift_table = self.get_timeshift(sll_x, sll_y, sl_s,
                                              grdpts_x, grdpts_y,
                                              vel_cor=vel_cor,
                                              static3d=static3d)

        spoint, _epoint = self.get_stream_offsets(stream, stime, etime)

        # loop with a sliding window over the dat trace array and apply bbfk
        nstat = len(stream)
        fs = stream[0].stats.sampling_rate
        if win_len < 0.:
            nsamp = int((etime - stime) * fs)
            print(nsamp)
            nstep = 1
        else:
            nsamp = int(win_len * fs)
            nstep = int(nsamp * win_frac)

        # generate plan for rfftr
        nfft = next_pow_2(nsamp)
        deltaf = fs / float(nfft)
        nlow = int(frqlow / float(deltaf) + 0.5)
        nhigh = int(frqhigh / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
        nf = nhigh - nlow + 1  # include upper and lower frequency
        # to speed up the routine a bit we estimate all steering vectors in
        # advance
        steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype=np.complex128)
        clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                             deltaf, time_shift_table, steer)
        r = np.empty((nf, nstat, nstat), dtype=np.complex128)
        ft = np.empty((nstat, nf), dtype=np.complex128)
        newstart = stime
        # 0.22 matches 0.2 of historical C bbfk.c
        tap = cosine_taper(nsamp, p=0.22)
        offset = 0
        count = 0  # iteration of loop
        relpow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
        abspow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
        while eotr:
            try:
                for i, tr in enumerate(stream):
                    dat = tr.data[spoint[i] + offset:
                                  spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
            except IndexError:
                break
            ft = np.ascontiguousarray(ft, np.complex128)
            relpow_map.fill(0.)
            abspow_map.fill(0.)
            # computing the covariances of the signal at different receivers
            dpow = 0.
            for i in range(nstat):
                for j in range(i, nstat):
                    r[:, i, j] = ft[i, :] * ft[j, :].conj()
                    if method == 1:
                        r[:, i, j] /= np.abs(r[:, i, j].sum())
                    if i != j:
                        r[:, j, i] = r[:, i, j].conjugate()
                    else:
                        dpow += np.abs(r[:, i, j].sum())
            dpow *= nstat
            if method == 1:
                # P(f) = 1/(e.H r(f)^-1 e)
                for n in range(nf):
                    r[n, :, :] = np.linalg.pinv(r[n, :, :], rcond=1e-6)

            errcode = clibsignal.generalizedBeamformer(
                relpow_map, abspow_map, steer, r, nstat, prewhiten,
                grdpts_x, grdpts_y, nf, dpow, method)
            if errcode != 0:
                msg = 'generalizedBeamforming exited with error %d'
                raise Exception(msg % errcode)
            ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
            relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
            if store is not None:
                store(relpow_map, abspow_map, count)
            count += 1

            # here we compute baz, slow
            slow_x = sll_x + ix * sl_s
            slow_y = sll_y + iy * sl_s

            slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
            if slow < 1e-8:
                slow = 1e-8
            # Transform from a slowness grid to polar coordinates.
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
            baz = azimut % -360 + 180
            if relpow > semb_thres and 1. / slow > vel_thres:
                if timestamp == 'julsec':
                    outtime = newstart
                elif timestamp == 'mlabday':
                    # 719163 == days between 1970 and 0001 + 1
                    outtime = UTCDateTime(newstart.timestamp /
                                          (24. * 3600) + 719163)
                else:
                    msg = "Option timestamp must be one of 'julsec'," \
                          " or 'mlabday'"
                    raise ValueError(msg)
                res.append(np.array([outtime, relpow, abspow, baz,
                                     slow]))
                if verbose:
                    print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
            if (newstart + (nsamp + nstep) / fs) > etime:
                eotr = False
            offset += nstep

            newstart += nstep / fs
        return np.array(res, dtype=float)

    @staticmethod
    def _three_c_dowhiten(fcoeffz, fcoeffn, fcoeffe, deltaf):
        """
        Amplitude spectra whitening with moving average and window width ww
        and weighting factor: 1/((Z+E+N)/3)
        """
        for nst in range(fcoeffz.shape[0]):
            for nwin in range(fcoeffz.shape[1]):
                ampz = np.abs(fcoeffz[nst, nwin, :])
                ampn = np.abs(fcoeffn[nst, nwin, :])
                ampe = np.abs(fcoeffe[nst, nwin, :])
                # todo: why choose this constant?
                ww = int(round(0.01 / deltaf))
                # Window width must be at least 2:
                if ww == 0:
                    ww = 2
                if ww % 2:
                    ww += 1
                n_freqs = len(ampz)
                csamp = np.zeros((n_freqs, 3), dtype=ampz.dtype)
                csamp[:, 0] = np.cumsum(ampz)
                csamp[:, 1] = np.cumsum(ampe)
                csamp[:, 2] = np.cumsum(ampn)
                ampw = np.zeros(n_freqs, dtype=csamp.dtype)
                for k in range(3):
                    ampw[ww / 2:n_freqs - ww / 2] += (csamp[ww:, k] -
                                                      csamp[:-ww, k]) / ww
                # Fill zero elements at start and end of array with closest
                # non-zero value.
                ampw[n_freqs - ww / 2:] = ampw[n_freqs - ww / 2 - 1]
                ampw[:ww / 2] = ampw[ww / 2]
                ampw *= 1 / 3.
                # Weights are 1/ampw unless ampw is very small, then 0.
                weight = np.where(ampw > np.finfo(np.float).eps * 10.,
                                  1. / (ampw + np.finfo(np.float).eps), 0.)
                fcoeffz[nst, nwin, :] *= weight
                fcoeffe[nst, nwin, :] *= weight
                fcoeffn[nst, nwin, :] *= weight
        return fcoeffz, fcoeffn, fcoeffe

    def _three_c_do_bf(self, stream_n, stream_e, stream_z, win_len, win_frac,
                       u, sub_freq_range, n_min_stns, polarisation,
                       whiten, coherency, win_average,
                       datalen_sec, uindex, verbose=False):
        # backazimuth range to search
        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.

        # Number of stations should be the same as the number of traces,
        # given the checks in the calling method.
        n_stats = len(stream_n.traces)
        npts = stream_n[0].stats.npts

        geo_array = self._geometry_dict_to_array(
            self.get_geometry_xyz(**self.center_of_gravity))
        # NB at this point these offset arrays will contain three times as many
        # entries as needed because each channel is listed individually. These
        # are cut later on by indexing with the ans array which sorts and
        # selects only the relevant entries.
        x_offsets = geo_array[:, 0]
        y_offsets = geo_array[:, 1]
        # This must be sorted the same as the entries in geo_array!
        # (or channel names, really)
        geo_items_names = []
        for key in sorted(self.geometry):
            geo_items_names.append(key)
        # This is necessary to use np.where below...
        geo_items_names = np.array(geo_items_names)

        # Arrays to hold all traces' data in one:
        _alldata_z = np.zeros((n_stats, npts)) * np.nan
        _alldata_e = _alldata_z.copy()
        _alldata_n = _alldata_z.copy()
        # Array used for sorting and selecting: So far, x_offsets contains
        # offsets for all channels, but the method needs only stations offsets
        # (i.e. a third of the length of the offset array).
        ans = []
        for i, (tr_N, tr_E, tr_Z) in enumerate(zip(stream_n, stream_e,
                                                   stream_z)):
            ans.append(np.where(geo_items_names == tr_N.id)[0][0])
            _alldata_n[i, :] = tr_N.data
            _alldata_e[i, :] = tr_E.data
            _alldata_z[i, :] = tr_Z.data

        fs = stream_n.traces[0].stats.sampling_rate
        nsamp = int(win_len * fs)
        # Number of samples to move forward by during a step.
        nstep = int(nsamp * win_frac)
        # Number of windows is determined by data length minus one window
        # length divided by step length, then adding the one omitted window.
        num_win = int((datalen_sec*fs - nsamp)/nstep) + 1
        alldata_z = np.zeros((n_stats, num_win, nsamp))
        alldata_n, alldata_e = alldata_z.copy(), alldata_z.copy()
        nst = np.zeros(num_win)

        # Iterate over the beamfoming windows:
        for i in range(num_win):
            for n in range(n_stats):
                if not np.isnan(_alldata_z[n, i * nstep:i * nstep +
                                           nsamp]).any() \
                        and not np.isnan(_alldata_n[n, i * nstep:i * nstep +
                                                    nsamp]).any() \
                        and not np.isnan(_alldata_e[n, i * nstep:i * nstep +
                                                    nsamp]).any():
                    # All data, tapered.
                    alldata_z[n, i, :] = _alldata_z[n, i * nstep:
                                                    i * nstep + nsamp] * \
                        cosine_taper(nsamp)
                    alldata_n[n, i, :] = _alldata_n[n, i * nstep:
                                                    i * nstep + nsamp] * \
                        cosine_taper(nsamp)
                    alldata_e[n, i, :] = _alldata_e[n, i * nstep:
                                                    i * nstep + nsamp] * \
                        cosine_taper(nsamp)
                    nst[i] += 1

        # Need an array of the starting times of the beamforming windows for
        # later reference (e.g. plotting). The precision of these should not
        # exceed what is reasonable given the sampling rate: the different
        # traces have the same start times only to within a sample.
        avg_starttime = UTCDateTime(np.mean([tr.stats.starttime.timestamp for
                                             tr in stream_n.traces]))
        window_start_times = \
            np.array([UTCDateTime(avg_starttime + i * nstep/fs,
                                  precision=len(str(fs).split('.')[0]))
                      for i in range(num_win) if i % win_average == 0])

        if verbose:
            print(nst, ' stations/window; average over ', win_average)

        # Do Fourier transform.
        deltat = stream_n.traces[0].stats.delta
        freq_range = np.fft.fftfreq(nsamp, deltat)
        # Use a narrower 'frequency range' of interest for evaluating incidence
        # angle.
        lowcorner = sub_freq_range[0]
        highcorner = sub_freq_range[1]
        index = np.where((freq_range >= lowcorner) &
                         (freq_range <= highcorner))[0]
        fr = freq_range[index]
        fcoeffz = np.fft.fft(alldata_z, n=nsamp, axis=-1) / nsamp
        fcoeffn = np.fft.fft(alldata_n, n=nsamp, axis=-1) / nsamp
        fcoeffe = np.fft.fft(alldata_e, n=nsamp, axis=-1) / nsamp
        fcoeffz = fcoeffz[:, :, index]
        fcoeffn = fcoeffn[:, :, index]
        fcoeffe = fcoeffe[:, :, index]
        deltaf = 1. / (nsamp * deltat)

        if whiten:
            fcoeffz, fcoeffn, fcoeffe = self._three_c_dowhiten(fcoeffz,
                                                               fcoeffn,
                                                               fcoeffe, deltaf)

        # slowness vector u and slowness vector component scale u_x and u_y
        theo_backazi = theo_backazi.reshape((theo_backazi.size, 1))
        u_y = -np.cos(theo_backazi)
        u_x = -np.sin(theo_backazi)

        # vector of source direction dependent plane wave travel-distance to
        # reference point (positive value for later arrival/negative for
        # earlier arr)
        x_offsets = np.array(x_offsets)
        y_offsets = np.array(y_offsets)
        # This sorts the offset value arrays.
        x_offsets = x_offsets[np.array(ans)]
        y_offsets = y_offsets[np.array(ans)]
        # The steering vector corresponds to the 'rho_m * cos(theta_m - theta)'
        # factor in Table 1 of Esmersoy et al., 1985.
        steering = u_y * y_offsets + u_x * x_offsets

        # polarizations [Z,E,N]
        # incident angle P-wave/S-wave or atan(H/V) Rayleigh-wave
        incs = np.arange(5, 90, 10) * math.pi / 180.

        def pol_love(azi):
            pol_e = math.cos(theo_backazi[azi])
            pol_n = -1. * math.sin(theo_backazi[azi])
            return pol_e, pol_n

        def pol_rayleigh_retro(azi):
            pol_e = math.sin(theo_backazi[azi])
            pol_n = math.cos(theo_backazi[azi])
            return pol_e, pol_n

        def pol_rayleigh_prog(azi):
            pol_e = -1 * math.sin(theo_backazi[azi])
            pol_n = -1 * math.cos(theo_backazi[azi])
            return pol_e, pol_n

        def pol_p(azi):
            pol_e = -1 * math.sin(theo_backazi[azi])
            pol_n = -1 * math.cos(theo_backazi[azi])
            return pol_e, pol_n

        def pol_sv(azi):
            pol_e = math.sin(theo_backazi[azi])
            pol_n = math.cos(theo_backazi[azi])
            return pol_e, pol_n

        cz = [0., 1j, 1j, 1., 1.]
        ch = (pol_love, pol_rayleigh_retro, pol_rayleigh_prog, pol_p, pol_sv)

        nfreq = len(fr)
        out_wins = int(np.floor(num_win / win_average))
        beamres = np.zeros((len(theo_backazi), u.size, out_wins, nfreq))
        incidence = np.zeros((out_wins, nfreq))
        win_average = int(win_average)
        for f in range(nfreq):
            omega = 2 * math.pi * fr[f]
            for win in range(0, out_wins * win_average, win_average):
                if any(nst[win:win + win_average] < n_min_stns) or any(
                                nst[win:win + win_average] != nst[win]):
                    continue
                sz = np.squeeze(fcoeffz[:, win, f])
                sn = np.squeeze(fcoeffn[:, win, f])
                se = np.squeeze(fcoeffe[:, win, f])

                y = np.concatenate((sz, sn, se))
                y = y.reshape(1, y.size)
                yt = y.T.copy()
                r = np.dot(yt, np.conjugate(y))

                for wi in range(1, win_average):
                    sz = np.squeeze(fcoeffz[:, win + wi, f])
                    sn = np.squeeze(fcoeffn[:, win + wi, f])
                    se = np.squeeze(fcoeffe[:, win + wi, f])

                    y = np.concatenate((sz, sn, se))
                    y = y.reshape(1, y.size)
                    yt = y.T.copy()
                    r += np.dot(yt, np.conjugate(y))

                r /= float(win_average)

                res = np.zeros((len(theo_backazi), len(u), len(incs)))
                for vel in range(len(u)):
                    e_steer = np.exp(-1j * steering * omega * u[vel])
                    e_steere = e_steer.copy()
                    e_steern = e_steer.copy()
                    e_steere = (e_steere.T * np.array([ch[polarisation](azi)[0]
                                for azi in range(len(theo_backazi))])).T
                    e_steern = (e_steern.T * np.array([ch[polarisation](azi)[1]
                                for azi in range(len(theo_backazi))])).T

                    if polarisation == 0:
                        w = np.concatenate(
                            (e_steer * cz[polarisation], e_steern, e_steere),
                            axis=1)
                        wt = w.T.copy()
                        if not coherency:
                            beamres[:, vel, win / win_average, f] = 1. / (
                                nst[win] * nst[win]) * abs(
                                (np.conjugate(w) * np.dot(r, wt).T).sum(1))
                        else:
                            beamres[:, vel, win / win_average, f] = 1. / (
                                nst[win]) * abs((np.conjugate(w) *
                                                 np.dot(r, wt).T).sum(1)) \
                                / abs(np.sum(np.diag(r[n_stats:, n_stats:])))

                    elif polarisation in [1, 2, 3]:
                        for inc_angle in range(len(incs)):
                            w = np.concatenate((e_steer * cz[polarisation] *
                                                np.cos(incs[inc_angle]),
                                                e_steern *
                                                np.sin(incs[inc_angle]),
                                                e_steere *
                                                np.sin(incs[inc_angle])),
                                               axis=1)
                            wt = w.T.copy()
                            if not coherency:
                                res[:, vel, inc_angle] = 1. / (
                                    nst[win] * nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(r, wt).T).sum(1))
                            else:
                                res[:, vel, inc_angle] = 1. / (nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(r, wt).T).sum(
                                        1)) / abs(np.sum(np.diag(r)))

                    elif polarisation == 4:
                        for inc_angle in range(len(incs)):
                            w = np.concatenate((e_steer * cz[polarisation] *
                                                np.sin(incs[inc_angle]),
                                                e_steern *
                                                np.cos(incs[inc_angle]),
                                                e_steere *
                                                np.cos(incs[inc_angle])),
                                               axis=1)
                            wt = w.T.copy()
                            if not coherency:
                                res[:, vel, inc_angle] = 1. / (
                                    nst[win] * nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(r, wt).T).sum(1))
                            else:
                                res[:, vel, inc_angle] = 1. / (nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(r, wt).T).sum(
                                        1)) / abs(np.sum(np.diag(r)))

                if polarisation > 0:
                    i, j, k = np.unravel_index(np.argmax(res[:, uindex, :]),
                                               res.shape)
                    beamres[:, :, win / win_average, f] = res[:, :, k]
                    incidence[win / win_average, f] = incs[k] * 180. / math.pi

        return beamres, fr, incidence, window_start_times

    def three_component_beamforming(self, stream_n, stream_e, stream_z, wlen,
                                    smin, smax, sstep, wavetype, freq_range,
                                    n_min_stns=7, win_average=1, win_frac=1,
                                    whiten=True, coherency=False):
        """
        Do three-component beamforming following [Esmersoy1985]_.

        Three streams representing N, E, Z oriented components must be given,
        where the traces contained are from the different stations. The
        traces must all have same length and start/end times (to within
        sampling distance). (hint: check length with trace.stats.npts)
        The given streams are not modified in place. All trimming, filtering,
        downsampling should be done previously.
        The beamforming can distinguish Love, prograde/retrograde Rayleigh, P
        and SV waves.
        Station location information is taken from the array's inventory, so
        that must contain station or channel location information about all
        traces used (or more, the inventory is then non-permanently 'pruned').
        NB all channels of a station must be located in the same location for
        this method.

        :param stream_n: Stream of all traces for the North component.
        :param stream_e: Stream of East components.
        :param stream_z: Stream of Up components. Will be ignored for Love
         waves.
        :param wlen: window length in seconds
        :param smin: minimum slowness of the slowness grid [s/km]
        :param smax: maximum slowness [s/km]
        :param sstep: slowness step [s/km]
        :param wavetype: 'love', 'rayleigh_prograde', 'rayleigh_retrograde',
         'P', or 'SV'
        :param freq_range: Frequency band (min, max) that is used for
         beamforming and returned. Ideally, use the frequency band of the
         pre-filter.
        :param n_min_stns: Minimum number of stations for which data must be
         present in a time window, otherwise that window is skipped.
        :param win_average: number of windows to average covariance matrix over
        :param win_frac: fraction of sliding window to use for step
        :param whiten: whether to whiten the frequency spectrum
        :param coherency: whether to normalise the powers
        :return: A :class:`~obspy.signal.array_analysis.BeamformerResult`
        object containing the beamforming results, with dimensions of
        backazimuth range, slowness range, number of windows and number of
        discrete frequencies; as well as frequency and incidence angle arrays
        (the latter will be zero for Love waves).
        """
        pol_dict = {'love': 0, 'rayleigh_retrograde': 1, 'rayleigh_prograde':
                    2, 'p': 3, 'sv': 4}
        if wavetype.lower() not in pol_dict:
            raise ValueError('Invalid option for wavetype: {}'
                             .format(wavetype))
        if len(set(len(vel.traces) for vel in (stream_n, stream_e,
                                               stream_z))) > 1:
            raise ValueError("All three streams must have same number of "
                             "traces.")
        if len(stream_n.traces) == 0:
            raise ValueError("Streams do not seem to contain any traces.")

        # from _array_analysis_helper:
        starttime = max(max([tr.stats.starttime for tr in st]) for st in
                        (stream_n, stream_e, stream_e))
        min_starttime = min(min([tr.stats.starttime for tr in st]) for st in
                            (stream_n, stream_e, stream_e))
        endtime = min(min([tr.stats.endtime for tr in st]) for st in
                      (stream_n, stream_e, stream_e))
        max_endtime = max(max([tr.stats.endtime for tr in st]) for st in
                          (stream_n, stream_e, stream_e))

        delta_common = stream_n.traces[0].stats.delta
        npts_common = stream_n.traces[0].stats.npts
        if max(abs(min_starttime - starttime),
               abs(max_endtime - endtime)) > delta_common:
            raise ValueError("Traces do not have identical start/end times. "
                             "Trim to same times (within sample accuracy) "
                             "and ensure all traces have the same number "
                             "of samples!")

        # Check for equal deltas and number of samples:
        for st in (stream_n, stream_e, stream_z):
            for tr in st:
                if tr.stats.npts != npts_common:
                    raise ValueError('Traces do not have identical number of '
                                     'samples.')
                if tr.stats.delta != delta_common:
                    raise ValueError('Traces do not have identical sampling '
                                     'rates.')
        datalen_sec = endtime - starttime

        # Sort all traces just to make sure they're in the same order.
        for st in (stream_n, stream_e, stream_z):
            st.sort()

        for trN, trE, trZ in zip(stream_n, stream_e, stream_z):
            if len(set('{}.{}'.format(tr.stats.network, tr.stats.station)
                       for tr in (trN, trE, trZ))) > 1:
                raise ValueError("Traces are not from same stations.")

        # Temporarily trim self.inventory so only stations/channels which are
        # actually represented in the traces are kept in the inventory.
        # Otherwise self.geometry and the xyz geometry arrays will have more
        # entries than the stream.
        invbkp = copy.deepcopy(self.inventory)
        allstreams = stream_n + stream_e + stream_z
        self.inventory_cull(allstreams)

        try:
            if wlen < smax * self.aperture:
                raise ValueError('Window length is smaller than maximum given'
                                 ' slowness times aperture.')
            # s/km  slowness range calculated
            u = np.arange(smin, smax, sstep)
            # Slowness range evaluated for (incidence) angle measurement
            # (Rayleigh, P, SV):
            # These values are a bit arbitrary for now:
            uindex = np.where((u > 0.5 * smax + smin) &
                              (u < 0.8 * smax + smin))[0]

            bf_results, freqs, incidence, window_start_times = \
                self._three_c_do_bf(stream_n, stream_e, stream_z,
                                    win_len=wlen, win_frac=win_frac, u=u,
                                    sub_freq_range=freq_range,
                                    n_min_stns=n_min_stns,
                                    polarisation=pol_dict[wavetype.lower()],
                                    whiten=whiten,
                                    coherency=coherency,
                                    win_average=win_average,
                                    datalen_sec=datalen_sec,
                                    uindex=uindex)

            out = BeamformerResult(inventory=self.inventory,
                                   win_starttimes=window_start_times,
                                   slowness_range=u, full_beamres=bf_results,
                                   freqs=freqs, incidence=incidence,
                                   method='3C ({})'.format(wavetype),
                                   timestep=wlen * win_frac * win_average)

        finally:
            self.inventory = invbkp

        return out

    # todo: make this use array geometry!
    @staticmethod
    def array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs,
                              array_coords, sigmau):
        """
        This routine calculates the best-fitting rigid body rotation and
        uniform strain as functions of time, and their formal errors, given
        three-component ground motion time series recorded on a seismic array.
        The theory implemented herein is presented in the papers
        [Spudich1995]_, (abbreviated S95 herein) [Spudich2008]_ (SF08) and
        [Spudich2009]_ (SF09).

        This is a translation of the Matlab Code presented in (SF09) with
        small changes in details only. Output has been checked to be the same
        as the original Matlab Code.

        .. note::
            ts\_ below means "time series"

        :param vp: P wave speed in the soil under the array (km/s).
        :param vs: S wave speed in the soil under the array Note - vp and vs
         may be any unit (e.g. miles/week), and this unit need not be related
         to the units of the station coordinates or ground motions, but the
         units of vp and vs must be the SAME because only their ratio is used.
        :type array_coords: numpy.ndarray
        :param array_coords: Array of dimension Na x 3, where Na is the number
         of stations in the array.  array_coords[i,j], i in arange(Na), j in
         arange(3) is j coordinate of station i.  units of array_coords may be
         anything, but see the "Discussion of input and output units" above.
         The origin of coordinates is arbitrary and does not affect the
         calculated strains and rotations.  Stations may be entered in any
         order.
        :type ts1: numpy.ndarray
        :param ts1: Array of x1-component seismograms, dimension nt x Na.
         ts1[j,k], j in arange(nt), k in arange(Na) contains the k'th time
         sample of the x1 component ground motion at station k. NOTE that the
         seismogram in column k must correspond to the station whose
         coordinates are in row k of in.array_coords. nt is the number of time
         samples in the seismograms.  Seismograms may be displacement,
         velocity, acceleration, jerk, etc.  See the "Discussion of input and
         output units" below.
        :type ts2: numpy.ndarray
        :param ts2: Aame as ts1, but for the x2 component of motion.
        :type ts3: numpy.ndarray
        :param ts3: Aame as ts1, but for the x3 (UP or DOWN) component of
         motion.
        :type sigmau: float or :class:`numpy.ndarray`
        :param sigmau: Standard deviation (NOT VARIANCE) of ground noise,
         corresponds to sigma-sub-u in S95 lines above eqn (A5).
         NOTE: This may be entered as a scalar, vector, or matrix!

         * If sigmau is a scalar, it will be used for all components of all
           stations.
         * If sigmau is a 1D array of length Na, sigmau[i] will be the noise
           assigned to all components of the station corresponding to
           array_coords[i,:]
         * If sigmau is a 2D array of dimension  Na x 3, then sigmau[i,j] is
           used as the noise of station i, component j.

         In all cases, this routine assumes that the noise covariance between
         different stations and/or components is zero.
        :type subarray: numpy.ndarray
        :param subarray: NumPy array of subarray stations to use. I.e. if
         subarray = array([1, 4, 10]), then only rows 1, 4, and 10 of
         array_coords will be used, and only ground motion time series in the
         first, fourth, and tenth columns of ts1 will be used. n_plus_1 is the
         number of elements in the subarray vector, and n is set to
         n_plus_1 - 1. To use all stations in the array, set
         in.subarray = arange(Na), where Na is the total number of stations in
         the array (equal to the number of rows of in.array_coords. Sequence of
         stations in the subarray vector is  unimportant; i.e. subarray =
         array([1, 4, 10]) will yield essentially the same rotations and
         strains as subarray = array([10, 4, 1]). "Essentially" because
         permuting subarray sequence changes the d vector, yielding a slightly
         different numerical result.
        :return: Dictionary with fields:

        **a:** (array, dimension 3N x 6)
            data mapping matrix 'a' of S95(A4)
        **g:** (array, dimension 6 x 3N)
            generalized inverse matrix relating ptilde and data vector, in
            S95(A5)
        **ce:** (4 x 4)
            covariance matrix of the 4 independent strain tensor elements e11,
            e21, e22, e33
        **ts_d:** (array, length nt)
            dilatation (trace of the 3x3 strain tensor) as a function of time
        **sigmad:** (scalar)
            standard deviation of dilatation
        **ts_dh:** (array, length nt)
            horizontal dilatation (also known as areal strain) (eEE+eNN) as a
            function of time
        **sigmadh:** (scalar)
            standard deviation of horizontal dilatation (areal strain)
        **ts_e:** (array, dimension nt x 3 x 3)
            strain tensor
        **ts_s:** (array, length nt)
            maximum strain ( .5*(max eigval of e - min eigval of e) as a
            function of time, where e is the 3x3 strain tensor
        **cgamma:** (4 x 4)
            covariance matrix of the 4 independent shear strain tensor elements
            g11, g12, g22, g33 (includes full covariance effects). gamma is
            traceless part of e.
        **ts_sh:** (array, length nt)
            maximum horizontal strain ( .5*(max eigval of eh - min eigval of
            eh) as a function of time, where eh is e(1:2,1:2)
        **cgammah:** (3 x 3)
            covariance matrix of the 3 independent horizontal shear strain
            tensor elements gamma11, gamma12, gamma22 gamma is traceless part
            of e.
        **ts_wmag:** (array, length nt)
            total rotation angle (radians) as a function of time.  I.e. if the
            rotation vector at the j'th time step is
            w = array([w1, w2, w3]), then ts_wmag[j] = sqrt(sum(w**2))
            positive for right-handed rotation
        **cw:** (3 x 3)
            covariance matrix of the 3 independent rotation tensor elements
            w21, w31, w32
        **ts_w1:** (array, length nt)
            rotation (rad) about the x1 axis, positive for right-handed
            rotation
        **sigmaw1:** (scalar)
            standard deviation of the ts_w1 (sigma-omega-1 in SF08)
        **ts_w2:** (array, length nt)
            rotation (rad) about the x2 axis, positive for right-handed
            rotation
        **sigmaw2:** (scalar)
            standard deviation of ts_w2 (sigma-omega-2 in SF08)
        **ts_w3:** (array, length nt)
            "torsion", rotation (rad) about a vertical up or down axis, i.e.
            x3, positive for right-handed rotation
        **sigmaw3:** (scalar)
            standard deviation of the torsion (sigma-omega-3 in SF08)
        **ts_tilt:** (array, length nt)
            tilt (rad) (rotation about a horizontal axis, positive for right
            handed rotation) as a function of time
            tilt = sqrt( w1^2 + w2^2)
        **sigmat:** (scalar)
            standard deviation of the tilt (not defined in SF08, From
            Papoulis (1965, p. 195, example 7.8))
        **ts_data:** (array, shape (nt x 3N))
            time series of the observed displacement differences, which are
            the di in S95 eqn A1
        **ts_pred:** (array, shape (nt x 3N))
            time series of the fitted model's predicted displacement difference
            Note that the fitted model displacement differences correspond
            to linalg.dot(a, ptilde), where a is the big matrix in S95 eqn A4
            and ptilde is S95 eqn A5
        **ts_misfit:** (array, shape (nt x 3N))
            time series of the residuals (fitted model displacement differences
            minus observed displacement differences). Note that the fitted
            model displacement differences correspond to linalg.dot(a, ptilde),
            where a is the big matrix in S95 eqn A4 and ptilde is S95 eqn A5
        **ts_m:** (array, length nt)
            Time series of M, misfit ratio of S95, p. 688
        **ts_ptilde:** (array, shape (nt x 6))
            solution vector p-tilde (from S95 eqn A5) as a function of time
        **cp:** (6 x 6)
            solution covariance matrix defined in SF08

        .. rubric:: Warnings

        This routine does not check to verify that your array is small
        enough to conform to the assumption that the array aperture is less
        than 1/4 of the shortest seismic wavelength in the data. See SF08
        for a discussion of this assumption.

        This code assumes that ts1[j,:], ts2[j,:], and ts3[j,:] are all sampled
        SIMULTANEOUSLY.

        .. rubric:: Notes

        (1) Note On Specifying Input Array And Selecting Subarrays

            This routine allows the user to input the coordinates and ground
            motion time series of all stations in a seismic array having Na
            stations and the user may select for analysis a subarray of
            n_plus_1 <= Na stations.

        (2) Discussion Of Physical Units Of Input And Output

            If the input seismograms are in units of displacement, the output
            strains and rotations will be in units of strain (unitless) and
            angle (radians).  If the input seismograms are in units of
            velocity, the output will be strain rate (units = 1/s) and rotation
            rate (rad/s).  Higher temporal derivative inputs yield higher
            temporal derivative outputs.

            Input units of the array station coordinates must match the spatial
            units of the seismograms.  For example, if the input seismograms
            are in units of m/s^2, array coordinates must be entered in m.

        (3) Note On Coordinate System

            This routine assumes x1-x2-x3 is a RIGHT handed orthogonal
            coordinate system. x3 must point either UP or DOWN.
        """
        # This assumes that all stations and components have the same number of
        # time samples, nt
        [nt, na] = np.shape(ts1)

        # check to ensure all components have same duration
        if ts1.shape != ts2.shape:
            raise ValueError('ts1 and ts2 have different sizes')
        if ts1.shape != ts3.shape:
            raise ValueError('ts1 and ts3 have different sizes')

        # check to verify that the number of stations in ts1 agrees with the
        # number of stations in array_coords
        nrac, _ = array_coords.shape
        if nrac != na:
            msg = 'ts1 has %s columns(stations) but array_coords has ' % na + \
                  '%s rows(stations)' % nrac
            raise ValueError(msg)

        # check stations in subarray exist
        if min(subarray) < 0:
            raise ValueError('Station number < 0 in subarray')
        if max(subarray) > na:
            raise ValueError('Station number > Na in subarray')

        # extract the stations of the subarray to be used
        subarraycoords = array_coords[subarray, :]

        # count number of subarray stations: n_plus_1 and number of station
        # offsets: n
        n_plus_1 = subarray.size
        n = n_plus_1 - 1

        if n_plus_1 < 3:
            msg = 'The problem is underdetermined for fewer than 3 stations'
            raise ValueError(msg)
        elif n_plus_1 == 3:
            msg = 'For a 3-station array the problem is even-determined'
            warnings.warn(msg)

        # ------------------- NOW SOME SEISMOLOGY!! --------------------------
        # constants
        eta = 1 - 2 * vs ** 2 / vp ** 2

        # form a matrix, which relates model vector of 6 displacement
        # derivatives to vector of observed displacement differences. S95(A3)
        # dim(a) = (3*n) * 6
        # model vector is [ u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ] (free surface
        # boundary conditions applied, S95(A2))
        # first initialize a to the null matrix
        a = np.zeros((n * 3, 6))
        z3t = np.zeros(3)
        # fill up a
        for i in range(n):
            ss = subarraycoords[(i + 1), :] - subarraycoords[0, :]
            a[(3 * i):(3 * i + 3), :] = np.c_[
                np.r_[ss, z3t], np.r_[z3t, ss],
                np.array([-eta * ss[2],
                          0., -ss[0], 0., -eta * ss[2], -ss[1]])].transpose()

        # ------------------------------------------------------
        # define data covariance matrix cd.
        # step 1 - define data differencing matrix d
        # dimension of d is (3*n) * (3*n_plus_1)
        i3 = np.eye(3)
        ii = np.eye(3 * n)
        d = -i3

        for i in range(n - 1):
            d = np.c_[d, -i3]
        d = np.r_[d, ii].T

        # step 2 - define displacement u covariance matrix cu
        # This assembles a covariance matrix cu that reflects actual
        # data errors.
        # populate cu depending on the size of sigmau
        if np.size(sigmau) == 1:
            # sigmau is a scalar.  Make all diag elements of cu the same
            cu = sigmau ** 2 * np.eye(3 * n_plus_1)
        elif np.shape(sigmau) == (np.size(sigmau),):
            # sigmau is a row or column vector
            # check dimension is okay
            if np.size(sigmau) != na:
                raise ValueError('sigmau must have %s elements' % na)
            junk = (np.c_[sigmau, sigmau, sigmau]) ** 2  # matrix of variances
            cu = np.diag(np.reshape(junk[subarray, :], (3 * n_plus_1)))
        elif sigmau.shape == (na, 3):
            cu = np.diag(np.reshape(((sigmau[subarray, :]) ** 2).transpose(),
                                    (3 * n_plus_1)))
        else:
            raise ValueError('sigmau has the wrong dimensions')

        # cd is the covariance matrix of the displ differences
        # dim(cd) is (3*n) * (3*n)
        cd = np.dot(np.dot(d, cu), d.T)

        # ---------------------------------------------------------
        # form generalized inverse matrix g.  dim(g) is 6 x (3*n)
        cdi = np.linalg.inv(cd)
        atcdia = np.dot(np.dot(a.T, cdi), a)
        g = np.dot(np.dot(np.linalg.inv(atcdia), a.T), cdi)

        condition_number = np.linalg.cond(atcdia)

        if condition_number > 100:
            msg = 'Condition number is %s' % condition_number
            warnings.warn(msg)

        # set up storage for vectors that will contain time series
        ts_wmag = np.empty(nt)
        ts_w1 = np.empty(nt)
        ts_w2 = np.empty(nt)
        ts_w3 = np.empty(nt)
        ts_tilt = np.empty(nt)
        ts_dh = np.empty(nt)
        ts_sh = np.empty(nt)
        ts_s = np.empty(nt)
        ts_pred = np.empty((nt, 3 * n))
        ts_misfit = np.empty((nt, 3 * n))
        ts_m = np.empty(nt)
        ts_data = np.empty((nt, 3 * n))
        ts_ptilde = np.empty((nt, 6))
        for array in (ts_wmag, ts_w1, ts_w2, ts_w3, ts_tilt, ts_dh, ts_sh,
                      ts_s, ts_pred, ts_misfit, ts_m, ts_data, ts_ptilde):
            array.fill(np.NaN)
        ts_e = np.empty((nt, 3, 3))
        ts_e.fill(np.NaN)

        # other matrices
        udif = np.empty((3, n))
        udif.fill(np.NaN)

        # ---------------------------------------------------------------
        # here we define 4x6 be and 3x6 bw matrices.  these map the solution
        # ptilde to strain or to rotation.  These matrices will be used
        # in the calculation of the covariances of strain and rotation.
        # Columns of both matrices correspond to the model solution vector
        # containing elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]'
        #
        # the rows of be correspond to e11 e21 e22 and e33
        be = np.zeros((4, 6))
        be[0, 0] = 2.
        be[1, 1] = 1.
        be[1, 3] = 1.
        be[2, 4] = 2.
        be[3, 0] = -2 * eta
        be[3, 4] = -2 * eta
        be *= .5
        #
        # the rows of bw correspond to w21 w31 and w32
        bw = np.zeros((3, 6))
        bw[0, 1] = 1.
        bw[0, 3] = -1.
        bw[1, 2] = 2.
        bw[2, 5] = 2.
        bw *= .5
        #
        # This is the 4x6 matrix mapping solution to total shear strain gamma
        # where gamma = strain - tr(strain)/3 * eye(3)
        # the four elements of shear are 11, 12, 22, and 33.  It is symmetric.
        aa = (2 + eta) / 3
        b = (1 - eta) / 3
        c = (1 + 2 * eta) / 3
        bgamma = np.zeros((4, 6))
        bgamma[0, 0] = aa
        bgamma[0, 4] = -b
        bgamma[2, 2] = .5
        bgamma[1, 3] = .5
        bgamma[2, 0] = -b
        bgamma[2, 4] = aa
        bgamma[3, 0] = -c
        bgamma[3, 4] = -c

        # This is the 3x6 matrix mapping solution to horizontal shear strain
        #  gamma the four elements of horiz shear are 11, 12, and 22.  It is
        #  symmetric.
        bgammah = np.zeros((3, 6))
        bgammah[0, 0] = .5
        bgammah[0, 4] = -.5
        bgammah[1, 1] = .5
        bgammah[1, 3] = .5
        bgammah[2, 0] = -.5
        bgammah[2, 4] = .5

        # Solution covariance matrix.  dim(cp) = 6 * 6
        # corresponding to solution elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]
        cp = np.dot(np.dot(g, cd), g.T)

        # Covariance of strain tensor elements
        # ce should be 4x4, correspond to e11, e21, e22, e33
        ce = np.dot(np.dot(be, cp), be.T)
        # cw should be 3x3 correspond to w21, w31, w32
        cw = np.dot(np.dot(bw, cp), bw.T)

        # cgamma is 4x4 correspond to 11, 12, 22, and 33.
        cgamma = np.dot(np.dot(bgamma, cp), bgamma.T)
        #
        #  cgammah is 3x3 correspond to 11, 12, and 22
        cgammah = np.dot(np.dot(bgammah, cp), bgammah.T)
        #
        #
        # covariance of the horizontal dilatation and the total dilatation
        # both are 1x1, i.e. scalars
        cdh = cp[0, 0] + 2 * cp[0, 4] + cp[4, 4]
        sigmadh = np.sqrt(cdh)

        # covariance of the (total) dilatation, ts_dd
        sigmadsq = (1 - eta) ** 2 * cdh
        sigmad = np.sqrt(sigmadsq)
        #
        # cw3, covariance of w3 rotation, i.e. torsion, is 1x1, i.e. scalar
        cw3 = (cp[1, 1] - 2 * cp[1, 3] + cp[3, 3]) / 4
        sigmaw3 = np.sqrt(cw3)

        # For tilt cannot use same approach because tilt is not a linear
        # function
        # of the solution.  Here is an approximation :
        # For tilt use conservative estimate from
        # Papoulis (1965, p. 195, example 7.8)
        sigmaw1 = np.sqrt(cp[5, 5])
        sigmaw2 = np.sqrt(cp[2, 2])
        sigmat = max(sigmaw1, sigmaw2) * np.sqrt(2 - np.pi / 2)

        #
        # BEGIN LOOP OVER DATA POINTS IN TIME SERIES==========================
        #
        for itime in range(nt):
            #
            # data vector is differences of stn i displ from stn 1 displ
            # sum the lengths of the displ difference vectors
            sumlen = 0
            for i in range(n):
                udif[0, i] = ts1[itime, subarray[i + 1]] - \
                    ts1[itime, subarray[0]]
                udif[1, i] = ts2[itime, subarray[i + 1]] - \
                    ts2[itime, subarray[0]]
                udif[2, i] = ts3[itime, subarray[i + 1]] - \
                    ts3[itime, subarray[0]]
                sumlen = sumlen + np.sqrt(np.sum(udif[:, i].T ** 2))

            data = udif.T.reshape(udif.size)
            #
            # form solution
            # ptilde is (u1,1 u1,2 u1,3 u2,1 u2,2 u2,3).T
            ptilde = np.dot(g, data)
            #
            # place in uij_vector the full 9 elements of the displacement
            # gradients uij_vector is
            # (u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 u3,1 u3,2 u3,3).T
            # The following implements the free surface boundary condition
            u31 = -ptilde[2]
            u32 = -ptilde[5]
            u33 = -eta * (ptilde[0] + ptilde[4])
            uij_vector = np.r_[ptilde, u31, u32, u33]
            #
            # calculate predicted data
            pred = np.dot(a, ptilde)  # 9/8/92.I.3(9) and 8/26/92.I.3.T bottom
            #
            # calculate  residuals (misfits concatenated for all stations)
            misfit = pred - data

            # Calculate ts_m, misfit ratio.
            # calculate summed length of misfits (residual displacements)
            misfit_sq = misfit ** 2
            misfit_sq = np.reshape(misfit_sq, (n, 3)).T
            misfit_sumsq = np.empty(n)
            misfit_sumsq.fill(np.NaN)
            for i in range(n):
                misfit_sumsq[i] = misfit_sq[:, i].sum()
            misfit_len = np.sum(np.sqrt(misfit_sumsq))
            ts_m[itime] = misfit_len / sumlen
            #
            ts_data[itime, 0:3 * n] = data.T
            ts_pred[itime, 0:3 * n] = pred.T
            ts_misfit[itime, 0:3 * n] = misfit.T
            ts_ptilde[itime, :] = ptilde.T
            #
            # ---------------------------------------------------------------
            # populate the displacement gradient matrix u
            u = np.zeros(9)
            u[:] = uij_vector
            u = u.reshape((3, 3))
            #
            # calculate strain tensors
            # Fung eqn 5.1 p 97 gives dui = (eij-wij)*dxj
            e = .5 * (u + u.T)
            ts_e[itime] = e

            # Three components of the rotation vector omega (=w here)
            w = np.empty(3)
            w.fill(np.NaN)
            w[0] = -ptilde[5]
            w[1] = ptilde[2]
            w[2] = .5 * (ptilde[3] - ptilde[1])

            # amount of total rotation is length of rotation vector
            ts_wmag[itime] = np.sqrt(np.sum(w ** 2))
            #
            # Calculate tilt and torsion
            ts_w1[itime] = w[0]
            ts_w2[itime] = w[1]
            ts_w3[itime] = w[2]  # torsion in radians
            ts_tilt[itime] = np.sqrt(w[0] ** 2 + w[1] ** 2)
            # 7/21/06.ii.6(19), amount of tilt in radians

            # ---------------------------------------------------------------
            #
            # Here I calculate horizontal quantities only
            # ts_dh is horizontal dilatation (+ --> expansion).
            # Total dilatation, ts_dd, will be calculated outside the time
            # step loop.
            #
            ts_dh[itime] = e[0, 0] + e[1, 1]
            #
            # find maximum shear strain in horizontal plane, and find its
            # azimuth
            eh = np.r_[np.c_[e[0, 0], e[0, 1]], np.c_[e[1, 0], e[1, 1]]]
            # 7/21/06.ii.2(4)
            gammah = eh - np.trace(eh) * np.eye(2) / 2.
            # 9/14/92.ii.4, 7/21/06.ii.2(5)

            # eigvecs are principal axes, eigvals are principal strains
            [eigvals, _eigvecs] = np.linalg.eig(gammah)
            # max shear strain, from Fung (1965, p71, eqn (8)
            ts_sh[itime] = .5 * (max(eigvals) - min(eigvals))

            # calculate max of total shear strain, not just horizontal strain
            # eigvecs are principal axes, eigvals are principal strains
            [eigvalt, _eigvect] = np.linalg.eig(e)
            # max shear strain, from Fung (1965, p71, eqn (8)
            ts_s[itime] = .5 * (max(eigvalt) - min(eigvalt))
            #

        # ====================================================================
        #
        # (total) dilatation is a scalar times horizontal dilatation owing to
        # their free surface boundary condition
        ts_d = ts_dh * (1 - eta)

        # load output structure
        out = dict()

        out['a'] = a
        out['g'] = g
        out['ce'] = ce

        out['ts_d'] = ts_d
        out['sigmad'] = sigmad

        out['ts_dh'] = ts_dh
        out['sigmadh'] = sigmadh

        out['ts_s'] = ts_s
        out['cgamma'] = cgamma

        out['ts_sh'] = ts_sh
        out['cgammah'] = cgammah

        out['ts_wmag'] = ts_wmag
        out['cw'] = cw

        out['ts_w1'] = ts_w1
        out['sigmaw1'] = sigmaw1
        out['ts_w2'] = ts_w2
        out['sigmaw2'] = sigmaw2
        out['ts_w3'] = ts_w3
        out['sigmaw3'] = sigmaw3

        out['ts_tilt'] = ts_tilt
        out['sigmat'] = sigmat

        out['ts_data'] = ts_data
        out['ts_pred'] = ts_pred
        out['ts_misfit'] = ts_misfit
        out['ts_m'] = ts_m
        out['ts_e'] = ts_e

        out['ts_ptilde'] = ts_ptilde
        out['cp'] = cp

        out['ts_m'] = ts_m

        return out

    @staticmethod
    def get_stream_offsets(stream, stime, etime):
        """
        Calculates start and end offsets relative to stime and etime for each
        trace in stream in samples.

        :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param stime: Start time
        :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param etime: End time
        :returns: start and end sample offset arrays
        """
        spoint = np.empty(len(stream), dtype=np.int32, order="C")
        epoint = np.empty(len(stream), dtype=np.int32, order="C")
        for i, tr in enumerate(stream):
            if tr.stats.starttime > stime:
                msg = "Specified stime %s is smaller than starttime %s " \
                      "in stream"
                raise ValueError(msg % (stime, tr.stats.starttime))
            if tr.stats.endtime < etime:
                msg = "Specified etime %s is bigger than endtime %s in stream"
                raise ValueError(msg % (etime, tr.stats.endtime))
            # now we have to adjust to the beginning of real start time
            spoint[i] = int(
                (stime - tr.stats.starttime) * tr.stats.sampling_rate + .5)
            epoint[i] = int(
                (tr.stats.endtime - etime) * tr.stats.sampling_rate + .5)
        return spoint, epoint

    def plot_radial_transfer_function(self, smin, smax, sstep, freqs):
        """
        Plot array transfer function radially, as function of slowness.
        """
        import matplotlib.pyplot as plt
        u = np.arange(smin, smax, sstep)
        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.
        theo_backazi = theo_backazi.reshape((theo_backazi.size, 1))
        u_y = -np.cos(theo_backazi)
        u_x = -np.sin(theo_backazi)
        geo_array = self._geometry_dict_to_array(
            self.get_geometry_xyz(**self.center_of_gravity))
        x_ = geo_array[:, 0]
        y_ = geo_array[:, 1]
        x_ = np.array(x_)
        y_ = np.array(y_)
        steering = u_y * y_ + u_x * x_
        theo_backazi = theo_backazi[:, 0]
        beamres = np.zeros((len(theo_backazi), u.size))
        for f in freqs:
            omega = 2. * math.pi * f
            r = np.ones((steering.shape[1], steering.shape[1]))
            for vel in range(len(u)):
                e_steer = np.exp(-1j * steering * omega * u[vel])
                w = e_steer
                wt = w.T.copy()
                beamres[:, vel] = 1. / (
                    steering.shape[1] * steering.shape[1]) * abs(
                    (np.conjugate(w) * np.dot(r, wt).T).sum(1))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            cmap = cm.hot_r
            contf = ax.contourf(theo_backazi, u,
                                beamres.T, 100, cmap=cmap, antialiased=True,
                                linstyles='dotted')
            ax.contour(theo_backazi, u, beamres.T, 100,
                       cmap=cmap)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_rgrids([0.1, 0.2, 0.3, 0.4, 0.5],
                          labels=['0.1', '0.2', '0.3', '0.4', '0.5'],
                          color='r')
            ax.set_rmax(u[-1])
            # This means that if u does not start at 0, the plot will show a
            # hole in the middle rather than stitching it up.
            ax.set_rmin(-0)
            fig.colorbar(contf)
            ax.grid(True)
            ax.set_title('Transfer function for frequency ' + str(f) + '.')
        plt.show()

    def plot_transfer_function_wavenumber(self, klim, kstep):
        """
        Plot array transfer function as function of wavenumber.

        :param klim: Maximum wavenumber (symmetric about zero in
         x and y directions).
        :param kstep: Step in wavenumber.
        """
        import matplotlib.pyplot as plt
        transff = self.array_transff_wavenumber(klim, kstep)
        self._plot_transff_helper(transff, klim, kstep)
        plt.xlabel('Wavenumber West-East')
        plt.ylabel('Wavenumber North-South')
        plt.show()

    def plot_transfer_function_freqslowness(self, slim, sstep, freq_min,
                                            freq_max, freq_step):
        """
        Plot array transfer function as function of slowness and frequency.

        :param slim: Maximum slowness (symmetric about zero in x and y
         directions).
        :param sstep: Step in slowness.
        :param freq_min: Minimum frequency in signal.
        :param freq_max: Maximum frequency in signal.
        :param freq_step: Frequency sample distance
        """
        import matplotlib.pyplot as plt
        transff = self.array_transff_freqslowness(slim, sstep, freq_min,
                                                  freq_max, freq_step)
        self._plot_transff_helper(transff, slim, sstep)
        plt.xlabel('Slowness West-East')
        plt.ylabel('Slowness North-South')
        plt.show()

    @staticmethod
    def _plot_transff_helper(transff, lim, step):
        """
        Plot array transfer function.

        :param transff: Transfer function to plot.
        :param lim: Maximum value of slowness/wavenumber.
        :param step: Step in slowness/wavenumber.
        """
        import matplotlib.pyplot as plt
        ranges = np.arange(-lim, lim + step, step)
        plt.pcolor(ranges, ranges, transff.T, cmap=cm.hot_r)
        plt.colorbar()
        plt.clim(vmin=0., vmax=1.)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

    def array_transff_wavenumber(self, klim, kstep):
        """
        Return array transfer function as a function of wavenumber difference.

        :param klim: Either a float to use symmetric limits for wavenumber
            differences or the tuple (kxmin, kxmax, kymin, kymax).
        :param kstep: Step in wavenumber.
        """
        return self._array_transff_helper(klim, kstep, 'wavenumber')

    def array_transff_freqslowness(self, slim, sstep, fmin, fmax,
                                   fstep):
        """
        Return array transfer function as a function of slowness difference
        and frequency.

        :param slim: Either a float to use symmetric limits for slowness
            differences or the tuple (sxmin, sxmax, symin, symax).
        :param sstep: Step in frequency.
        :param fmin: Minimum frequency in signal.
        :param fmax: Maximum frequency in signal.
        :param fstep: Frequency sample distance.
        """

        return self._array_transff_helper(slim, sstep, 'slowness', fmin, fmax,
                                          fstep)

    def _array_transff_helper(self, plim, pstep, param, fmin=None, fmax=None,
                              fstep=None):
        """
        Return array transfer function as function of wavenumber or slowness
        and frequency.

        :param plim: Either a float to use symmetric limits for slowness/
            wavenumber differences or the tuple (pxmin, pxmax, sxmin, sxmax).
        :param pstep: Step in wavenumber/slowness.
        :param param: 'wavenumber' or 'slowness'
        :param fmin: Minimum frequency (only for slowness calculation).
        :param fmax: Maximum frequency (only for slowness calculation).
        :param fstep: Frequency sample distance (only with slowness).
        """
        if isinstance(plim, float) or isinstance(plim, int):
            pxmin = -plim
            pxmax = plim
            pymin = -plim
            pymax = plim
        elif len(plim) == 4:
            pxmin = plim[0]
            pxmax = plim[1]
            pymin = plim[2]
            pymax = plim[3]
        else:
            raise TypeError('Parameter slim must either be a float '
                            'or a tuple of length 4.')
        geometry = self._geometry_dict_to_array(self.get_geometry_xyz(
            **self.center_of_gravity))
        npx = int(np.ceil((pxmax + pstep / 10. - pxmin) / pstep))
        npy = int(np.ceil((pymax + pstep / 10. - pymin) / pstep))
        transff = np.empty((npx, npy))

        if param is 'wavenumber':
            for i, kx in enumerate(np.arange(pxmin, pxmax + pstep / 10.,
                                             pstep)):
                for j, ky in enumerate(np.arange(pymin, pymax + pstep / 10.,
                                                 pstep)):
                    _sum = 0j
                    for k in range(len(geometry)):
                        _sum += np.exp(complex(0., geometry[k, 0] * kx +
                                               geometry[k, 1] * ky))
                    transff[i, j] = abs(_sum) ** 2

        elif param is 'slowness':
            nf = int(np.ceil((fmax + fstep / 10. - fmin) / fstep))
            buff = np.zeros(nf)
            for i, sx in enumerate(np.arange(pxmin, pxmax + pstep / 10.,
                                             pstep)):
                for j, sy in enumerate(np.arange(pymin, pymax + pstep / 10.,
                                                 pstep)):
                    for k, f in enumerate(np.arange(fmin, fmax + fstep / 10.,
                                                    fstep)):
                        _sum = 0j
                        for l in np.arange(len(geometry)):
                            _sum += np.exp(complex(0., (geometry[l, 0] * sx +
                                                        geometry[l, 1] * sy) *
                                                   2 * np.pi * f))
                        buff[k] = abs(_sum) ** 2
                    transff[i, j] = cumtrapz(buff, dx=fstep)[-1]

        transff /= transff.max()
        return transff

    def _beamforming(self, stream, sll_x, slm_x, sll_y, slm_y, sl_s, frqlow,
                     frqhigh, stime, etime, win_len=-1, win_frac=0.5,
                     verbose=False, timestamp='mlabday',
                     method="DLS", nthroot=1, store=None,
                     correct_3dplane=False, static3d=False, vel_cor=4.):
        """
        Method for Delay and Sum/Phase Weighted Stack/Whitened Slowness Power

        :param stream: Stream object.
        :param sll_x: slowness x min (lower)
        :param slm_x: slowness x max
        :param sll_y: slowness y min (lower)
        :param slm_y: slowness y max
        :param sl_s: slowness step
        :type stime: UTCDateTime
        :param stime: Starttime of interest
        :type etime: UTCDateTime
        :param etime: Endtime of interest
        :param win_len: length for sliding window analysis, default is -1
         which means the whole trace;
        :param win_frac of win_len which is used to 'hop' forward in time
        :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec'
         returns the timestamp in secons since 1970-01-01T00:00:00,
         'mlabday' returns the timestamp in days (decimals represent hours,
         minutes and seconds) since '0001-01-01T00:00:00' as needed for
         matplotlib date plotting (see e.g. matplotlibs num2date).
        :param method: the method to use "DLS" delay and sum; "PWS" phase
         weighted stack; "SWP" slowness weightend power spectrum
        :param nthroot: nth-root processing; nth gives the root (1,2,3,4),
         default 1 (no nth-root)
        :type store: function
        :param store: A custom function which gets called on each iteration. It
         is called with the relative power map and the time offset as first and
         second arguments and the iteration number as third argument. Useful
         for storing or plotting the map for each iteration.
        :param correct_3dplane: if Yes than a best (LSQ) plane will be fitted
         into the array geometry. Mainly used with small apature arrays at
         steep flanks.
        :param static3d: if yes the station height of am array station is
         taken into account according to the formula:
            tj = -xj*sxj - yj*syj + zj*cos(inc)/vel_cor
         the inc angle is slowness dependend and thus must
         be estimated for each grid-point:
            inc = asin(v_cor*slow)
        :param vel_cor: Velocity for the upper layer (static correction)
         in km/s.
        :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
         backazimut, slowness, maximum beam (for DLS)
        """
        res = []
        eotr = True

        # check that sampling rates do not vary
        fs = stream[0].stats.sampling_rate
        nstat = len(stream)
        if len(stream) != len(stream.select(sampling_rate=fs)):
            msg = 'in sonic sampling rates of traces in stream are not equal'
            raise ValueError(msg)

        # loop with a sliding window over the dat trace array and apply bbfk

        grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
        grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

        abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
        geometry = self._geometry_dict_to_array(self.get_geometry_xyz(
            correct_3dplane=correct_3dplane,
            **self.center_of_gravity))

        if verbose:
            print("geometry:")
            print(geometry)
            print("stream contains following traces:")
            print(stream)
            print("stime = " + str(stime) + ", etime = " + str(etime))

        time_shift_table = self.get_timeshift(sll_x, sll_y, sl_s,
                                              grdpts_x, grdpts_y,
                                              vel_cor=vel_cor,
                                              static3d=static3d)

        mini = np.min(time_shift_table[:, :, :])
        maxi = np.max(time_shift_table[:, :, :])
        spoint, _epoint = self.get_stream_offsets(stream, (stime - mini),
                                                  (etime - maxi))

        # recalculate the maximum possible trace length
        #    ndat = int(((etime-maxi) - (stime-mini))*fs)
        if win_len < 0:
            nsamp = int(((etime - maxi) - (stime - mini)) * fs)
        else:
            # nsamp = int((win_len-np.abs(maxi)-np.abs(mini)) * fs)
            nsamp = int(win_len * fs)

        if nsamp <= 0:
            print('Data window too small for slowness grid')
            print('Must exit')
            quit()

        nstep = int(nsamp * win_frac)

        stream.detrend()
        newstart = stime
        offset = 0
        count = 0
        while eotr:
            max_beam = 0.
            if method == 'DLS':
                for x in range(grdpts_x):
                    for y in range(grdpts_y):
                        singlet = 0.
                        beam = np.zeros(nsamp, dtype='f8')
                        for i in range(nstat):
                            s = spoint[i] + int(
                                time_shift_table[i, x, y] * fs + 0.5)
                            try:
                                shifted = stream[i].data[s + offset:
                                                         s + nsamp + offset]
                                if len(shifted) < nsamp:
                                    shifted = np.pad(
                                        shifted, (0, nsamp - len(shifted)),
                                        'constant', constant_values=(0, 1))
                                singlet += 1. / nstat * np.sum(shifted *
                                                               shifted)
                                beam += 1. / nstat * np.power(
                                    np.abs(shifted), 1. / nthroot) * \
                                    shifted / np.abs(shifted)
                            except IndexError:
                                break
                        beam = np.power(np.abs(beam), nthroot) * \
                            beam / np.abs(beam)
                        bs = np.sum(beam*beam)
                        abspow_map[x, y] = bs / singlet
                        if abspow_map[x, y] > max_beam:
                            max_beam = abspow_map[x, y]
                            beam_max = beam
            if method == 'PWS':
                for x in range(grdpts_x):
                    for y in range(grdpts_y):
                        singlet = 0.
                        beam = np.zeros(nsamp, dtype='f8')
                        stack = np.zeros(nsamp, dtype='c8')
                        for i in range(nstat):
                            s = spoint[i] + int(time_shift_table[i, x, y] *
                                                fs + 0.5)
                            try:
                                shifted = sp.signal.hilbert(stream[i].data[
                                    s + offset: s + nsamp + offset])
                                if len(shifted) < nsamp:
                                    shifted = np.pad(
                                        shifted, (0, nsamp - len(shifted)),
                                        'constant', constant_values=(0, 1))
                            except IndexError:
                                break
                            phase = np.arctan2(shifted.imag, shifted.real)
                            stack.real += np.cos(phase)
                            stack.imag += np.sin(phase)
                        coh = 1. / nstat * np.abs(stack)
                        for i in range(nstat):
                            s = spoint[i] + int(
                                time_shift_table[i, x, y] * fs + 0.5)
                            shifted = stream[i].data[
                                s + offset: s + nsamp + offset]
                            singlet += 1. / nstat * np.sum(shifted * shifted)
                            beam += 1. / nstat * shifted * np.power(coh,
                                                                    nthroot)
                        bs = np.sum(beam * beam)
                        abspow_map[x, y] = bs / singlet
                        if abspow_map[x, y] > max_beam:
                            max_beam = abspow_map[x, y]
                            beam_max = beam
            if method == 'SWP':
                # generate plan for rfftr
                nfft = next_pow_2(nsamp)
                deltaf = fs / float(nfft)
                nlow = int(frqlow / float(deltaf) + 0.5)
                nhigh = int(frqhigh / float(deltaf) + 0.5)
                nlow = max(1, nlow)  # avoid using the offset
                nhigh = min(nfft / 2 - 1, nhigh)  # avoid using nyquist
                nf = nhigh - nlow + 1  # include upper and lower frequency

                steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16')
                spec = np.zeros((nstat, nf), dtype='c16')
                time_shift_table *= -1.
                clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                                     deltaf, time_shift_table, steer)
                try:
                    for i in range(nstat):
                        dat = stream[i].data[spoint[i] + offset:
                                             spoint[i] + offset + nsamp]

                        tap = cosine_taper(nsamp, p=0.22)
                        dat = (dat - dat.mean()) * tap
                        spec[i, :] = np.fft.rfft(dat, nfft)[nlow: nlow + nf]
                except IndexError:
                    break

                for i in range(grdpts_x):
                    for j in range(grdpts_y):
                        for k in range(nf):
                            for l in range(nstat):
                                steer[k, i, j, l] *= spec[l, k]

                beam = np.absolute(np.sum(steer, axis=3))
                less = np.max(beam, axis=1)
                max_buffer = np.max(less, axis=1)

                for i in range(grdpts_x):
                    for j in range(grdpts_y):
                        abspow_map[i, j] = np.sum(beam[:, i, j] /
                                                  max_buffer[:],
                                                  axis=0) / float(nf)

                beam_max = stream[0].data[spoint[0] + offset:
                                          spoint[0] + nsamp + offset]

            ix, iy = np.unravel_index(abspow_map.argmax(), abspow_map.shape)
            abspow = abspow_map[ix, iy]
            if store is not None:
                store(abspow_map, beam_max, count)
            count += 1
            print(count)
            # here we compute baz, slow
            slow_x = sll_x + ix * sl_s
            slow_y = sll_y + iy * sl_s

            slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
            if slow < 1e-8:
                slow = 1e-8
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
            baz = azimut % -360 + 180
            res.append(np.array([newstart.timestamp, abspow, baz, slow_x,
                                 slow_y, slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
            if (newstart + (nsamp + nstep) / fs) > etime:
                eotr = False
            offset += nstep

            newstart += nstep / fs
        res = np.array(res)
        if timestamp == 'julsec':
            pass
        elif timestamp == 'mlabday':
            # 719162 == hours between 1970 and 0001
            res[:, 0] = res[:, 0] / (24. * 3600) + 719162
        else:
            msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
            raise ValueError(msg)
        return np.array(res)
    #    return(baz,slow,slow_x,slow_y,abspow_map,beam_max)

    def _vespagram_baz(self, stream, time_shift_table, starttime, endtime,
                       method="DLS", nthroot=1):
        """
        Estimating the azimuth or slowness vespagram

        :param stream: Stream object.
            items/attributes. See param coordsys
        :type starttime: UTCDateTime
        :param starttime: Starttime of interest
        :type endtime: UTCDateTime
        :param endtime: Endtime of interest
        :return: numpy.ndarray of beams with different slownesses
        """
        fs = stream[0].stats.sampling_rate

        mini = min(min(i.values()) for i in list(time_shift_table.values()))
        maxi = max(max(i.values()) for i in list(time_shift_table.values()))
        spoint, _ = self.get_stream_offsets(stream, (starttime - mini),
                                            (endtime - maxi))

        # Recalculate the maximum possible trace length
        ndat = int(((endtime - maxi) - (starttime - mini)) * fs)
        beams = np.zeros((len(time_shift_table), ndat), dtype='f8')

        max_beam = 0.0
        slow = 0.0

        slownesses = sorted(time_shift_table.keys())
        sll = slownesses[0]
        sls = slownesses[1] - sll

        for _i, slowness in enumerate(time_shift_table.keys()):
            singlet = 0.0
            if method == 'DLS':
                for _j, tr in enumerate(stream.traces):
                    station = tr.id
                    s = spoint[_j] + int(time_shift_table[slowness][station] *
                                         fs + 0.5)
                    shifted = tr.data[s: s + ndat]
                    singlet += 1. / len(stream) * np.sum(shifted * shifted)
                    beams[_i] += 1. / len(stream) * np.power(np.abs(shifted),
                                                             1. / nthroot) * \
                        shifted / np.abs(shifted)

                beams[_i] = np.power(np.abs(beams[_i]), nthroot) * \
                    beams[_i] / np.abs(beams[_i])

                bs = np.sum(beams[_i] * beams[_i])
                bs /= singlet

                if bs > max_beam:
                    max_beam = bs
                    beam_max = _i
                    slow = np.abs(sll + slowness * sls)

            elif method == 'PWS':
                stack = np.zeros(ndat, dtype='c8')
                nstat = len(stream)
                for i in range(nstat):
                    s = spoint[i] + int(time_shift_table[i, _i] * fs + 0.5)
                    try:
                        shifted = sp.signal.hilbert(stream[i].data[s:s + ndat])
                    except IndexError:
                        break
                    phase = np.arctan2(shifted.imag, shifted.real)
                    stack.real += np.cos(phase)
                    stack.imag += np.sin(phase)
                coh = 1. / nstat * np.abs(stack)
                for i in range(nstat):
                    s = spoint[i] + int(time_shift_table[i, _i] * fs + 0.5)
                    shifted = stream[i].data[s: s + ndat]
                    singlet += 1. / nstat * np.sum(shifted * shifted)
                    beams[_i] += 1. / nstat * shifted * np.power(coh, nthroot)
                bs = np.sum(beams[_i] * beams[_i])
                bs = bs / singlet
                if bs > max_beam:
                    max_beam = bs
                    beam_max = _i
                    slow = np.abs(sll + _i * sls)
                    if slow < 1e-8:
                        slow = 1e-8
            else:
                msg = "Method '%s' unknown." % method
                raise ValueError(msg)

        return slow, beams, beam_max, max_beam

    @staticmethod
    def _geometry_dict_to_array(geometry):
        """
        Take a geometry dictionary (as provided by self.geometry, or by
        get_geometry_xyz) and convert to a numpy array, as used in some
        methods.
        """
        geom_array = np.empty((len(geometry), 3))
        try:
            for _i, (key, value) in enumerate(sorted(list(geometry.items()))):
                geom_array[_i, 0] = value["x"]
                geom_array[_i, 1] = value["y"]
                geom_array[_i, 2] = value["z"]
        except KeyError:
            for _i, (key, value) in enumerate(sorted(list(geometry.items()))):
                geom_array[_i, 0] = float(value["latitude"])
                geom_array[_i, 1] = float(value["longitude"])
                geom_array[_i, 2] = value["absolute_height_in_km"]
        return geom_array

    def correct_with_3dplane(self, geometry):
        """
        Correct a given array geometry with a best-fitting plane.

        :type geometry: dict
        :param geometry: Nested dictionary of stations, as returned for example
            by :attr:`geometry` or :meth:`get_geometry_xyz`.
        :return: The corrected geometry as dictionary, with the same keys as
            passed in.
        """
        # sort keys in the nested dict to be alphabetical:
        coord_sys_keys = sorted(list(geometry.items())[0][1].keys())
        if coord_sys_keys[0] == 'x':
            pass
        elif coord_sys_keys[0] == 'absolute_height_in_km':
            # set manually because order is important.
            coord_sys_keys = ['latitude', 'longitude', 'absolute_height_in_km']
        else:
            raise KeyError("Geometry dictionary does not have correct keys.")
        orig_geometry = geometry.copy()
        geometry = self._geometry_dict_to_array(geometry)
        a = geometry
        u, s, vh = np.linalg.linalg.svd(a)
        v = vh.conj().transpose()
        # satisfies the plane equation a*x + b*y + c*z = 0
        result = np.zeros((len(geometry), 3))
        # now we are seeking the station positions on that plane
        # geometry[:,2] += v[2,-1]
        n = v[:, -1]
        result[:, 0] = (geometry[:, 0] - n[0] * (
            n[0] * geometry[:, 0] + geometry[:, 1] * n[1] + n[2] *
            geometry[:, 2]) / (
                            n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
        result[:, 1] = (geometry[:, 1] - n[1] * (
            n[0] * geometry[:, 0] + geometry[:, 1] * n[1] + n[2] *
            geometry[:, 2]) / (
                            n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
        result[:, 2] = (geometry[:, 2] - n[2] * (
            n[0] * geometry[:, 0] + geometry[:, 1] * n[1] + n[2] *
            geometry[:, 2]) / (
                            n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
        geometry = result[:]
        # print("Best fitting plane-coordinates :\n", geometry)

        # convert geometry array back to a dictionary.
        geodict = {}
        # The sorted list is necessary to match the station IDs (the keys in
        # the geometry dict) to the correct array row (or column?), same as is
        # done in _geometry_dict_to_array, but backwards.
        for _i, (key, value) in enumerate(sorted(
                list(orig_geometry.items()))):
            geodict[key] = {coord_sys_keys[0]: geometry[_i, 0],
                            coord_sys_keys[1]: geometry[_i, 1],
                            coord_sys_keys[2]: geometry[_i, 2]}
        geometry = geodict
        return geometry

    @staticmethod
    def _plot_array_analysis(out, sllx, slmx, slly, slmy, sls,
                             filename_patterns, baz_plot, method,
                             st_workon, starttime, wlen, endtime):
        """
        Some plotting taken out from _array_analysis_helper. Can't do the array
        response overlay now though.
        :param baz_plot: Whether to show backazimuth-slowness map (True) or
         slowness x-y map (False).
        """
        import matplotlib.pyplot as plt
        trace = []
        t, rel_power, abs_power, baz, slow = out.T
        baz[baz < 0.0] += 360
        # now let's do the plotting
        cmap = cm.rainbow
        # we will plot everything in s/deg
        slow = degrees2kilometers(slow)
        sllx = degrees2kilometers(sllx)
        slmx = degrees2kilometers(slmx)
        slly = degrees2kilometers(slly)
        slmy = degrees2kilometers(slmy)
        sls = degrees2kilometers(sls)

        numslice = len(t)
        powmap = []

        slx = np.arange(sllx - sls, slmx, sls)
        sly = np.arange(slly - sls, slmy, sls)
        if baz_plot:
            maxslowg = np.sqrt(slmx * slmx + slmy * slmy)
            bzs = np.arctan2(sls, np.sqrt(
                slmx * slmx + slmy * slmy)) * 180 / np.pi
            xi = np.arange(0., maxslowg, sls)
            yi = np.arange(-180., 180., bzs)
            grid_x, grid_y = np.meshgrid(xi, yi)
        # reading in the rel-power maps
        for i in range(numslice):
            powmap.append(np.load(filename_patterns[0] % i))
            if method != 'FK':
                trace.append(np.load(filename_patterns[1] % i))
        # remove last item as a cludge to get plotting to work - not sure
        # it's always clever or just a kind of rounding or modulo problem
        if len(slx) == len(powmap[0][0]) + 1:
            slx = slx[:-1]
        if len(sly) == len(powmap[0][1]) + 1:
            sly = sly[:-1]

        npts = st_workon[0].stats.npts
        df = st_workon[0].stats.sampling_rate
        t = np.arange(0, npts / df, 1 / df)

        # if we choose windowlen > 0. we now move through our slices
        for i in range(numslice):
            st = UTCDateTime(t[i]) - starttime
            if wlen <= 0:
                en = endtime
            else:
                en = st + wlen
            print(UTCDateTime(t[i]))
            # add polar and colorbar axes
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_axes([0.1, 0.87, 0.7, 0.10])
            # here we plot the first trace on top of the slowness map
            # and indicate the possibiton of the lsiding window as green box
            if method == 'FK':
                ax1.plot(t, st_workon[0].data, 'k')
                if wlen > 0.:
                    try:
                        ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                    except IndexError:
                        pass
            else:
                t = np.arange(0, len(trace[i]) / df, 1 / df)
                ax1.plot(t, trace[i], 'k')

            ax1.yaxis.set_major_locator(MaxNLocator(3))

            ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])

            # if we have chosen the baz_plot option a re-griding
            # of the sx,sy slowness map is needed
            if baz_plot:
                slowgrid = []
                power = np.asarray(powmap[i])
                for ix, sx in enumerate(slx):
                    for iy, sy in enumerate(sly):
                        bbaz = np.arctan2(sx, sy) * 180 / np.pi + 180.
                        if bbaz > 180.:
                            bbaz = -180. + (bbaz - 180.)
                        slowgrid.append((np.sqrt(sx * sx + sy * sy), bbaz,
                                         power[ix, iy]))

                slowgrid = np.asarray(slowgrid)
                sl = slowgrid[:, 0]
                bz = slowgrid[:, 1]
                slowg = slowgrid[:, 2]
                grid = interpolate.griddata((sl, bz), slowg,
                                            (grid_x, grid_y),
                                            method='nearest')
                ax.pcolormesh(xi, yi, grid, cmap=cmap)

                ax.set_xlabel('slowness [s/deg]')
                ax.set_ylabel('backazimuth [deg]')
                ax.set_xlim(xi[0], xi[-1])
                ax.set_ylim(yi[0], yi[-1])
            else:
                ax.set_xlabel('slowness [s/deg]')
                ax.set_ylabel('slowness [s/deg]')
                slow_x = np.cos((baz[i] + 180.) * np.pi / 180.) * slow[i]
                slow_y = np.sin((baz[i] + 180.) * np.pi / 180.) * slow[i]
                ax.pcolormesh(slx, sly, powmap[i].T)
                ax.arrow(0, 0, slow_y, slow_x, head_width=0.005,
                         head_length=0.01, fc='k', ec='k')
                ax.set_ylim(slx[0], slx[-1])
                ax.set_xlim(sly[0], sly[-1])
            new_time = t[i]

            result = "BAZ: %.2f, Slow: %.2f s/deg, Time %s" % (
                baz[i], slow[i], UTCDateTime(new_time))
            ax.set_title(result)
            plt.show()


class BeamformerResult(object):
    """
    Contains results from beamforming and attached plotting methods.

    This class is an attempt to standardise the output of the various
    beamforming algorithms in :class:`SeismicArray`, and provide a range of
    plotting options. To that end, the raw output from the beamformers is
    combined with metadata to identify what method produced it with which
    parameters, including: the frequency and slowness ranges, time frame and
    the inventory defining the array (this will only include the stations for
    which data were actually available during the considered time frame).

    .. rubric:: Data transformation

    The different beamforming algorithms produce output in different
    formats, so it needs to be standardised. The
    :meth:`slowness_whitened_power`, :meth:`phase_weighted_stack`,
    :meth:`delay_and_sum`, and :meth:`fk_analysis` routines internally
    perform grid searches of slownesses in x and y directions, producing
    numpy arrays of power for every slowness, time window and discrete
    frequency. Only the maximum relative and absolute powers of each time
    window are returned, as well as the slowness values at which they were
    found. The x-y slowness values are also converted to radial slowness and
    backazimuth. Returning the complete data arrays for these methods is not
    currently implemented.

    Working somewhat differently, :meth:`three_component_beamforming` returns
    a four-dimensional numpy array of relative powers for every backazimuth,
    slowness, time window and discrete frequency, but no absolute powers. This
    data allows the creation of the same plots as the previous methods, as the
    maximum powers is easily calulated from the full power array.

    .. rubric:: Concatenating results

    To allow for the creation of beamformer output plots over long
    timescales where the beamforming might performed not all at once,
    the results may be concatenated (added). This is done by simple adding
    syntax:

    >>> long_results = beamresult_1 + beamresult_2 # doctest:+SKIP

    Of course, this only makes sense if the two added result objects are
    consistent. The :meth:`__add__` method checks that they were created by
    the same beamforming method, with the same slowness and frequency
    ranges. Only then are the data combined.

    :param inventory: The inventory that was actually used in the beamforming.
    :param win_starttimes: Start times of the beamforming windows.
    :type win_starttimes: numpy array of :class:`obspy.core.utcdatetime.UTCDateTime`
    :param slowness_range: The slowness range used for the beamforming.
    :param max_rel_power: Maximum relative power at every timestep.
    :param max_abs_power: Maximum absolute power at every timestep.
    :param max_pow_baz: Backazimuth of the maximum power value at every
     timestep.
    :param max_pow_slow: Slowness of the maximum power value at every timestep.
    :param full_beamres: 4D numpy array holding relative power results for
     every backazimuth, slowness, window and discrete frequency (in that
     order).
    :param freqs: The discrete frequencies used for which the full_beamres
     was computed.
    :param incidence: Rayleigh wave incidence angles (only from three
     component beamforming).
    :param method: Method used for the beamforming.
    """

    def __init__(self, inventory, win_starttimes, slowness_range,
                 max_rel_power=None, max_abs_power=None, max_pow_baz=None,
                 max_pow_slow=None, full_beamres=None, freqs=None,
                 incidence=None, method=None, timestep=None):

        self.inventory = copy.deepcopy(inventory)
        self.win_starttimes = win_starttimes
        self.starttime = win_starttimes[0]
        if timestep is not None:
            self.timestep = timestep
        elif timestep is None and len(win_starttimes) == 1:
            msg = "Can't calculate a timestep. Please set manually."
            warnings.warn(msg)
            self.timestep = None
        else:
            self.timestep = win_starttimes[1] - win_starttimes[0]
        if self.timestep is not None:
            # Don't use unjustified higher precision.
            try:
                self.endtime = UTCDateTime(win_starttimes[-1] + self.timestep,
                                           precision=win_starttimes[-1].
                                           precision)
            except AttributeError:
                self.endtime = UTCDateTime(win_starttimes[-1] + self.timestep)
        self.max_rel_power = max_rel_power
        if max_rel_power is not None:
            self.max_rel_power = self.max_rel_power.astype(float)
        self.max_abs_power = max_abs_power
        if max_abs_power is not None:
            self.max_abs_power = self.max_abs_power.astype(float)
        self.max_pow_baz = max_pow_baz
        if max_pow_baz is not None:
            self.max_pow_baz = self.max_pow_baz.astype(float)
        self.max_pow_slow = max_pow_slow
        if self.max_pow_slow is not None:
            self.max_pow_slow = self.max_pow_slow.astype(float)
        if len(slowness_range) == 1:
            raise ValueError("Need at least two slowness values.")
        self.slowness_range = slowness_range.astype(float)
        self.freqs = freqs
        self.incidence = incidence
        self.method = method

        if full_beamres is not None and full_beamres.ndim != 4:
            raise ValueError("Full beamresults should be 4D array.")
        self.full_beamres = full_beamres
        # FK and other 1cbf return max relative (and absolute) powers,
        # as well as the slowness and azimuth where appropriate. 3cbf as of
        # now returns the whole results.
        if(max_rel_power is None and max_pow_baz is None and
                max_pow_slow is None and full_beamres is not None):
            self._calc_max_values()

    def __add__(self, other):
        """
        Add two sequential BeamformerResult instances. Must have been created
        with identical frequency and slowness ranges.
        """
        if not isinstance(other, BeamformerResult):
            raise TypeError('unsupported operand types')
        if self.method != other.method:
            raise ValueError('Methods must be equal.')
        if self.freqs is None or other.freqs is None:
            attrs = ['slowness_range']
        else:
            attrs = ['freqs', 'slowness_range']
        if any((self.__dict__[attr] != other.__dict__[attr])
               for attr in attrs):
            raise ValueError('Frequency and slowness range parameters must be '
                             'equal.')
        times = np.append(self.win_starttimes, other.win_starttimes)
        if self.full_beamres is not None and other.full_beamres is not None:
            full_beamres = np.append(self.full_beamres,
                                     other.full_beamres, axis=2)
            max_rel_power, max_pow_baz, \
                max_pow_slow, max_abs_power = None, None, None, None
        else:
            full_beamres = None
            max_rel_power = np.append(self.max_rel_power, other.max_rel_power)
            max_pow_baz = np.append(self.max_pow_baz, other.max_pow_baz)
            max_pow_slow = np.append(self.max_pow_slow, other.max_pow_slow)
            if self.max_abs_power is not None:
                max_abs_power = np.append(self.max_abs_power,
                                          other.max_abs_power)
            else:
                max_abs_power = None

        out = self.__class__(self.inventory, times,
                             full_beamres=full_beamres,
                             slowness_range=self.slowness_range,
                             freqs=self.freqs,
                             method=self.method,
                             max_abs_power=max_abs_power,
                             max_rel_power=max_rel_power,
                             max_pow_baz=max_pow_baz,
                             max_pow_slow=max_pow_slow)
        return out

    def _calc_max_values(self):
        """
        If the maximum power etc. values are unset, but the full results are
        available (currently only from :meth:`three_component_beamforming`),
        calculate the former from the latter.
        """
        # Average over all frequencies.
        freqavg = self.full_beamres.mean(axis=3)
        num_win = self.full_beamres.shape[2]
        self.max_rel_power = np.empty(num_win, dtype=float)
        self.max_pow_baz = np.empty_like(self.max_rel_power)
        self.max_pow_slow = np.empty_like(self.max_rel_power)
        for win in range(num_win):
            self.max_rel_power[win] = freqavg[:, :, win].max()
            ibaz = np.where(freqavg[:, :, win] == freqavg[:, :, win].max())[0]
            islow = np.where(freqavg[:, :, win] == freqavg[:, :, win].max())[1]
            # Add [0] in case of multiple matches.
            self.max_pow_baz[win] = np.arange(0, 362, 2)[ibaz[0]]
            self.max_pow_slow[win] = self.slowness_range[islow[0]]

    def _get_plotting_timestamps(self, extended=False):
        """
        Convert the times to the time reference matplotlib uses and return as
        timestamps. Returns the timestamps in days (decimals represent hours,
        minutes and seconds) since '0001-01-01T00:00:00' as needed for
        matplotlib date plotting (see e.g. matplotlibs num2date).
        """
        if extended:
            # With pcolormesh, will miss one window if only plotting window
            # start times.
            plot_times = list(self.win_starttimes)
            plot_times.append(self.endtime)
        else:
            plot_times = self.win_starttimes
        # Honestly, this is black magic to me.
        newtimes = np.array([t.timestamp / (24*3600) + 719163
                             for t in plot_times])
        return newtimes

    def plot_baz_hist(self, show=True):
        """
        Plot a backazimuth - slowness radial histogram.

        The backazimuth and slowness values of the maximum relative powers
        of each beamforming window are counted into bins defined
        by slowness and backazimuth, weighted by the power.

        :param show: Whether to call plt.show() immediately.
        """
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt
        cmap = cm.hot_r
        # Can't plot negative slownesses:
        sll = abs(self.slowness_range).min()
        slm = self.slowness_range.max()

        # choose number of azimuth bins in plot
        # (desirably 360 degree/azimuth_bins is an integer!)
        azimuth_bins = 36
        # number of slowness bins
        slowness_bins = len(self.slowness_range)
        # Plot is not too readable beyond a certain number of bins.
        slowness_bins = 30 if slowness_bins > 30 else slowness_bins
        abins = np.arange(azimuth_bins + 1) * 360. / azimuth_bins
        sbins = np.linspace(sll, slm, slowness_bins + 1)

        # sum rel power in bins given by abins and sbins
        hist, baz_edges, sl_edges = \
            np.histogram2d(self.max_pow_baz, self.max_pow_slow,
                           bins=[abins, sbins], weights=self.max_rel_power)

        # transform to radian
        baz_edges = np.radians(baz_edges)

        # add polar and colorbar axes
        fig = plt.figure(figsize=(8, 8))
        cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        dh = abs(sl_edges[1] - sl_edges[0])
        dw = abs(baz_edges[1] - baz_edges[0])

        # circle through backazimuth
        for i, row in enumerate(hist):
            ax.bar(left=(i * dw) * np.ones(slowness_bins),
                   height=dh * np.ones(slowness_bins),
                   width=dw, bottom=dh * np.arange(slowness_bins),
                   color=cmap(row / hist.max()))

        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])

        # set slowness limits
        ax.set_ylim(sll, slm)
        ColorbarBase(cax, cmap=cmap,
                     norm=Normalize(vmin=hist.min(), vmax=hist.max()))
        plt.suptitle('{} beamforming: results from \n{} to {}'
                     .format(self.method, self.starttime,
                             self.endtime))
        if show is True:
            plt.show()

    def plot_bf_results_over_time(self, show=True):
        """
        Plot beamforming results over time, with the relative power as
        colorscale.

        :param show: Whether to call plt.show() immediately.
        """
        import matplotlib.pyplot as plt
        labels = ['Rel. Power', 'Abs. Power', 'Backazimuth', 'Slowness']
        datas = [self.max_rel_power, self.max_abs_power,
                 self.max_pow_baz, self.max_pow_slow]
        # To account for e.g. the _beamforming method not returning absolute
        # powers:
        for data, lab in zip(reversed(datas), reversed(labels)):
            if data is None:
                datas.remove(data)
                labels.remove(lab)

        xlocator = mdates.AutoDateLocator(interval_multiples=True)
        ymajorlocator = MultipleLocator(90)

        fig = plt.figure()
        for i, (data, lab) in enumerate(zip(datas, labels)):
            ax = fig.add_subplot(len(labels), 1, i + 1)
            ax.scatter(self._get_plotting_timestamps(), data,
                       c=self.max_rel_power, alpha=0.6, edgecolors='none')
            ax.set_ylabel(lab)
            timemargin = 0.05 * (self._get_plotting_timestamps()[-1] -
                                 self._get_plotting_timestamps()[0])
            ax.set_xlim(self._get_plotting_timestamps()[0] - timemargin,
                        self._get_plotting_timestamps()[-1] + timemargin)
            if lab == 'Backazimuth':
                ax.set_ylim(0, 360)
                ax.yaxis.set_major_locator(ymajorlocator)
            else:
                datamargin = 0.05 * (data.max() - data.min())
                ax.set_ylim(data.min() - datamargin, data.max() + datamargin)
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

        fig.suptitle('{} beamforming: results from \n{} to {}'
                     .format(self.method, self.starttime,
                             self.endtime))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.9, right=0.95, bottom=0.2,
                            hspace=0.1)
        if show is True:
            plt.show()

    def plot_power(self, show=True):
        """
        Plot relative power as a function of backazimuth and time, like a
        Vespagram.

        Requires full 4D results, at the moment only provided by
        :meth:`three_component_beamforming`.

        :param show: Whether to call plt.show() immediately.
        """
        import matplotlib.pyplot as plt
        if self.full_beamres is None:
            raise ValueError('Insufficient data. Try other plotting options.')
        # Prepare data.
        freqavg = self.full_beamres.mean(axis=3)
        num_win = self.full_beamres.shape[2]
        # This is 2D, with time windows and baz (in this order)
        # as indices.
        maxazipows = np.array([[azipows.T[t].max() for azipows in freqavg]
                               for t in range(num_win)])
        azis = np.arange(0, 362, 2)
        labels = ['baz']
        maskedazipows = np.ma.array(maxazipows, mask=np.isnan(maxazipows))
        # todo (maybe) implement plotting of slowness map corresponding to the
        # max powers
        datas = [maskedazipows]  # , maskedslow]

        xlocator = mdates.AutoDateLocator(interval_multiples=True)
        ymajorlocator = MultipleLocator(90)
        fig = plt.figure()
        for i, (data, lab) in enumerate(zip(datas, labels)):
            ax = fig.add_subplot(len(labels), 1, i + 1)

            pc = ax.pcolormesh(self._get_plotting_timestamps(extended=True),
                               azis, data.T, cmap=cm.get_cmap('hot_r'),
                               rasterized=True)
            timemargin = 0.05 * (self._get_plotting_timestamps(extended=True
                                                               )[-1] -
                                 self._get_plotting_timestamps()[0])
            ax.set_xlim(self._get_plotting_timestamps()[0] - timemargin,
                        self._get_plotting_timestamps(extended=True)[-1] +
                        timemargin)
            ax.set_ylim(0, 360)
            ax.yaxis.set_major_locator(ymajorlocator)
            cbar = fig.colorbar(pc)
            cbar.solids.set_rasterized(True)
            ax.set_ylabel('Backazimuth')
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

        fig.suptitle('{} beamforming: results from \n{} to {}'
                     .format(self.method, self.starttime,
                             self.endtime))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.9, right=0.95, bottom=0.2,
                            hspace=0.1)
        if show is True:
            plt.show()

    def plot_bf_plots(self, average_windows=True, average_freqs=True,
                      plot_frequencies=None, show=True):
        """
        Plot beamforming results as individual polar plots of relative power as
        function of backazimuth and slowness.

        Can plot results averaged over windows/frequencies, results for each
        window and every frequency individually, or for selected frequencies
        only.

        :param average_windows: Whether to plot an average of results over all
         windows.
        :param average_freqs: Whether to plot an average of results over all
         frequencies.
        :param plot_frequencies: List of discrete frequencies for which windows
         should be plotted, if not plotting an average of frequencies (ignored
         if average_freqs is True).
        :param show: Whether to call plt.show() immediately.
        """
        import matplotlib.pyplot as plt
        if self.full_beamres is None:
            raise ValueError('Insufficient data for this plotting method.')
        if average_freqs is True and plot_frequencies is not None:
            warnings.warn("Ignoring plot_frequencies, only plotting an average"
                          " of all frequencies.")
        if(type(plot_frequencies) is not list and
           type(plot_frequencies) is not tuple and
           plot_frequencies is not None):
            plot_frequencies = [plot_frequencies]

        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.

        def _actual_plotting(bfres, title):
            """
            Pass in a 2D bfres array of beamforming results with
            averaged or selected windows and frequencies.
            """
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            cmap = cm.hot_r
            contf = ax.contourf(theo_backazi, self.slowness_range, bfres.T,
                                100, cmap=cmap, antialiased=True,
                                linstyles='dotted')
            ax.contour(theo_backazi, self.slowness_range, bfres.T, 100,
                       cmap=cmap)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_rmax(self.slowness_range[-1])
            # Setting this to -0 means that if the slowness range doesn't
            # start at 0, the plot is shown with a 'hole' in the middle
            # rather than stitching it up. This behaviour shows the
            # resulting shapes much better.
            ax.set_rmin(-0)
            fig.colorbar(contf)
            ax.grid(True)
            ax.set_title(title)

        beamresnz = self.full_beamres
        if average_windows:
            beamresnz = beamresnz.mean(axis=2)
        if average_freqs:
            # Always an average over the last axis, whether or not windows
            # were averaged.
            beamresnz = beamresnz.mean(axis=beamresnz.ndim - 1)

        if average_windows and average_freqs:
            _actual_plotting(beamresnz,
                             '{} beamforming result, averaged over all time '
                             'windows\n ({} to {}) and frequencies.'
                             .format(self.method, self.starttime,
                                     self.endtime))

        if average_windows and not average_freqs:
            if plot_frequencies is None:
                warnings.warn('No frequency specified for plotting.')
            else:
                for plot_freq in plot_frequencies:
                    # works because freqs is a range
                    ifreq = np.searchsorted(self.freqs, plot_freq)
                    _actual_plotting(beamresnz[:, :, ifreq],
                                     '{} beamforming result, averaged over all'
                                     ' time windows\n for frequency {} Hz.'
                                     .format(self.method, self.freqs[ifreq]))

        if average_freqs and not average_windows:
            for iwin in range(len(beamresnz[0, 0, :])):
                _actual_plotting(beamresnz[:, :, iwin],
                                 '{} beamforming result, averaged over all '
                                 'frequencies,\n for window {} '
                                 '(starting {})'
                                 .format(self.method, iwin,
                                         self.win_starttimes[iwin]))

        # Plotting all windows, selected frequencies.
        if average_freqs is False and average_windows is False:
            if plot_frequencies is None:
                warnings.warn('No frequency specified for plotting.')
            else:
                for plot_freq in plot_frequencies:
                    ifreq = np.searchsorted(self.freqs, plot_freq)
                    for iwin in range(len(beamresnz[0, 0, :, 0])):
                        _actual_plotting(beamresnz[:, :, iwin, ifreq],
                                         '{} beamforming result, for frequency'
                                         ' {} Hz,\n, window {} (starting {}).'
                                         .format(self.method,
                                                 self.freqs[ifreq], iwin,
                                                 self.win_starttimes[iwin]))

        if show is True:
            plt.show()

    def plot_radial_transfer_function(self, plot_freqs=None):
        """
        Plot the radial transfer function of the array and slowness range used
        to produce this result.

        :param plot_freqs: List of discrete frequencies for which the transfer
         function should be plotted. Defaults to the minimum and maximum of the
         frequency range used in the generation of this results object.
        """
        if plot_freqs is None:
            plot_freqs = [self.freqs[0], self.freqs[-1]]
        if type(plot_freqs) is float or type(plot_freqs) is float:
            plot_freqs = [plot_freqs]

        # Need absolute values:
        absolute_slownesses = [abs(_) for _ in self.slowness_range]

        # Need to create an array object:
        plot_array = SeismicArray('plot_array', self.inventory)
        plot_array.plot_radial_transfer_function(min(absolute_slownesses),
                                                 max(absolute_slownesses),
                                                 self.slowness_range[1] -
                                                 self.slowness_range[0],
                                                 plot_freqs)

    def plot_transfer_function_freqslowness(self):
        """
        Plot an x-y transfer function of the array used to produce this
        result, as a function of the set slowness and frequency ranges.
        """
        # Need absolute values:
        absolute_slownesses = [abs(_) for _ in self.slowness_range]
        # Need to create an array object:
        arr = SeismicArray('plot_array', self.inventory)
        arr.plot_transfer_function_freqslowness(max(absolute_slownesses),
                                                self.slowness_range[1] -
                                                self.slowness_range[0],
                                                min(self.freqs),
                                                max(self.freqs),
                                                abs(self.freqs[1] -
                                                    self.freqs[0]))


if __name__ == '__main__':
    import doctest

    doctest.testmod(exclude_empty=True)
