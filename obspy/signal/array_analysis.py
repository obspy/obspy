#!/usr/bin/env python
"""
Function for Array Analysis


Coordinate conventions:

* Right handed
* X positive to east
* Y positive to north
* Z positive up

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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from obspy.core import UTCDateTime
from obspy.geodetics import gps2DistAzimuth, degrees2kilometers, \
    kilometer2degrees
from obspy.signal.util import utlGeoKm, nextpow2
from obspy.signal.headers import clibsignal
from obspy.core import Stream, Trace
from scipy.integrate import cumtrapz
from obspy.signal.invsim import cosTaper
from obspy.core.util import AttribDict
import warnings

import cartopy
import matplotlib.pyplot as plt


class SeismicArray(object):
    """
    Class representing a seismic array.
    """

    def __init__(self, name="", ):
        self.name = name
        self.inventory = None

    def add_inventory(self, inv):
        # todo add some commentary and type hint
        if self.inventory is not None:
            raise NotImplementedError("Already has an inventory attached.")
        self.inventory = inv

    def inventory_cull(self, st):
        """
        From the array inventory permanently remove all entries for stations
        that do not have traces in given stream st. Useful e.g. for beamforming
        applications where self.geometry would return geometry for more
        stations than are actually present in the data.
        """
        inv = self.inventory
        # check what station/channel IDs are in the data
        stations_present = list(set(tr.getId() for tr in st))
        # delete all channels that are not represented
        for k, netw in reversed(list(enumerate(inv.networks))):
            for j, stn in reversed(list(enumerate(netw.stations))):
                for i, cha in reversed(list(enumerate(stn.channels))):
                    if ("{}.{}.{}.{}".format(netw.code, stn.code,
                                             cha.location_code, cha.code)
                       not in stations_present):
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
        Return a dictionary of lat, lon, height values for each item (station
        or channel level, if available) in the array inventory.
        Item codes are SEED ID strings.
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
        lats, lngs, hgts = self.__coordinate_values()
        return {
            "latitude": np.mean(lats),
            "longitude": np.mean(lngs),
            "absolute_height_in_km": np.mean(hgts)}

    @property
    def geometry(self):
        """
        A dictionary of latitude, longitude and absolute height [km] values
        for each station in the array inventory.
        """
        return self._get_geometry()

    @property
    def aperture(self):
        """
        The aperture of the array in kilometers.
        """
        distances = []
        geo = self.geometry
        # todo: add a unit test and remove the list()
        for location, coordinates in list(geo.items()):
            for other_location, other_coordinates in list(geo.items()):
                if location == other_location:
                    continue
                distances.append(gps2DistAzimuth(
                    coordinates["latitude"], coordinates["longitude"],
                    other_coordinates["latitude"],
                    other_coordinates["longitude"])[0] / 1000.0)

        return max(distances)

    @property
    def extent(self):
        lats, lngs, hgt = self.__coordinate_values()

        return {
            "min_latitude": min(lats),
            "max_latitude": max(lats),
            "min_longitude": min(lngs),
            "max_longitude": max(lngs),
            "min_absolute_height_in_km": min(hgt),
            "max_absolute_height_in_km": max(hgt)}

    def __coordinate_values(self):
        geo = self.geometry
        lats, lngs, hgt = [], [], []
        for coordinates in list(geo.values()):
            lats.append(coordinates["latitude"]),
            lngs.append(coordinates["longitude"]),
            hgt.append(coordinates["absolute_height_in_km"])
        return lats, lngs, hgt

    def __str__(self):
        """
        Pretty representation of the array.
        """
        ret_str = "Seismic Array '{name}'\n".format(name=self.name)
        ret_str += "\t{count} Stations\n".format(count=len(self.geometry))
        ret_str += "\tAperture: {aperture:.2f} km".format(
            aperture=self.aperture)
        return ret_str

    def get_geometry_xyz(self, latitude, longitude, absolute_height_in_km,
                         correct_3dplane=False):
        """
        Method to calculate the array geometry and each station's offset
        in km relative to a given reference (centre) point.
        Example: To obtain it in relation to the center of gravity, use
        self.get_geometry_xyz(**self.center_of_gravity).

        :param latitude: Latitude of reference origin
        :param longitude: Longitude of reference origin
        :param absolute_height_in_km: Elevation of reference origin
        :param correct_3dplane: Corrects the returned geometry by a
               best-fitting 3D plane.
               This might be important if the array is located on an inclined
               slope (e.g., at a volcano).
        :return: Returns the geometry of the stations as dictionary.
        """
        geometry = {}
        for key, value in list(self.geometry.items()):
            x, y = utlGeoKm(longitude, latitude, value["longitude"],
                            value["latitude"])
            geometry[key] = {
                "x": x,
                "y": y,
                "z": absolute_height_in_km - value["absolute_height_in_km"]
            }
        if correct_3dplane:
            correct_with_3dplane(geometry)
        return geometry

    def find_closest_station(self, latitude, longitude,
                             absolute_height_in_km=0.0):
        min_distance = None
        min_distance_station = None

        for key, value in list(self.geometry.items()):
            # output in [km]
            x, y = utlGeoKm(longitude, latitude, value["longitude"],
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
                          static_3D=False, vel_cor=4.0):
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
        :param static_3D: a correction of the station height is applied using
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

                if static_3D:
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
                      vel_cor=4., static_3D=False):
        """
        Returns timeshift table for the geometry of the current array, in
        kilometres relative to a given centre (uses geometric centre if not
        specified).

        :param sllx: slowness x min (lower)
        :param slly: slowness y min (lower)
        :param sls: slowness step
        :param grdpts_x: number of grid points in x direction
        :param grdpts_x: number of grid points in y direction
        :param latitude: latitude of reference origin
        :param longitude: longitude of reference origin
        :param absolute_height: elevation of reference origin, in km
        :param vel_cor: correction velocity (upper layer) in km/s
        :param static_3D: a correction of the station height is applied using
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

        geometry = _geometry_dict_to_array(geom)

        if static_3D:
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
                        print(
                            "Warning correction velocity smaller than apparent"
                            " velocity")
                        inc = np.pi / 2.
                    time_shift_tbl[:, k, l] = sx * geometry[:, 0] + \
                                              sy * geometry[:, 1] + \
                                              geometry[:, 2] * np.cos(inc) \
                                              / vel_cor
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
                  endtime, latitude, longitude, absolute_height_in_km,
                  method="DLS", nthroot=1, static_3D=False, vel_cor=4.0):
        baz = float(event_or_baz)
        time_shift_table = self.get_timeshift_baz(
            sll, slm, sls, baz, latitude, longitude, absolute_height_in_km,
            static_3D=static_3D, vel_cor=vel_cor)

        vg = self.vespagram_baz(stream, time_shift_table, starttime=starttime,
                                endtime=endtime, method=method,
                                nthroot=nthroot)

    def derive_rotation_from_array(self, stream, vp, vs, sigmau, latitude,
                                   longitude, absolute_height_in_km=0.0):
        geo = self.geometry

        # todo: what is this and should I use it for the _geometry_dict_to_array??
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

            x, y = utlGeoKm(longitude, latitude,
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
                                filter=True, plots=(),
                                static3D=False, vel_corr=4.8, wlen=-1,
                                slx=(-10, 10), sly=(-10, 10), sls=0.5):
        """
        Slowness whitened power analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param filter: Whether to bandpass data to selected frequency range
        :type filter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3D: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3D: bool
        :param vel_corr: Correction velocity for static topography correction in
         km/s.
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
        :param plots: List or tuple of desired output plots, e.g.
         ("baz_slow_map"). Supported options:
         "baz_slow_map" for a backazimuth-slowness map,
         "slowness_xy" for a slowness_xy map,
         "baz_hist" for a backazimuth-slowness polar histogram as in
          :func:`plot_baz_hist`,
         "bf_time_dep" for a plot of beamforming results over time as in
          :func:`plot_bf_results_over_time`.
        """
        return self._array_analysis_helper(stream=stream, method="SWP",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           filter=filter, plots=plots,
                                           static3D=static3D,
                                           vel_corr=vel_corr, wlen=wlen,
                                           slx=slx, sly=sly, sls=sls)

    def phase_weighted_stack(self, stream, frqlow, frqhigh,
                             filter=True, plots=(),
                             static3D=False,
                             vel_corr=4.8, wlen=-1, slx=(-10, 10),
                             sly=(-10, 10), sls=0.5):
        """
        Phase weighted stack analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param filter: Whether to bandpass data to selected frequency range
        :type filter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3D: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3D: bool
        :param vel_corr: Correction velocity for static topography correction in
         km/s.
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
        :param plots: List or tuple of desired output plots, e.g.
         ("baz_slow_map"). Supported options:
         "baz_slow_map" for a backazimuth-slowness map,
         "slowness_xy" for a slowness_xy map,
         "baz_hist" for a backazimuth-slowness polar histogram as in
          :func:`plot_baz_hist`,
         "bf_time_dep" for a plot of beamforming results over time as in
          :func:`plot_bf_results_over_time`.
        """
        return self._array_analysis_helper(stream=stream, method="PWS",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           filter=filter, plots=plots,
                                           static3D=static3D,
                                           vel_corr=vel_corr, wlen=wlen,
                                           slx=slx, sly=sly, sls=sls)

    def delay_and_sum(self, stream, frqlow, frqhigh,
                      filter=True, plots=(), static3D=False,
                      vel_corr=4.8, wlen=-1, slx=(-10, 10),
                      sly=(-10, 10), sls=0.5):
        """
        Delay and sum analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param filter: Whether to bandpass data to selected frequency range
        :type filter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3D: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3D: bool
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
        :param plots: List or tuple of desired output plots, e.g.
         ("baz_slow_map"). Supported options:
         "baz_slow_map" for a backazimuth-slowness map,
         "slowness_xy" for a slowness_xy map,
         "baz_hist" for a backazimuth-slowness polar histogram as in
          :func:`plot_baz_hist`,
         "bf_time_dep" for a plot of beamforming results over time as in
          :func:`plot_bf_results_over_time`.
        """
        return self._array_analysis_helper(stream=stream, method="DLS",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           filter=filter, plots=plots,
                                           static3D=static3D,
                                           vel_corr=vel_corr, wlen=wlen,
                                           slx=slx, sly=sly, sls=sls)

    def fk_analysis(self, stream, frqlow, frqhigh,
                    filter=True, plots=(),
                    static3D=False, vel_corr=4.8, wlen=-1, wfrac=0.8,
                    slx=(-10, 10), sly=(-10, 10), sls=0.5):
        """
        FK analysis.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param filter: Whether to bandpass data to selected frequency range
        :type filter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3D: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3D: bool
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
        :param plots: List or tuple of desired output plots, e.g.
         ("baz_slow_map"). Supported options:
         "baz_slow_map" for backazimuth-slowness maps for each window,
         "slowness_xy" for slowness_xy maps for each window,
         "baz_hist" for a backazimuth-slowness polar histogram as in
          :func:`plot_baz_hist`,
         "bf_time_dep" for a plot of beamforming results over time as in
          :func:`plot_bf_results_over_time`.
        """
        return self._array_analysis_helper(stream=stream, method="FK",
                                           frqlow=frqlow, frqhigh=frqhigh,
                                           filter=filter, plots=plots,
                                           static3D=static3D,
                                           vel_corr=vel_corr,
                                           wlen=wlen, wfrac=wfrac,
                                           slx=slx, sly=sly, sls=sls)

    def _array_analysis_helper(self, stream, method, frqlow, frqhigh,
                               filter=True, static3D=False,
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
        :param filter: Whether to bandpass data to selected frequency range
        :type filter: bool
        :param frqlow: Low corner of frequency range for array analysis
        :type frqlow: float
        :param frqhigh: High corner of frequency range for array analysis
        :type frqhigh: float
        :param static3D: static correction of topography using `vel_corr` as
         velocity (slow!)
        :type static3D: bool
        :param vel_corr: Correction velocity for static topography correction in
         km/s.
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
        :param plots: List or tuple of desired output plots, e.g.
         ("baz_slow_map"). Supported options:
         "baz_slow_map" for a backazimuth-slowness map,
         "slowness_xy" for a slowness_xy map,
         "baz_hist" for a backazimuth-slowness polar histogram as in
          :func:`plot_baz_hist`,
         "bf_time_dep" for a plot of beamforming results over time as in
          :func:`plot_bf_results_over_time`.
        """

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

        # Do not modify the given stream in place.
        st_workon = stream.copy()
        # Trim the stream so all traces are present.
        starttime = max([tr.stats.starttime for tr in st_workon])
        endtime = min([tr.stats.endtime for tr in st_workon])
        st_workon.trim(starttime, endtime)

        self._attach_coords_to_stream(st_workon)

        if filter:
            st_workon.filter('bandpass', freqmin=frqlow, freqmax=frqhigh,
                             zerophase=True)
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
        invbkp = copy.deepcopy(self.inventory)
        self.inventory_cull(st_workon)
        try:
            if method == 'FK':
                kwargs = dict(
                    #slowness grid: X min, X max, Y min, Y max, Slow Step
                    sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                    # sliding window properties
                    win_len=wlen, win_frac=wfrac,
                    # frequency properties
                    frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
                    # restrict output
                    store=dump,
                    semb_thres=-1e9, vel_thres=-1e9, verbose=False,
                    # use mlabday to be compatible with matplotlib
                    timestamp='mlabday', stime=starttime, etime=endtime,
                    method=0, correct_3dplane=False, vel_cor=vel_corr,
                    static_3D=static3D)

                # here we do the array processing
                start = UTCDateTime()
                out = self.array_processing(st_workon, **kwargs)
                print("Total time in routine: %f\n" % (UTCDateTime() - start))

                # make output human readable, adjust backazimuth to values
                # between 0 and 360
                t, rel_power, abs_power, baz, slow = out.T
                baz[baz < 0.0] += 360

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
                    verbose=False, timestamp='mlabday',
                    stime=starttime, etime=endtime, vel_cor=vel_corr,
                    static_3D=False)

                # here we do the array processing
                start = UTCDateTime()
                out = self.beamforming(st_workon, **kwargs)
                print("Total time in routine: %f\n" % (UTCDateTime() - start))

                # make output human readable, adjust backazimuth to values
                # between 0 and 360
                t, rel_power, baz, slow_x, slow_y, slow = out.T
                baz[baz < 0.0] += 360

            # now let's do the plotting
            if "baz_slow_map" in plots:
                _plot_array_analysis(out, sllx, slmx, slly, slmy, sls,
                                     filename_patterns, True, method,
                                     st_workon, starttime, wlen, endtime)
            if "slowness_xy" in plots:
                _plot_array_analysis(out, sllx, slmx, slly, slmy, sls,
                                     filename_patterns, False, method,
                                     st_workon, starttime, wlen, endtime)
            if "baz_hist" in plots:
                plot_baz_hist(out, starttime, endtime,
                              slowness=(min(sllx, slly), max(slmx, slmy)),
                              sls=sls)
            if "bf_time_dep" in plots:
                plot_bf_results_over_time(out, starttime, endtime)

            # Return the beamforming results to allow working more on them,
            # make other plots etc.
            return out
        finally:
            self.inventory = invbkp
            shutil.rmtree(tmpdir)

    def plot_transfer_function(self, stream, sx=(-10, 10),
                               sy=(-10, 10), sls=0.5, freqmin=0.1, freqmax=4.0,
                               numfreqs=10, coordsys='lonlat',
                               correct3dplane=False, static3D=False,
                               velcor=4.8):
        """
        Array Response wrapper routine for MESS 2014.

        :param stream: Waveforms for the array processing.
        :type stream: :class:`obspy.core.stream.Stream`
        :param slx: Min/Max slowness for analysis in x direction.
        :type slx: (float, float)
        :param sly: Min/Max slowness for analysis in y direction.
        :type sly: (float, float)
        :param sls: step width of slowness grid
        :type sls: float
        :param frqmin: Low corner of frequency range for array analysis
        :type frqmin: float
        :param frqmax: High corner of frequency range for array analysis
        :type frqmax: float
        :param numfreqs: number of frequency values used for computing array
         transfer function
        :type numfreqs: int
        :param coordsys: defined coordinate system of stations (lonlat or km)
        :type coordsys: string
        :param correct_3dplane: correct for an inclined surface (not used)
        :type correct_3dplane: bool
        :param static_3D: correct topography
        :type static_3D: bool
        :param velcor: velocity used for static_3D correction
        :type velcor: float
        """
        self._attach_coords_to_stream(stream)

        sllx, slmx = sx
        slly, slmy = sx
        sllx = kilometer2degrees(sllx)
        slmx = kilometer2degrees(slmx)
        slly = kilometer2degrees(slly)
        slmy = kilometer2degrees(slmy)
        sls = kilometer2degrees(sls)

        stepsfreq = (freqmax - freqmin) / float(numfreqs)
        transff = self.array_transff_freqslowness(
            stream, (sllx, slmx, slly, slmy), sls, freqmin, freqmax, stepsfreq,
            coordsys=coordsys, correct_3dplane=False, static_3D=static3D,
            vel_cor=velcor)

        sllx = degrees2kilometers(sllx)
        slmx = degrees2kilometers(slmx)
        slly = degrees2kilometers(slly)
        slmy = degrees2kilometers(slmy)
        sls = degrees2kilometers(sls)

        slx = np.arange(sllx, slmx + sls, sls)
        sly = np.arange(slly, slmy + sls, sls)
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        #ax.pcolormesh(slx, sly, transff.T)
        ax.contour(sly, slx, transff.T, 10)
        ax.set_xlabel('slowness [s/deg]')
        ax.set_ylabel('slowness [s/deg]')
        ax.set_ylim(slx[0], slx[-1])
        ax.set_xlim(sly[0], sly[-1])
        plt.show()

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

    def array_processing(self, stream, win_len, win_frac, sll_x, slm_x, sll_y,
                         slm_y, sl_s, semb_thres, vel_thres, frqlow, frqhigh,
                         stime, etime, prewhiten, verbose=False,
                         timestamp='mlabday', method=0, correct_3dplane=False,
                         vel_cor=4., static_3D=False, store=None):
        """
        Method for FK-Analysis/Capon

        :param stream: Stream object, the trace.stats dict like class must
            contain an :class:`~obspy.core.util.attribdict.AttribDict` with
            'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
            'y', 'elevation' (in km) items/attributes. See param ``coordsys``.
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
        :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
            the timestamp in seconds since 1970-01-01T00:00:00, 'mlabday'
            returns the timestamp in days (decimals represent hours, minutes
            and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
            date plotting (see e.g. matplotlib's num2date)
        :type method: int
        :param method: the method to use 0 == bf, 1 == capon
        :param vel_cor: correction velocity (upper layer) in km/s
        :param static_3D: a correction of the station height is applied using
            vel_cor the correction is done according to the formula:
            t = rxy*s - rz*cos(inc)/vel_cor
            where inc is defined by inv = asin(vel_cor*slow)
        :type store: function
        :param store: A custom function which gets called on each iteration. It is
            called with the relative power map and the time offset as first and
            second arguments and the iteration number as third argument. Useful for
            storing or plotting the map for each iteration. For this purpose the
            dump function of this module can be used.
        :return: :class:`numpy.ndarray` of timestamp, relative relpow, absolute
            relpow, backazimuth, slowness
        """
        res = []
        eotr = True

        # check that sampling rates do not vary
        fs = stream[0].stats.sampling_rate
        if len(stream) != len(stream.select(sampling_rate=fs)):
            msg = ('in array-processing sampling rates of traces in stream are '
                   'not equal')
            raise ValueError(msg)

        grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
        grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

        if correct_3dplane:
            correct_with_3dplane(self.geometry)

        if verbose:
            print("geometry:")
            print(self.geometry)
            print("stream contains following traces:")
            print(stream)
            print("stime = " + str(stime) + ", etime = " + str(etime))

        time_shift_table = self.get_timeshift(sll_x, sll_y, sl_s,
                                              grdpts_x, grdpts_y,
                                              vel_cor=vel_cor,
                                              static_3D=static_3D)
        # offset of arrays
        mini = np.min(time_shift_table[:, :, :])
        maxi = np.max(time_shift_table[:, :, :])

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
        nfft = nextpow2(nsamp)
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
        R = np.empty((nf, nstat, nstat), dtype=np.complex128)
        ft = np.empty((nstat, nf), dtype=np.complex128)
        newstart = stime
        # 0.22 matches 0.2 of historical C bbfk.c
        tap = cosTaper(nsamp, p=0.22)
        offset = 0
        count = 0
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
                    R[:, i, j] = ft[i, :] * ft[j, :].conj()
                    if method == 1:
                        R[:, i, j] /= np.abs(R[:, i, j].sum())
                    if i != j:
                        R[:, j, i] = R[:, i, j].conjugate()
                    else:
                        dpow += np.abs(R[:, i, j].sum())
            dpow *= nstat
            if method == 1:
                # P(f) = 1/(e.H R(f)^-1 e)
                for n in range(nf):
                    R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)

            errcode = clibsignal.generalizedBeamformer(
                relpow_map, abspow_map, steer, R, nstat, prewhiten,
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
            azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
            baz = azimut % -360 + 180
            if relpow > semb_thres and 1. / slow > vel_thres:
                res.append(np.array([newstart.timestamp, relpow, abspow, baz,
                                     slow]))
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
            # 719163 == days between 1970 and 0001 + 1
            res[:, 0] = res[:, 0] / (24. * 3600) + 719163
        else:
            msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
            raise ValueError(msg)
        return np.array(res)

    @staticmethod
    def _three_c_dowhiten(fcoeffZ, fcoeffN, fcoeffE, deltaf):
        # amplitude spectra whitening with moving average and window width ww
        # and weighting factor: 1/((Z+E+N)/3)
        for nst in range(fcoeffZ.shape[0]):
            for nwin in range(fcoeffZ.shape[1]):
                ampZ = np.abs(fcoeffZ[nst, nwin, :])
                ampN = np.abs(fcoeffN[nst, nwin, :])
                ampE = np.abs(fcoeffE[nst, nwin, :])
                ww = int(round(0.01 / deltaf))
                if ww % 2:
                    ww += 1
                nn = len(ampZ)
                csamp = np.zeros((nn, 3), dtype=ampZ.dtype)
                csamp[:, 0] = np.cumsum(ampZ)
                csamp[:, 1] = np.cumsum(ampE)
                csamp[:, 2] = np.cumsum(ampN)
                ampw = np.zeros(nn, dtype=csamp.dtype)
                for k in range(3):
                    ampw[ww / 2:nn - ww / 2] += (csamp[ww:, k] - csamp[:-ww, k])\
                                                / ww
                ampw[nn - ww / 2:] = ampw[nn - ww / 2 - 1]
                ampw[:ww / 2] = ampw[ww / 2]
                ampw *= 1 / 3.
                weight = np.where(ampw > np.finfo(np.float).eps * 10.,
                                  1. / (ampw + np.finfo(np.float).eps), 0.)
                fcoeffZ[nst, nwin, :] *= weight
                fcoeffE[nst, nwin, :] *= weight
                fcoeffN[nst, nwin, :] *= weight
        return fcoeffZ, fcoeffN, fcoeffE

    def _three_c_plot_transfer_function(self, u, periods):
        """
        Plot transfer function of input array geometry 2D.
        """
        # todo: merge with the other transff functions here
        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.
        theo_backazi = theo_backazi.reshape((theo_backazi.size, 1))
        u_y = -np.cos(theo_backazi)
        u_x = -np.sin(theo_backazi)
        geo_array = _geometry_dict_to_array(
            self.get_geometry_xyz(**self.center_of_gravity))
        x_ = geo_array[:, 0]
        y_ = geo_array[:, 1]
        x_ = np.array(x_)
        y_ = np.array(y_)
        steering = u_y * y_ + u_x * x_
        theo_backazi = theo_backazi[:, 0]
        beamres = np.zeros((len(theo_backazi), u.size))
        for p in periods:
            omega = 2. * math.pi / p
            R = np.ones((steering.shape[1], steering.shape[1]))
            for vel in range(len(u)):
                e_steer = np.exp(-1j * steering * omega * u[vel])
                w = e_steer
                wT = w.T.copy()
                beamres[:, vel] = 1. / (
                    steering.shape[1] * steering.shape[1]) * abs(
                    (np.conjugate(w) * np.dot(R, wT).T).sum(1))
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            jetmap = cm.get_cmap('jet')
            CONTF = ax.contourf((theo_backazi[::-1] + math.pi / 2.), u,
                                beamres.T, 100, cmap=jetmap, antialiased=True,
                                linstyles='dotted')
            ax.contour((theo_backazi[::-1] + math.pi / 2.), u, beamres.T, 100,
                       cmap=jetmap)
            ax.set_thetagrids([0, 45., 90., 135., 180., 225., 270., 315.],
                              labels=['90', '45', '0', '315', '270', '225',
                                      '180', '135'])
            ax.set_rgrids([0.1, 0.2, 0.3, 0.4, 0.5],
                          labels=['0.1', '0.2', '0.3', '0.4', '0.5'],
                          color='r')
            ax.set_rmax(u[-1])
            fig.colorbar(CONTF)
            ax.grid(True)
            ax.set_title('Transfer function s ' + str(p))
        plt.show()

    def _three_c_do_bf(self, stream_N, stream_E, stream_Z, win_len, u,
                       sub_freq_range, n_min_stns, polarisation,
                       whiten, coherency, win_average,
                       datalen_sec, uindex):
        # backazimuth range to search
        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.

        # Number of stations should be the same as the number of traces,
        # given the checks in the calling method.
        n_stats = len(stream_N.traces)
        npts = stream_N[0].stats.npts

        geo_array = _geometry_dict_to_array(
            self.get_geometry_xyz(**self.center_of_gravity))
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
        _alldataZ = np.zeros((n_stats, npts)) * np.nan
        _alldataE = _alldataZ.copy()
        _alldataN = _alldataZ.copy()
        # array used for sorting if needed
        ans = []
        for i, (tr_N, tr_E, tr_Z) in enumerate(zip(stream_N, stream_E,
                                                   stream_Z)):
            ans.append(np.where(geo_items_names == tr_N.id)[0][0])
            _alldataN[i, :] = tr_N.data
            _alldataE[i, :] = tr_E.data
            _alldataZ[i, :] = tr_Z.data

        fs = stream_N.traces[0].stats.sampling_rate
        nsamp = int(win_len * fs)
        num_win = int(np.floor(datalen_sec / win_len))
        alldataZ = np.zeros((n_stats, num_win, nsamp))
        alldataN, alldataE = alldataZ.copy(), alldataZ.copy()
        nst = np.zeros(num_win)

        # Iterate over the beamfoming windows:
        for i in range(num_win):
            for n in range(n_stats):
                if not np.isnan(_alldataZ[n, i * nsamp:(
                            i + 1) * nsamp]).any() and not np.isnan(
                        _alldataN[n, i * nsamp:(
                                    i + 1) * nsamp]).any() and not np.isnan(
                        _alldataE[n, i * nsamp:(i + 1) * nsamp]).any():
                    alldataZ[n, i, :] = _alldataZ[n, i * nsamp:(
                                                                         i + 1) * nsamp] * cosTaper(
                        nsamp)
                    alldataN[n, i, :] = _alldataN[n, i * nsamp:(
                                                                         i + 1) * nsamp] * cosTaper(
                        nsamp)
                    alldataE[n, i, :] = _alldataE[n, i * nsamp:(
                                                                         i + 1) * nsamp] * cosTaper(
                        nsamp)
                    nst[i] += 1

        print(nst, ' stations/window; average over ', win_average)

        # Do Fourier transform.
        deltat = stream_N.traces[0].stats.delta
        freq_range = np.fft.fftfreq(nsamp, deltat)
        # Use a narrower 'frequency range' of interest for evaluating incidence
        # angle.
        lowcorner = sub_freq_range[0]
        highcorner = sub_freq_range[1]
        index = np.where((freq_range > lowcorner)
                         & (freq_range < highcorner))[0]
        fr = freq_range[index]
        fcoeffZ = np.fft.fft(alldataZ, n=nsamp, axis=-1) / nsamp
        fcoeffN = np.fft.fft(alldataN, n=nsamp, axis=-1) / nsamp
        fcoeffE = np.fft.fft(alldataE, n=nsamp, axis=-1) / nsamp
        fcoeffZ = fcoeffZ[:, :, index]
        fcoeffN = fcoeffN[:, :, index]
        fcoeffE = fcoeffE[:, :, index]
        deltaf = 1. / (nsamp * deltat)

        if whiten:
            fcoeffZ, fcoeffN, fcoeffE = self._three_c_dowhiten(fcoeffZ,
                                                               fcoeffN,
                                                               fcoeffE, deltaf)

        # slowness vector u and slowness vector component scale u_x and u_y
        theo_backazi = theo_backazi.reshape((theo_backazi.size, 1))
        u_y = -np.cos(theo_backazi)
        u_x = -np.sin(theo_backazi)

        # vector of source direction dependent plane wave travel-distance to
        # reference point (positive value for later arrival/negative for
        # earlier arr)
        x_offsets = np.array(x_offsets)
        y_offsets = np.array(y_offsets)
        # This sorts the offset value arrays
        x_offsets = x_offsets[np.array(ans)]
        y_offsets = y_offsets[np.array(ans)]
        steering = u_y * y_offsets + u_x * x_offsets

        # polarizations [Z,E,N]
        # incident angle P-wave/S-wave or atan(H/V) Rayleigh-wave
        incs = np.arange(5, 90, 10) * math.pi / 180.

        def pol_love(azi):
            polE = math.cos(theo_backazi[azi])
            polN = -1. * math.sin(theo_backazi[azi])
            return polE, polN

        def pol_rayleigh_retro(azi):
            polE = math.sin(theo_backazi[azi])
            polN = math.cos(theo_backazi[azi])
            return polE, polN

        def pol_rayleigh_prog(azi):
            polE = -1 * math.sin(theo_backazi[azi])
            polN = -1 * math.cos(theo_backazi[azi])
            return polE, polN

        def pol_P(azi):
            polE = -1 * math.sin(theo_backazi[azi])
            polN = -1 * math.cos(theo_backazi[azi])
            return polE, polN

        def pol_SV(azi):
            polE = math.sin(theo_backazi[azi])
            polN = math.cos(theo_backazi[azi])
            return polE, polN

        Cz = [0., 1j, 1j, 1., 1.]
        Ch = (pol_love, pol_rayleigh_retro, pol_rayleigh_prog, pol_P, pol_SV)

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
                Sz = np.squeeze(fcoeffZ[:, win, f])
                Sn = np.squeeze(fcoeffN[:, win, f])
                Se = np.squeeze(fcoeffE[:, win, f])

                Y = np.concatenate((Sz, Sn, Se))
                Y = Y.reshape(1, Y.size)
                YT = Y.T.copy()
                R = np.dot(YT, np.conjugate(Y))

                for wi in range(1, win_average):
                    Sz = np.squeeze(fcoeffZ[:, win + wi, f])
                    Sn = np.squeeze(fcoeffN[:, win + wi, f])
                    Se = np.squeeze(fcoeffE[:, win + wi, f])

                    Y = np.concatenate((Sz, Sn, Se))
                    Y = Y.reshape(1, Y.size)
                    YT = Y.T.copy()
                    R += np.dot(YT, np.conjugate(Y))

                R /= float(win_average)

                res = np.zeros((len(theo_backazi), len(u), len(incs)))
                for vel in range(len(u)):
                    e_steer = np.exp(-1j * steering * omega * u[vel])
                    e_steerE = e_steer.copy()
                    e_steerN = e_steer.copy()
                    e_steerE = (e_steerE.T * np.array([Ch[polarisation](azi)[0]
                                for azi in range(len(theo_backazi))])).T
                    e_steerN = (e_steerN.T * np.array([Ch[polarisation](azi)[1]
                                for azi in range(len(theo_backazi))])).T

                    if polarisation == 0:
                        w = np.concatenate(
                            (e_steer * Cz[polarisation], e_steerN, e_steerE),
                            axis=1)
                        wT = w.T.copy()
                        if not coherency:
                            beamres[:, vel, win / win_average, f] = 1. / (
                                nst[win] * nst[win]) * abs(
                                (np.conjugate(w) * np.dot(R, wT).T).sum(1))
                        else:
                            beamres[:, vel, win / win_average, f] = 1. / (
                                nst[win]) * abs((np.conjugate(w)
                                                 * np.dot(R, wT).T).sum(1)) \
                                / abs(np.sum(np.diag(R[n_stats:, n_stats:])))

                    elif polarisation in [1, 2, 3]:
                        for inc_angle in range(len(incs)):
                            w = np.concatenate((e_steer * Cz[polarisation]
                                                * np.cos(incs[inc_angle]),
                                                e_steerN
                                                * np.sin(incs[inc_angle]),
                                                e_steerE
                                                * np.sin(incs[inc_angle])),
                                               axis=1)
                            wT = w.T.copy()
                            if not coherency:
                                res[:, vel, inc_angle] = 1. / (
                                    nst[win] * nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(R, wT).T).sum(1))
                            else:
                                res[:, vel, inc_angle] = 1. / (nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(R, wT).T).sum(
                                        1)) / abs(np.sum(np.diag(R)))

                    elif polarisation == 4:
                        for inc_angle in range(len(incs)):
                            w = np.concatenate((e_steer * Cz[polarisation]
                                                * np.sin(incs[inc_angle]),
                                                e_steerN
                                                * np.cos(incs[inc_angle]),
                                                e_steerE
                                                * np.cos(incs[inc_angle])),
                                               axis=1)
                            wT = w.T.copy()
                            if not coherency:
                                res[:, vel, inc_angle] = 1. / (
                                    nst[win] * nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(R, wT).T).sum(1))
                            else:
                                res[:, vel, inc_angle] = 1. / (nst[win]) * abs(
                                    (np.conjugate(w) * np.dot(R, wT).T).sum(
                                        1)) / abs(np.sum(np.diag(R)))

                if polarisation > 0:
                    i, j, k = np.unravel_index(np.argmax(res[:, uindex, :]),
                                               res.shape)
                    beamres[:, :, win / win_average, f] = res[:, :, k]
                    incidence[win / win_average, f] = incs[k] * 180. / math.pi

        # Could call plot of every window here. Don't.

        return beamres, fr, incidence

    @staticmethod
    def three_c_beamform_plotter(beamresult, u, freqs, plot_frequencies=(),
                                 average_windows=True, average_freqs=True):
        """
        Pass in an unaveraged beamresult, i.e. with 4 axes. Dud windows should
        (not happen or) be signified by all zeros, so np.nonzero can catch
        them.
        """
        if average_freqs is True and len(plot_frequencies) > 0:
            warnings.warn("Ignoring plot_frequencies, only plotting an average"
                          "of all frequencies.")
        theo_backazi = np.arange(0, 362, 2) * math.pi / 180.

        def _actual_plotting(bfres, title):
            """
            Pass in a 2D bfres array of beamforming results with
            averaged or selected windows and frequencies.
            """
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1, projection='polar')
            jetmap = cm.get_cmap('jet')
            CONTF = ax.contourf((theo_backazi[::-1] + math.pi / 2.), u,
                                bfres.T, 100, cmap=jetmap, antialiased=True,
                                linstyles='dotted')
            ax.contour((theo_backazi[::-1] + math.pi / 2.), u, bfres.T, 100,
                       cmap=jetmap)
            ax.set_thetagrids([0, 45., 90., 135., 180., 225., 270., 315.],
                              labels=['90', '45', '0', '315', '270', '225',
                                      '180', '135'])
            # ax.set_rgrids([0.1,0.2,0.3,0.4,0.5],labels=['0.1','0.2','0.3',
            # '0.4','0.5'],color='r')
            ax.set_rmax(u[-1])
            fig.colorbar(CONTF)
            ax.grid(True)
            ax.set_title(title)

        # Try to remove all windows where actually no beamforming happened -
        # ought to be full of zeros; don't want to include those in the
        # averages (although maybe the absolute values don't really matter in
        # the end).
        #nonzero_windows = np.nonzero(beamresult[0, 0, :, 0])[0]
        #beamresnz = beamresult[:, :, nonzero_windows, :]
        beamresnz = beamresult
        if average_windows:
            beamresnz = beamresnz.mean(axis=2)
        if average_freqs:
            # Always an average over the last axis, whether or not windows
            # were averaged.
            beamresnz = beamresnz.mean(axis=beamresnz.ndim - 1)

        if average_windows and average_freqs:
            _actual_plotting(beamresnz, 'Averaged BF result.')

        if average_windows and not average_freqs:
            for plot_freq in plot_frequencies:
                # works because freqs is a range
                ifreq = np.searchsorted(freqs, plot_freq)
                _actual_plotting(beamresnz[:, :, ifreq],
                                 'Averaged windows, frequency {}'
                                 .format(freqs[ifreq]))

        if average_freqs and not average_windows:
            for iwin in range(len(beamresnz[0, 0, :, 0])):
                _actual_plotting(beamresnz[:, :, iwin],
                                 'Averaged all frequencies, window {}'
                                 .format(iwin))

        # Plotting all windows, selected frequencies.
        if average_freqs is False and average_windows is False:
            for plot_freq in plot_frequencies:
                ifreq = np.searchsorted(freqs, plot_freq)
                for iwin in range(len(beamresnz[0, 0, :, 0])):
                    _actual_plotting(beamresnz[:, :, iwin, ifreq],
                                     'BF result window {}, freq {}'
                                     .format(iwin, freqs[ifreq]))

        plt.plot()
        plt.show()

    def three_component_beamforming(self, stream_N, stream_E, stream_Z, wlen,
                                    smin, smax, sstep, wavetype,
                                    freq_range, plot_frequencies=(7, 14),
                                    n_min_stns=7, win_average=1,
                                    plot_transff=False,
                                    plot_average_freqs=True):
        """
        Do three component beamforming following Esmersoy 1985...
        Three streams representing N, E, Z oriented components must be given,
        where the traces contained are from the different stations. The
        traces must all have same length and start/end times (to within
        sampling distance). (hint: check length with trace.stats.npts)
        The given streams are not modified in place. All trimming, filtering,
        downsampling should be done previously.
        The beamforming can distinguish Love, prograde/retrograde Rayleigh, P
        and SV waves.
        Station location information is taken from the array's inventory, so
        that must contain station/channel location information about all traces
        used (it may contain more than used in the traces as well).

        :param stream_N: Stream of all traces for the North component.
        :param stream_E: stream of East components
        :param stream_Z: stream of Up components. Will be ignored for Love
         waves.
        :param wlen: window length in seconds
        :param smin: minimum slowness of the slowness grid [km/s]
        :param smax: maximum slowness [km/s]
        :param sstep: slowness step [km/s]
        :param wavetype: 'love', 'rayleigh_prograde', 'rayleigh_retrograde',
         'P', or 'SV'
        :param freq_range: Frequency band (min, max) that is used for
         beamforming and returned. Ideally, use the frequency band of the
         pre-filter.
        :param plot_frequencies: frequencies to plot [s]
        :param n_min_stns: required minimum number of stations
        :param win_average: number of windows to average covariance matrix over
        :param plot_transff: whether to also plot the transfer function of the
         array
        :param plot_average_freqs: whether to plot an average of results for
         all frequencies
        :return: A four dimensional :class:`numpy.ndarray` of the beamforming
         results, with dimensions of backazimuth range, slowness range, number
         of windows and number of discrete frequencies; as well as frequency
         and incidence angle arrays (the latter will be zero for Love waves).
        """
        pol_dict = {'love': 0, 'rayleigh_retrograde': 1, 'rayleigh_prograde':
                    2, 'P': 3, 'SV': 4}
        wavetype = wavetype.lower()
        if wavetype not in pol_dict:
            raise ValueError('Invalid option for wavetype: {}'
                             .format(wavetype))

        # from _array_analysis_helper:
        starttime = max(max([tr.stats.starttime for tr in st]) for st in
                        (stream_N, stream_E, stream_E))
        min_starttime = min(min([tr.stats.starttime for tr in st]) for st in
                            (stream_N, stream_E, stream_E))
        endtime = min(min([tr.stats.endtime for tr in st]) for st in
                      (stream_N, stream_E, stream_E))
        max_endtime = max(max([tr.stats.endtime for tr in st]) for st in
                          (stream_N, stream_E, stream_E))

        delta_common = stream_N.traces[0].stats.delta
        npts_common = stream_N.traces[0].stats.npts
        if max(abs(min_starttime - starttime),
               abs(max_endtime - endtime)) > delta_common:
            raise ValueError("Traces do not have identical start/end times. "
                             "Trim to same times (within sample accuracy) "
                             "and ensure all traces have the same number "
                             "of samples!")

        # Check for equal deltas and number of samples:
        for st in (stream_N, stream_E, stream_Z):
            for tr in st:
                if tr.stats.npts != npts_common:
                    raise ValueError('Traces do not have identical number of '
                                     'samples.')
                if tr.stats.delta != delta_common:
                    raise ValueError('Traces do not have identical sampling '
                                     'rates.')
        datalen_sec = endtime - starttime

        # Sort all traces just to make sure they're in the same order.
        for st in (stream_N, stream_E, stream_Z):
            st.sort()

        for trN, trE, trZ in zip(stream_N, stream_E, stream_Z):
            if len(set('{}.{}'.format(tr.stats.network, tr.stats.station)
                       for tr in (trN, trE, trZ))) > 1:
                raise ValueError("Traces are not from same stations.")

        # Temporarily trim self.inventory so only stations/channels which are
        # actually represented in the traces are kept in the inventory.
        # Otherwise self.geometry and the xyz geometry arrays will have more
        # entries than the stream.
        invbkp = copy.deepcopy(self.inventory)
        allstreams = stream_N + stream_E + stream_Z
        self.inventory_cull(allstreams)

        if wlen < smax * self.aperture:
            raise ValueError('Window length is smaller than maximum given'
                             ' slowness times aperture.')
        try:
            # s/km  slowness range calculated
            u = np.arange(smin, smax, sstep)
            # Slowness range evaluated for (incidence) angle measurement
            # (Rayleigh, P, SV):
            # These values are a bit arbitrary for now:
            uindex = np.where((u > 0.5 * smax + smin)
                              & (u < 0.8 * smax + smin))[0]

            bf_results, freqs, incidence = \
                self._three_c_do_bf(stream_N, stream_E, stream_Z,
                                    win_len=wlen, u=u,
                                    sub_freq_range=freq_range,
                                    n_min_stns=n_min_stns,
                                    polarisation=pol_dict[wavetype],
                                    whiten=False,
                                    coherency=True,
                                    win_average=win_average,
                                    datalen_sec=datalen_sec,
                                    uindex=uindex)

            self.three_c_beamform_plotter(bf_results,
                                          plot_frequencies=plot_frequencies,
                                          u=u, freqs=freqs,
                                          average_freqs=plot_average_freqs)

        finally:
            self.inventory = invbkp

        # todo: take this out, it's better as its own method
        if plot_transff:
            self._three_c_plot_transfer_function(u, plot_frequencies)

        return bf_results, freqs, incidence

    @staticmethod
    def array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs,
                              array_coords, sigmau):
        """
        This routine calculates the best-fitting rigid body rotation and
        uniform strain as functions of time, and their formal errors, given
        three-component ground motion time series recorded on a seismic array.
        The theory implemented herein is presented in the papers [Spudich1995]_,
        (abbreviated S95 herein) [Spudich2008]_ (SF08) and [Spudich2009]_ (SF09).

        This is a translation of the Matlab Code presented in (SF09) with
        small changes in details only. Output has been checked to be the same
        as the original Matlab Code.

        .. note::
            ts\_ below means "time series"

        :type vp: float
        :param vp: P wave speed in the soil under the array (km/s)
        :type vs: float
        :param vs: S wave speed in the soil under the array Note - vp and vs may be
            any unit (e.g. miles/week), and this unit need not be related to the
            units of the station coordinates or ground motions, but the units of vp
            and vs must be the SAME because only their ratio is used.
        :type array_coords: numpy.ndarray
        :param array_coords: array of dimension Na x 3, where Na is the number of
            stations in the array.  array_coords[i,j], i in arange(Na), j in
            arange(3) is j coordinate of station i.  units of array_coords may be
            anything, but see the "Discussion of input and output units" above.
            The origin of coordinates is arbitrary and does not affect the
            calculated strains and rotations.  Stations may be entered in any
            order.
        :type ts1: numpy.ndarray
        :param ts1: array of x1-component seismograms, dimension nt x Na.
            ts1[j,k], j in arange(nt), k in arange(Na) contains the k'th time
            sample of the x1 component ground motion at station k. NOTE that the
            seismogram in column k must correspond to the station whose coordinates
            are in row k of in.array_coords. nt is the number of time samples in
            the seismograms.  Seismograms may be displacement, velocity,
            acceleration, jerk, etc.  See the "Discussion of input and output
            units" below.
        :type ts2: numpy.ndarray
        :param ts2: same as ts1, but for the x2 component of motion.
        :type ts3: numpy.ndarray
        :param ts3: same as ts1, but for the x3 (UP or DOWN) component of motion.
        :type sigmau: float or :class:`numpy.ndarray`
        :param sigmau: standard deviation (NOT VARIANCE) of ground noise,
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
        :param subarray: NumPy array of subarray stations to use. I.e. if subarray
            = array([1, 4, 10]), then only rows 1, 4, and 10 of array_coords will
            be used, and only ground motion time series in the first, fourth, and
            tenth columns of ts1 will be used. n_plus_1 is the number of elements
            in the subarray vector, and N is set to n_plus_1 - 1. To use all
            stations in the array, set in.subarray = arange(Na), where Na is the
            total number of stations in the array (equal to the number of rows of
            in.array_coords. Sequence of stations in the subarray vector is
            unimportant; i.e.  subarray = array([1, 4, 10]) will yield essentially
            the same rotations and strains as subarray = array([10, 4, 1]).
            "Essentially" because permuting subarray sequence changes the d vector,
            yielding a slightly different numerical result.
        :return: Dictionary with fields:

            **A:** (array, dimension 3N x 6)
                data mapping matrix 'A' of S95(A4)
            **g:** (array, dimension 6 x 3N)
                generalized inverse matrix relating ptilde and data vector, in
                S95(A5)
            **Ce:** (4 x 4)
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
            **Cgamma:** (4 x 4)
                covariance matrix of the 4 independent shear strain tensor elements
                g11, g12, g22, g33 (includes full covariance effects). gamma is
                traceless part of e.
            **ts_sh:** (array, length nt)
                maximum horizontal strain ( .5*(max eigval of eh - min eigval of
                eh) as a function of time, where eh is e(1:2,1:2)
            **Cgammah:** (3 x 3)
                covariance matrix of the 3 independent horizontal shear strain
                tensor elements gamma11, gamma12, gamma22 gamma is traceless part
                of e.
            **ts_wmag:** (array, length nt)
                total rotation angle (radians) as a function of time.  I.e. if the
                rotation vector at the j'th time step is
                w = array([w1, w2, w3]), then ts_wmag[j] = sqrt(sum(w**2))
                positive for right-handed rotation
            **Cw:** (3 x 3)
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
                to linalg.dot(A, ptilde), where A is the big matrix in S95 eqn A4
                and ptilde is S95 eqn A5
            **ts_misfit:** (array, shape (nt x 3N))
                time series of the residuals (fitted model displacement differences
                minus observed displacement differences). Note that the fitted
                model displacement differences correspond to linalg.dot(A, ptilde),
                where A is the big matrix in S95 eqn A4 and ptilde is S95 eqn A5
            **ts_M:** (array, length nt)
                Time series of M, misfit ratio of S95, p. 688
            **ts_ptilde:** (array, shape (nt x 6))
                solution vector p-tilde (from S95 eqn A5) as a function of time
            **Cp:** (6 x 6)
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
            stations and the user may select for analysis a subarray of n_plus_1
            <= Na stations.

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

        # check to verify that the number of stations in ts1 agrees with the number
        # of stations in array_coords
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
        # offsets: N
        n_plus_1 = subarray.size
        N = n_plus_1 - 1

        if n_plus_1 < 3:
            msg = 'The problem is underdetermined for fewer than 3 stations'
            raise ValueError(msg)
        elif n_plus_1 == 3:
            msg = 'For a 3-station array the problem is even-determined'
            warnings.warn(msg)

        # ------------------- NOW SOME SEISMOLOGY!! --------------------------
        # constants
        eta = 1 - 2 * vs ** 2 / vp ** 2

        # form A matrix, which relates model vector of 6 displacement
        # derivatives to vector of observed displacement differences. S95(A3)
        # dim(A) = (3*N) * 6
        # model vector is [ u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ] (free surface
        # boundary conditions applied, S95(A2))
        # first initialize A to the null matrix
        A = np.zeros((N * 3, 6))
        z3t = np.zeros(3)
        # fill up A
        for i in range(N):
            ss = subarraycoords[(i + 1), :] - subarraycoords[0, :]
            A[(3 * i):(3 * i + 3), :] = np.c_[
                np.r_[ss, z3t], np.r_[z3t, ss],
                np.array([-eta * ss[2],
                          0., -ss[0], 0., -eta * ss[2], -ss[1]])].transpose()

        # ------------------------------------------------------
        # define data covariance matrix Cd.
        # step 1 - define data differencing matrix D
        # dimension of D is (3*N) * (3*n_plus_1)
        I3 = np.eye(3)
        II = np.eye(3 * N)
        D = -I3

        for i in range(N - 1):
            D = np.c_[D, -I3]
        D = np.r_[D, II].T

        # step 2 - define displacement u covariance matrix Cu
        # This assembles a covariance matrix Cu that reflects actual
        # data errors.
        # populate Cu depending on the size of sigmau
        if np.size(sigmau) == 1:
            # sigmau is a scalar.  Make all diag elements of Cu the same
            Cu = sigmau ** 2 * np.eye(3 * n_plus_1)
        elif np.shape(sigmau) == (np.size(sigmau),):
            # sigmau is a row or column vector
            # check dimension is okay
            if np.size(sigmau) != na:
                raise ValueError('sigmau must have %s elements' % na)
            junk = (np.c_[sigmau, sigmau, sigmau]) ** 2  # matrix of variances
            Cu = np.diag(np.reshape(junk[subarray, :], (3 * n_plus_1)))
        elif sigmau.shape == (na, 3):
            Cu = np.diag(np.reshape(((sigmau[subarray, :]) ** 2).transpose(),
                                    (3 * n_plus_1)))
        else:
            raise ValueError('sigmau has the wrong dimensions')

        # Cd is the covariance matrix of the displ differences
        # dim(Cd) is (3*N) * (3*N)
        Cd = np.dot(np.dot(D, Cu), D.T)

        # ---------------------------------------------------------
        # form generalized inverse matrix g.  dim(g) is 6 x (3*N)
        Cdi = np.linalg.inv(Cd)
        AtCdiA = np.dot(np.dot(A.T, Cdi), A)
        g = np.dot(np.dot(np.linalg.inv(AtCdiA), A.T), Cdi)

        condition_number = np.linalg.cond(AtCdiA)

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
        ts_pred = np.empty((nt, 3 * N))
        ts_misfit = np.empty((nt, 3 * N))
        ts_M = np.empty(nt)
        ts_data = np.empty((nt, 3 * N))
        ts_ptilde = np.empty((nt, 6))
        for array in (ts_wmag, ts_w1, ts_w2, ts_w3, ts_tilt, ts_dh, ts_sh,
                      ts_s, ts_pred, ts_misfit, ts_M, ts_data, ts_ptilde):
            array.fill(np.NaN)
        ts_e = np.empty((nt, 3, 3))
        ts_e.fill(np.NaN)

        # other matrices
        udif = np.empty((3, N))
        udif.fill(np.NaN)

        # ---------------------------------------------------------------
        # here we define 4x6 Be and 3x6 Bw matrices.  these map the solution
        # ptilde to strain or to rotation.  These matrices will be used
        # in the calculation of the covariances of strain and rotation.
        # Columns of both matrices correspond to the model solution vector
        # containing elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]'
        #
        # the rows of Be correspond to e11 e21 e22 and e33
        Be = np.zeros((4, 6))
        Be[0, 0] = 2.
        Be[1, 1] = 1.
        Be[1, 3] = 1.
        Be[2, 4] = 2.
        Be[3, 0] = -2 * eta
        Be[3, 4] = -2 * eta
        Be *= .5
        #
        # the rows of Bw correspond to w21 w31 and w32
        Bw = np.zeros((3, 6))
        Bw[0, 1] = 1.
        Bw[0, 3] = -1.
        Bw[1, 2] = 2.
        Bw[2, 5] = 2.
        Bw *= .5
        #
        # this is the 4x6 matrix mapping solution to total shear strain gamma
        # where gamma = strain - tr(strain)/3 * eye(3)
        # the four elements of shear are 11, 12, 22, and 33.  It is symmetric.
        aa = (2 + eta) / 3
        b = (1 - eta) / 3
        c = (1 + 2 * eta) / 3
        Bgamma = np.zeros((4, 6))
        Bgamma[0, 0] = aa
        Bgamma[0, 4] = -b
        Bgamma[2, 2] = .5
        Bgamma[1, 3] = .5
        Bgamma[2, 0] = -b
        Bgamma[2, 4] = aa
        Bgamma[3, 0] = -c
        Bgamma[3, 4] = -c
        #
        # this is the 3x6 matrix mapping solution to horizontal shear strain
        # gamma
        # the four elements of horiz shear are 11, 12, and 22.  It is symmetric.
        Bgammah = np.zeros((3, 6))
        Bgammah[0, 0] = .5
        Bgammah[0, 4] = -.5
        Bgammah[1, 1] = .5
        Bgammah[1, 3] = .5
        Bgammah[2, 0] = -.5
        Bgammah[2, 4] = .5

        # solution covariance matrix.  dim(Cp) = 6 * 6
        # corresponding to solution elements [u1,1 u1,2 u1,3 u2,1 u2,2 u2,3 ]
        Cp = np.dot(np.dot(g, Cd), g.T)

        # Covariance of strain tensor elements
        # Ce should be 4x4, correspond to e11, e21, e22, e33
        Ce = np.dot(np.dot(Be, Cp), Be.T)
        # Cw should be 3x3 correspond to w21, w31, w32
        Cw = np.dot(np.dot(Bw, Cp), Bw.T)

        # Cgamma is 4x4 correspond to 11, 12, 22, and 33.
        Cgamma = np.dot(np.dot(Bgamma, Cp), Bgamma.T)
        #
        #  Cgammah is 3x3 correspond to 11, 12, and 22
        Cgammah = np.dot(np.dot(Bgammah, Cp), Bgammah.T)
        #
        #
        # covariance of the horizontal dilatation and the total dilatation
        # both are 1x1, i.e. scalars
        Cdh = Cp[0, 0] + 2 * Cp[0, 4] + Cp[4, 4]
        sigmadh = np.sqrt(Cdh)

        # covariance of the (total) dilatation, ts_dd
        sigmadsq = (1 - eta) ** 2 * Cdh
        sigmad = np.sqrt(sigmadsq)
        #
        # Cw3, covariance of w3 rotation, i.e. torsion, is 1x1, i.e. scalar
        Cw3 = (Cp[1, 1] - 2 * Cp[1, 3] + Cp[3, 3]) / 4
        sigmaw3 = np.sqrt(Cw3)

        # For tilt cannot use same approach because tilt is not a linear
        # function
        # of the solution.  Here is an approximation :
        # For tilt use conservative estimate from
        # Papoulis (1965, p. 195, example 7.8)
        sigmaw1 = np.sqrt(Cp[5, 5])
        sigmaw2 = np.sqrt(Cp[2, 2])
        sigmat = max(sigmaw1, sigmaw2) * np.sqrt(2 - np.pi / 2)

        #
        # BEGIN LOOP OVER DATA POINTS IN TIME SERIES==========================
        #
        for itime in range(nt):
            #
            # data vector is differences of stn i displ from stn 1 displ
            # sum the lengths of the displ difference vectors
            sumlen = 0
            for i in range(N):
                udif[0, i] = ts1[itime, subarray[i + 1]] - ts1[itime, subarray[0]]
                udif[1, i] = ts2[itime, subarray[i + 1]] - ts2[itime, subarray[0]]
                udif[2, i] = ts3[itime, subarray[i + 1]] - ts3[itime, subarray[0]]
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
            pred = np.dot(A, ptilde)  # 9/8/92.I.3(9) and 8/26/92.I.3.T bottom
            #
            # calculate  residuals (misfits concatenated for all stations)
            misfit = pred - data

            # Calculate ts_M, misfit ratio.
            # calculate summed length of misfits (residual displacements)
            misfit_sq = misfit ** 2
            misfit_sq = np.reshape(misfit_sq, (N, 3)).T
            misfit_sumsq = np.empty(N)
            misfit_sumsq.fill(np.NaN)
            for i in range(N):
                misfit_sumsq[i] = misfit_sq[:, i].sum()
            misfit_len = np.sum(np.sqrt(misfit_sumsq))
            ts_M[itime] = misfit_len / sumlen
            #
            ts_data[itime, 0:3 * N] = data.T
            ts_pred[itime, 0:3 * N] = pred.T
            ts_misfit[itime, 0:3 * N] = misfit.T
            ts_ptilde[itime, :] = ptilde.T
            #
            # ---------------------------------------------------------------
            # populate the displacement gradient matrix U
            U = np.zeros(9)
            U[:] = uij_vector
            U = U.reshape((3, 3))
            #
            # calculate strain tensors
            # Fung eqn 5.1 p 97 gives dui = (eij-wij)*dxj
            e = .5 * (U + U.T)
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
            # 7/21/06.II.6(19), amount of tilt in radians

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
            # 7/21/06.II.2(4)
            gammah = eh - np.trace(eh) * np.eye(2) / 2.
            # 9/14/92.II.4, 7/21/06.II.2(5)

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

        out['A'] = A
        out['g'] = g
        out['Ce'] = Ce

        out['ts_d'] = ts_d
        out['sigmad'] = sigmad

        out['ts_dh'] = ts_dh
        out['sigmadh'] = sigmadh

        out['ts_s'] = ts_s
        out['Cgamma'] = Cgamma

        out['ts_sh'] = ts_sh
        out['Cgammah'] = Cgammah

        out['ts_wmag'] = ts_wmag
        out['Cw'] = Cw

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
        out['ts_M'] = ts_M
        out['ts_e'] = ts_e

        out['ts_ptilde'] = ts_ptilde
        out['Cp'] = Cp

        out['ts_M'] = ts_M

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
                msg = "Specified stime %s is smaller than starttime %s in stream"
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

    def array_transff_wavenumber(self, klim, kstep):
        """
        Returns array transfer function as a function of wavenumber difference

        :param klim: either a float to use symmetric limits for wavenumber
            differences or the tuple (kxmin, kxmax, kymin, kymax)
        """
        coords = _geometry_dict_to_array(self.get_geometry_xyz(
            **self.center_of_gravity))

        if isinstance(klim, float):
            kxmin = -klim
            kxmax = klim
            kymin = -klim
            kymax = klim
        elif isinstance(klim, tuple):
            if len(klim) == 4:
                kxmin = klim[0]
                kxmax = klim[1]
                kymin = klim[2]
                kymax = klim[3]
        else:
            raise TypeError('klim must either be a float or a tuple of length 4')

        nkx = int(np.ceil((kxmax + kstep / 10. - kxmin) / kstep))
        nky = int(np.ceil((kymax + kstep / 10. - kymin) / kstep))

        transff = np.empty((nkx, nky))

        for i, kx in enumerate(np.arange(kxmin, kxmax + kstep / 10., kstep)):
            for j, ky in enumerate(np.arange(kymin, kymax + kstep / 10., kstep)):
                _sum = 0j
                for k in range(len(coords)):
                    _sum += np.exp(complex(0.,
                                           coords[k, 0] * kx + coords[k, 1] * ky))
                transff[i, j] = abs(_sum) ** 2

        transff /= transff.max()
        return transff

    def array_transff_freqslowness(self, slim, sstep, fmin, fmax,
                                   fstep):
        """
        Returns array transfer function as a function of slowness difference and
        frequency.

        :param slim: either a float to use symmetric limits for slowness
            differences or the tupel (sxmin, sxmax, symin, symax)
        :type fmin: float
        :param fmin: minimum frequency in signal
        :type fmax: float
        :param fmin: maximum frequency in signal
        :type fstep: float
        :param fmin: frequency sample distance
        """
        geometry = _geometry_dict_to_array(self.get_geometry_xyz(
            **self.center_of_gravity))

        if isinstance(slim, float):
            sxmin = -slim
            sxmax = slim
            symin = -slim
            symax = slim
        elif isinstance(slim, tuple):
            if len(slim) == 4:
                sxmin = slim[0]
                sxmax = slim[1]
                symin = slim[2]
                symax = slim[3]
        else:
            raise TypeError('slim must either be a float '
                            'or a tuple of length 4')

        nsx = int(np.ceil((sxmax + sstep / 10. - sxmin) / sstep))
        nsy = int(np.ceil((symax + sstep / 10. - symin) / sstep))
        nf = int(np.ceil((fmax + fstep / 10. - fmin) / fstep))

        transff = np.empty((nsx, nsy))
        buff = np.zeros(nf)

        for i, sx in enumerate(np.arange(sxmin, sxmax + sstep / 10., sstep)):
            for j, sy in enumerate(np.arange(symin, symax + sstep / 10., sstep)):
                for k, f in enumerate(np.arange(fmin, fmax + fstep / 10., fstep)):
                    _sum = 0j
                    for l in np.arange(len(geometry)):
                        _sum += np.exp(complex(
                            0., (geometry[l, 0] * sx + geometry[l, 1] * sy) *
                            2 * np.pi * f))
                    buff[k] = abs(_sum) ** 2
                transff[i, j] = cumtrapz(buff, dx=fstep)[-1]

        transff /= transff.max()
        return transff

    def beamforming(self, stream, sll_x, slm_x, sll_y, slm_y, sl_s, frqlow,
                    frqhigh, stime, etime, win_len=-1, win_frac=0.5,
                    verbose=False, timestamp='mlabday',
                    method="DLS", nthroot=1, store=None, correct_3dplane=False,
                    static_3D=False, vel_cor=4.):
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
            is
            called with the relative power map and the time offset as first and
            second arguments and the iteration number as third argument. Useful
            for
            storing or plotting the map for each iteration.
        :param correct_3dplane: if Yes than a best (LSQ) plane will be fitted
            into the array geometry.
            Mainly used with small apature arrays at steep flanks
        :param static_3D: if yes the station height of am array station is
            taken into account accoring the formula:
                tj = -xj*sxj - yj*syj + zj*cos(inc)/vel_cor
            the inc angle is slowness dependend and thus must
            be estimated for each grid-point:
                inc = asin(v_cor*slow)
        :param vel_cor: Velocity for the upper layer (static correction) in km/s
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
        geometry = _geometry_dict_to_array(self.get_geometry_xyz(
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
                                              static_3D=static_3D)

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
                nfft = nextpow2(nsamp)
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

                        tap = cosTaper(nsamp, p=0.22)
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

    def vespagram_baz(self, stream, time_shift_table, starttime, endtime,
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
                for _j, tr in stream:
                    station = "%s.%s" % (tr.stats.network, tr.stats.station)
                    s = spoint[_j] + int(time_shift_table[slowness][station] *
                                         fs + 0.5)
                    shifted = tr.data[s: s + ndat]
                    singlet += 1. / len(stream) * np.sum(shifted * shifted)
                    beams[_i] += 1. / len(stream) * np.power(
                        np.abs(shifted), 1. / nthroot) * shifted / \
                                 np.abs(shifted)

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


def correct_with_3dplane(geometry):
    """
    Correct a given array geometry with a best-fitting plane.
    :param geometry: nested dictionary of stations, as returned for example by
    self.geometry or self.get_geometry_xyz.
    :return: Returns corrected geometry, again as dict.
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
    geometry = _geometry_dict_to_array(geometry)
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
    #print("Best fitting plane-coordinates :\n", geometry)

    # convert geometry array back to a dictionary.
    geodict = {}
    # The sorted list is necessary to match the station IDs (the keys in the
    # geometry dict) to the correct array row (or column?), same as is done in
    # _geometry_dict_to_array, but backwards.
    for _i, (key, value) in enumerate(sorted(
            list(orig_geometry.items()))):
        geodict[key] = {coord_sys_keys[0]: geometry[_i, 0],
                        coord_sys_keys[1]: geometry[_i, 1],
                        coord_sys_keys[2]: geometry[_i, 2]}
    geometry = geodict
    return geometry


def _plot_array_analysis(out, sllx, slmx, slly, slmy, sls, filename_patterns,
                         baz_plot, method, st_workon, starttime, wlen,
                         endtime):
    """
    Some plotting taken out from _array_analysis_helper. Can't do the array
    response overlay now though.
    :param baz_plot: Whether to show backazimuth-slowness map (True) or
     slowness x-y map (False).
    """
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
    T = np.arange(0, npts / df, 1 / df)

    # if we choose windowlen > 0. we now move through our slices
    for i in range(numslice):
        slow_x = np.sin((baz[i] + 180.) * np.pi / 180.) * slow[i]
        slow_y = np.cos((baz[i] + 180.) * np.pi / 180.) * slow[i]
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
            ax1.plot(T, st_workon[0].data, 'k')
            if wlen > 0.:
                try:
                    ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                except IndexError:
                    pass
        else:
            T = np.arange(0, len(trace[i]) / df, 1 / df)
            ax1.plot(T, trace[i], 'k')

        ax1.yaxis.set_major_locator(MaxNLocator(3))

        ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])

        # if we have chosen the baz_plot option a re-griding
        # of the sx,sy slowness map is needed
        if baz_plot:
            slowgrid = []
            transgrid = []
            pow = np.asarray(powmap[i])
            for ix, sx in enumerate(slx):
                for iy, sy in enumerate(sly):
                    bbaz = np.arctan2(sx, sy) * 180 / np.pi + 180.
                    if bbaz > 180.:
                        bbaz = -180. + (bbaz - 180.)
                    slowgrid.append((np.sqrt(sx * sx + sy * sy), bbaz,
                                     pow[ix, iy]))

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


def plot_baz_hist(out, t_start=None, t_end=None, slowness=(0, 3), sls=0.1):
    """
    Plot a backazimuth - slowness histogram.
    :param out: beamforming result e.g. from SeismicArray.fk_analysis.
    :param slowness: the radial axis limits.
    :param sls: slowness step (bin width)
    """
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    cmap = cm.hot_r
    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    # Can't plot negative slownesses:
    sll = slowness[0] if slowness[0] > 0 else 0
    slm = slowness[1]

    # choose number of azimuth bins in plot
    # (desirably 360 degree/N is an integer!)
    N = 36
    # number of slowness bins
    N2 = math.ceil((slowness[1] - slowness[0]) / sls)
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(sll, slm, N2 + 1)

    # sum rel power in bins given by abins and sbins
    hist, baz_edges, sl_edges = \
        np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)

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
        bars = ax.bar(left=(i * dw) * np.ones(N2),
                      height=dh * np.ones(N2),
                      width=dw, bottom=dh * np.arange(N2),
                      color=cmap(row / hist.max()))

    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])

    # set slowness limits
    ax.set_ylim(sll, slm)
    ColorbarBase(cax, cmap=cmap,
                 norm=Normalize(vmin=hist.min(), vmax=hist.max()))
    if t_start is not None and t_end is not None:
        plt.suptitle('Time: {} - {}'.format(t_start, t_end))
    plt.show()


def plot_bf_results_over_time(out, t_start, t_end):
    import matplotlib.dates as mdates
    # Plot
    labels = ['rel.power', 'abs.power', 'baz', 'slow']

    xlocator = mdates.AutoDateLocator()
    fig = plt.figure()
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(4, 1, i + 1)
        ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                   edgecolors='none')
        ax.set_ylabel(lab)
        ax.set_xlim(out[0, 0], out[-1, 0])
        ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

    fig.suptitle('Time-dependent beamforming results %s' % (
        t_start.strftime('%Y-%m-%d'), ))
    #fig.autofmt_xdate()
    fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
    plt.show()


if __name__ == '__main__':
    import doctest

    doctest.testmod(exclude_empty=True)
