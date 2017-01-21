#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to handle geographical points

These functions are used to allow taup models to process input data with
source and receiver locations given as latitudes and longitudes. The functions
are set up to handle an elliptical planet model, but we do not have ellipticity
corrections for travel times. Although changing the shape of the planet from
something other than spherical would change the epicentral distance, the change
in the distance for the ray to pass through each layer has a larger effect.
We do not make the larger correction.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
import warnings

import numpy as np

from .helper_classes import TimeDistGeo
from ..geodetics import gps2dist_azimuth, kilometer2degrees
import obspy.geodetics.base as geodetics


if geodetics.HAS_GEOGRAPHICLIB:
    from geographiclib.geodesic import Geodesic


def calc_dist(source_latitude_in_deg, source_longitude_in_deg,
              receiver_latitude_in_deg, receiver_longitude_in_deg,
              radius_of_planet_in_km, flattening_of_planet):
    """
    Given the source and receiver location, calculate the azimuth and distance.

    :param source_latitude_in_deg: Source location latitude in degrees
    :type source_latitude_in_deg: float
    :param source_longitude_in_deg: Source location longitude in degrees
    :type source_longitude_in_deg: float
    :param receiver_latitude_in_deg: Receiver location latitude in degrees
    :type receiver_latitude_in_deg: float
    :param receiver_longitude_in_deg: Receiver location longitude in degrees
    :type receiver_longitude_in_deg: float
    :param radius_of_planet_in_km: Radius of the planet in km
    :type radius_of_planet_in_km: float
    :param flattening_of_planet: Flattening of planet (0 for a sphere)
    :type receiver_longitude_in_deg: float

    :return: distance_in_deg
    :rtype: float
    """
    if geodetics.HAS_GEOGRAPHICLIB:
        ellipsoid = Geodesic(a=radius_of_planet_in_km * 1000.0,
                             f=flattening_of_planet)
        g = ellipsoid.Inverse(source_latitude_in_deg,
                              source_longitude_in_deg,
                              receiver_latitude_in_deg,
                              receiver_longitude_in_deg)
        distance_in_deg = g['a12']

    else:
        # geographiclib is not installed - use obspy/geodetics
        values = gps2dist_azimuth(source_latitude_in_deg,
                                  source_longitude_in_deg,
                                  receiver_latitude_in_deg,
                                  receiver_longitude_in_deg,
                                  a=radius_of_planet_in_km * 1000.0,
                                  f=flattening_of_planet)
        distance_in_km = values[0] / 1000.0
        # NB - km2deg assumes spherical planet... generate a warning
        if flattening_of_planet != 0.0:
            msg = "Assuming spherical planet when calculating epicentral " + \
                  "distance. Install the Python module 'geographiclib' " + \
                  "to solve this."
            warnings.warn(msg)
        distance_in_deg = kilometer2degrees(distance_in_km,
                                            radius=radius_of_planet_in_km)
    return distance_in_deg


def add_geo_to_arrivals(arrivals, source_latitude_in_deg,
                        source_longitude_in_deg, receiver_latitude_in_deg,
                        receiver_longitude_in_deg, radius_of_planet_in_km,
                        flattening_of_planet):
    """
    Add geographical information to arrivals.

    :param arrivals: Set of taup arrivals
    :type: :class:`Arrivals`
    :param source_latitude_in_deg: Source location latitude in degrees
    :type source_latitude_in_deg: float
    :param source_longitude_in_deg: Source location longitude in degrees
    :type source_longitude_in_deg: float
    :param receiver_latitude_in_deg: Receiver location latitude in degrees
    :type receiver_latitude_in_deg: float
    :param receiver_longitude_in_deg: Receiver location longitude in degrees
    :type receiver_longitude_in_deg: float
    :param radius_of_planet_in_km: Radius of the planet in km
    :type radius_of_planet_in_km: float
    :param flattening_of_planet: Flattening of planet (0 for a sphere)
    :type receiver_longitude_in_deg: float

    :return: List of ``Arrival`` objects, each of which has the time,
        corresponding phase name, ray parameter, takeoff angle, etc. as
        attributes.
    :rtype: :class:`Arrivals`
    """
    if geodetics.HAS_GEOGRAPHICLIB:
        if not geodetics.GEOGRAPHICLIB_VERSION_AT_LEAST_1_34:
            # geographiclib is not installed ...
            # and  obspy/geodetics does not help much
            msg = ("This functionality needs the Python module "
                   "'geographiclib' in version 1.34 or higher.")
            raise ImportError(msg)
        ellipsoid = Geodesic(a=radius_of_planet_in_km * 1000.0,
                             f=flattening_of_planet)
        g = ellipsoid.Inverse(source_latitude_in_deg, source_longitude_in_deg,
                              receiver_latitude_in_deg,
                              receiver_longitude_in_deg)
        azimuth = g['azi1']
        line = ellipsoid.Line(source_latitude_in_deg, source_longitude_in_deg,
                              azimuth)

        # We may need to update many arrival objects
        # and each could have pierce points and a
        # path
        for arrival in arrivals:
            # check if we go in minor or major arc direction
            distance = arrival.purist_distance % 360.
            if distance > 180.:
                sign = -1
                az_arr = (azimuth + 180.) % 360.
            else:
                sign = 1
                az_arr = azimuth
            arrival.azimuth = az_arr

            if arrival.pierce is not None:
                geo_pierce = np.empty(arrival.pierce.shape, dtype=TimeDistGeo)

                for i, pierce_point in enumerate(arrival.pierce):
                    dir_degrees = np.degrees(sign * pierce_point['dist'])
                    pos = line.ArcPosition(dir_degrees)
                    geo_pierce[i] = (pierce_point['p'], pierce_point['time'],
                                     pierce_point['dist'],
                                     pierce_point['depth'],
                                     pos['lat2'], pos['lon2'])
                arrival.pierce = geo_pierce

            if arrival.path is not None:
                geo_path = np.empty(arrival.path.shape, dtype=TimeDistGeo)
                for i, path_point in enumerate(arrival.path):
                    dir_degrees = np.degrees(sign * path_point['dist'])
                    pos = line.ArcPosition(dir_degrees)
                    geo_path[i] = (path_point['p'], path_point['time'],
                                   path_point['dist'], path_point['depth'],
                                   pos['lat2'], pos['lon2'])
                arrival.path = geo_path

    else:
        # geographiclib is not installed ...
        # and  obspy/geodetics does not help much
        msg = "You need to install the Python module 'geographiclib' in " + \
              "order to add geographical information to arrivals."
        raise ImportError(msg)

    return arrivals
