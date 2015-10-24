#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to handle geographical points

These functions are used to alow taup models
to process input data with source and station 
locations given as lstitudes and longitudes. The
functions are set up to hande an elliptical Earth
model, but we do not have ellipticity corrections
for travel times. Although changing the 
shape of the Earth from something other than spherical
would change the epicentral distance the change
in distance along the ray path has a larger effect,
and we do not make that correction.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np

from .helper_classes import TimeDistGeo

# We should plug into the obspy geodetics
# module, and handle cases where geographiclib
# does not exist, but we would need new functions.
import geographiclib.geodesic as geod


def calc_dist(source_latitude_in_deg, source_longitude_in_deg,
              station_latitude_in_deg, station_longitude_in_deg,
              radius_of_earth_in_km, flattening_of_earth):
    """
    Given the source and station location, calculate the azimuth and distance

    :param source_latitude_in_deg: Source location latitude in degrees
    :type source_latitude_in_deg: float
    :param source_longitude_in_deg: Source location longitue in degrees
    :type source_longitude_in_deg: float
    :param station_latitude_in_deg: Station location latitude in degrees
    :type station_latitude_in_deg: float
    :param station_longitude_in_deg: Station location longitude in degrees
    :type station_longitude_in_deg: float
    :param radius_of_earth_in_km: Radius of the Earth in km
    :type radius_of_earth_in_km: float
    :param flattening_of_earth: Flattening of earth (0 for a sphere)
    :type station_longitude_in_deg: float

    :return: distance_in_deg
    :rtype: float
    """
    ellipsoid=geod.Geodesic(a=radius_of_earth_in_km*1000.0, 
                            f=flattening_of_earth)
    g = ellipsoid.Inverse(source_latitude_in_deg, source_longitude_in_deg,
                          station_latitude_in_deg, station_longitude_in_deg)
    distance_in_deg = g['a12']

    return distance_in_deg

def add_geo_to_arrivals(arrivals, source_latitude_in_deg, 
                  source_longitude_in_deg, station_latitude_in_deg, 
                  station_longitude_in_deg, radius_of_earth_in_km, 
                  flattening_of_earth):
    """
    Add geographical information to arrivals

    :param arrivals: Set of taup arrivals
    :type: :class:`Arrivals`
    :param source_latitude_in_deg: Source location latitude in degrees
    :type source_latitude_in_deg: float
    :param source_longitude_in_deg: Source location longitue in degrees
    :type source_longitude_in_deg: float
    :param station_latitude_in_deg: Station location latitude in degrees
    :type station_latitude_in_deg: float
    :param station_longitude_in_deg: Station location longitude in degrees
    :type station_longitude_in_deg: float
    :param radius_of_earth_in_km: Radius of the Earth in km
    :type radius_of_earth_in_km: float
    :param flattening_of_earth: Flattening of earth (0 for a sphere)
    :type station_longitude_in_deg: float

    :return: List of ``Arrival`` objects, each of which has the time,
        corresponding phase name, ray parameter, takeoff angle, etc. as
        attributes.
    :rtype: :class:`Arrivals`
    """
    ellipsoid=geod.Geodesic(a=radius_of_earth_in_km*1000.0, 
                            f=flattening_of_earth)
    g = ellipsoid.Inverse(source_latitude_in_deg, source_longitude_in_deg,
                          station_latitude_in_deg, station_longitude_in_deg)
    distance_in_deg = g['a12']
    azimuth = g['azi1']
    line = ellipsoid.Line(source_latitude_in_deg,
                          source_longitude_in_deg, azimuth)

    # We may need to update many arrival objects
    # and each could have pierce points and a 
    # path
    for arrival in arrivals:

        if arrival.pierce is not None:
            pathList = []
            for pierce_point in arrival.pierce:
                pos = line.ArcPosition(np.degrees(pierce_point['dist']))
                diffTDG = np.array([(
                    pierce_point['p'],
                    pierce_point['time'],
                    pierce_point['dist'],
                    pierce_point['depth'],
                    pos['lat2'],
                    pos['lon2'])], dtype=TimeDistGeo)
                pathList.append(diffTDG)
            arrival.pierce = np.concatenate(pathList)

        if arrival.path is not None:
            pathList = []
            for path_point in arrival.path:
                pos = line.ArcPosition(np.degrees(path_point['dist']))
                diffTDG = np.array([(
                    path_point['p'],
                    path_point['time'],
                    path_point['dist'],
                    path_point['depth'],
                    pos['lat2'],
                    pos['lon2'])], dtype=TimeDistGeo)
                pathList.append(diffTDG)
            arrival.path = np.concatenate(pathList)

    return arrivals
