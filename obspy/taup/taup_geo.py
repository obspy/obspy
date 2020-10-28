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
    Given the source and receiver location, calculate distance.

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
    return calc_dist_azi(source_latitude_in_deg, source_longitude_in_deg,
                         receiver_latitude_in_deg, receiver_longitude_in_deg,
                         radius_of_planet_in_km, flattening_of_planet)[0]


def calc_dist_azi(source_latitude_in_deg, source_longitude_in_deg,
                  receiver_latitude_in_deg, receiver_longitude_in_deg,
                  radius_of_planet_in_km, flattening_of_planet):
    """
    Given the source and receiver location, calculate the azimuth from the
    source to the receiver at the source, the backazimuth from the receiver
    to the source at the receiver and distance between the source and receiver.

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

    :returns: distance_in_deg (in degrees), source_receiver_azimuth (in
              degrees) and receiver_to_source_backazimuth (in degrees).
    :rtype: tuple of three floats
    """
    if geodetics.HAS_GEOGRAPHICLIB:
        ellipsoid = Geodesic(a=radius_of_planet_in_km * 1000.0,
                             f=flattening_of_planet)
        g = ellipsoid.Inverse(source_latitude_in_deg,
                              source_longitude_in_deg,
                              receiver_latitude_in_deg,
                              receiver_longitude_in_deg)
        distance_in_deg = g['a12']
        source_receiver_azimuth = g['azi1'] % 360
        receiver_to_source_backazimuth = (g['azi2'] + 180) % 360

    else:
        # geographiclib is not installed - use obspy/geodetics
        values = gps2dist_azimuth(source_latitude_in_deg,
                                  source_longitude_in_deg,
                                  receiver_latitude_in_deg,
                                  receiver_longitude_in_deg,
                                  a=radius_of_planet_in_km * 1000.0,
                                  f=flattening_of_planet)
        distance_in_km = values[0] / 1000.0
        source_receiver_azimuth = values[1] % 360
        receiver_to_source_backazimuth = values[2] % 360
        # NB - km2deg assumes spherical planet... generate a warning
        if flattening_of_planet != 0.0:
            msg = "Assuming spherical planet when calculating epicentral " + \
                  "distance. Install the Python module 'geographiclib' " + \
                  "to solve this."
            warnings.warn(msg)
        distance_in_deg = kilometer2degrees(distance_in_km,
                                            radius=radius_of_planet_in_km)
    return (distance_in_deg, source_receiver_azimuth,
            receiver_to_source_backazimuth)


def add_geo_to_arrivals(arrivals, source_latitude_in_deg,
                        source_longitude_in_deg, receiver_latitude_in_deg,
                        receiver_longitude_in_deg, radius_of_planet_in_km,
                        flattening_of_planet, resample=False):
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
    :param resample: adds sample points to allow for easy cartesian
                     interpolation. This is especially useful for phases
                     like Pdiff.
    :type resample: boolean


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
                    signed_dist = np.degrees(sign * pierce_point['dist'])
                    pos = line.ArcPosition(signed_dist)
                    geo_pierce[i] = (pierce_point['p'], pierce_point['time'],
                                     pierce_point['dist'],
                                     pierce_point['depth'],
                                     pos['lat2'], pos['lon2'])
                arrival.pierce = geo_pierce

            # choose whether we need to resample the trace
            if arrival.path is not None:
                if resample:
                    rplanet = radius_of_planet_in_km
                    # compute approximate distance between sampling points
                    mindist = 200  # km
                    radii = rplanet - arrival.path['depth']
                    rmean = np.sqrt(radii[1:] * radii[:-1])
                    diff_dists = rmean * np.diff(arrival.path['dist'])
                    npts_extra = np.floor(diff_dists / mindist).astype(np.int)

                    # count number of extra points and initialize array
                    npts_old = len(arrival.path)
                    npts_new = int(npts_old + np.sum(npts_extra))
                    geo_path = np.empty(npts_new, dtype=TimeDistGeo)

                    # now loop through path, adding extra points
                    i_new = 0
                    for i_old, path_point in enumerate(arrival.path):
                        # first add the original point at the new index
                        dist = np.degrees(sign * path_point['dist'])
                        pos = line.ArcPosition(dist)
                        geo_path[i_new] = (path_point['p'], path_point['time'],
                                           path_point['dist'],
                                           path_point['depth'],
                                           pos['lat2'], pos['lon2'])
                        i_new += 1

                        if i_old > npts_old - 2:
                            continue

                        # now check if we need to add new points
                        npts_new = npts_extra[i_old]
                        if npts_new > 0:
                            # if yes, distribute them linearly between the old
                            # and the next point
                            next_point = arrival.path[i_old + 1]
                            dist_next = np.degrees(sign * next_point['dist'])
                            dists_new = np.linspace(dist, dist_next,
                                                    npts_new + 2)[1: -1]

                            # now get all interpolated parameters
                            xs = [dist, dist_next]
                            ys = [path_point['p'], next_point['p']]
                            p_interp = np.interp(dists_new, xs, ys)
                            ys = [path_point['time'], next_point['time']]
                            time_interp = np.interp(dists_new, xs, ys)
                            ys = [path_point['depth'], next_point['depth']]
                            depth_interp = np.interp(dists_new, xs, ys)
                            pos_interp = [line.ArcPosition(dist_new)
                                          for dist_new in dists_new]
                            lat_interp = [point['lat2']
                                          for point in pos_interp]
                            lon_interp = [point['lon2']
                                          for point in pos_interp]

                            # add them to geo_path
                            for i_extra in range(npts_new):
                                geo_path[i_new] = (p_interp[i_extra],
                                                   time_interp[i_extra],
                                                   dists_new[i_extra],
                                                   depth_interp[i_extra],
                                                   lat_interp[i_extra],
                                                   lon_interp[i_extra])
                                i_new += 1

                    arrival.path = geo_path
                else:
                    geo_path = np.empty(arrival.path.shape, dtype=TimeDistGeo)
                    for i, path_point in enumerate(arrival.path):
                        signed_dist = np.degrees(sign * path_point['dist'])
                        pos = line.ArcPosition(signed_dist)
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
