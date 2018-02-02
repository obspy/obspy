# -*- coding: utf-8 -*-
"""
Various geodetic utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math
import warnings

import numpy as np
from scipy.stats import circmean

from obspy.core.util.misc import to_int_or_zero


# checking for geographiclib
try:
    import geographiclib  # @UnusedImport # NOQA
    from geographiclib.geodesic import Geodesic
    HAS_GEOGRAPHICLIB = True
    try:
        GEOGRAPHICLIB_VERSION_AT_LEAST_1_34 = [1, 34] <= list(map(
            to_int_or_zero, geographiclib.__version__.split(".")))
    except AttributeError:
        GEOGRAPHICLIB_VERSION_AT_LEAST_1_34 = False
except ImportError:
    HAS_GEOGRAPHICLIB = False
    GEOGRAPHICLIB_VERSION_AT_LEAST_1_34 = False


WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563


def calc_vincenty_inverse(lat1, lon1, lat2, lon2, a=WGS84_A, f=WGS84_F):
    """
    Vincenty Inverse Solution of Geodesics on the Ellipsoid.

    Computes the distance between two geographic points on the WGS84
    ellipsoid and the forward and backward azimuths between these points.

    :param lat1: Latitude of point A in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon1: Longitude of point A in degrees (positive for eastern,
        negative for western hemisphere)
    :param lat2: Latitude of point B in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon2: Longitude of point B in degrees (positive for eastern,
        negative for western hemisphere)
    :param a: Radius of Earth in m. Uses the value for WGS84 by default.
    :param f: Flattening of Earth. Uses the value for WGS84 by default.
    :return: (Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees)
    :raises: This method may have no solution between two nearly antipodal
        points; an iteration limit traps this case and a ``StopIteration``
        exception will be raised.

    .. note::
        This code is based on an implementation incorporated in
        Matplotlib Basemap Toolkit 0.9.5 http://sourceforge.net/projects/\
matplotlib/files/matplotlib-toolkits/basemap-0.9.5/
        (matplotlib/toolkits/basemap/greatcircle.py)

        Algorithm from Geocentric Datum of Australia Technical Manual.

        * http://www.icsm.gov.au/gda/
        * http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf, pp. 15

        It states::

            Computations on the Ellipsoid

            There are a number of formulae that are available to calculate
            accurate geodetic positions, azimuths and distances on the
            ellipsoid.

            Vincenty's formulae (Vincenty, 1975) may be used for lines ranging
            from a few cm to nearly 20,000 km, with millimetre accuracy. The
            formulae have been extensively tested for the Australian region, by
            comparison with results from other formulae (Rainsford, 1955 &
            Sodano, 1965).

            * Inverse problem: azimuth and distance from known latitudes and
              longitudes
            * Direct problem: Latitude and longitude from known position,
              azimuth and distance.
    """
    # Check inputs
    if lat1 > 90 or lat1 < -90:
        msg = "Latitude of Point 1 out of bounds! (-90 <= lat1 <=90)"
        raise ValueError(msg)
    while lon1 > 180:
        lon1 -= 360
    while lon1 < -180:
        lon1 += 360
    if lat2 > 90 or lat2 < -90:
        msg = "Latitude of Point 2 out of bounds! (-90 <= lat2 <=90)"
        raise ValueError(msg)
    while lon2 > 180:
        lon2 -= 360
    while lon2 < -180:
        lon2 += 360

    b = a * (1 - f)  # semiminor axis

    if (abs(lat1 - lat2) < 1e-8) and (abs(lon1 - lon2) < 1e-8):
        return 0.0, 0.0, 0.0

    # convert latitudes and longitudes to radians:
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    tan_u1 = (1 - f) * math.tan(lat1)
    tan_u2 = (1 - f) * math.tan(lat2)

    u_1 = math.atan(tan_u1)
    u_2 = math.atan(tan_u2)

    dlon = lon2 - lon1
    last_dlon = -4000000.0  # an impossible value
    omega = dlon

    # Iterate until no significant change in dlon or iterlimit has been
    # reached (http://www.movable-type.co.uk/scripts/latlong-vincenty.html)
    iterlimit = 100
    try:
        while (last_dlon < -3000000.0 or dlon != 0 and
               abs((last_dlon - dlon) / dlon) > 1.0e-9):
            sqr_sin_sigma = pow(math.cos(u_2) * math.sin(dlon), 2) + \
                pow((math.cos(u_1) * math.sin(u_2) - math.sin(u_1) *
                     math.cos(u_2) * math.cos(dlon)), 2)
            sin_sigma = math.sqrt(sqr_sin_sigma)
            cos_sigma = math.sin(u_1) * math.sin(u_2) + math.cos(u_1) * \
                math.cos(u_2) * math.cos(dlon)
            sigma = math.atan2(sin_sigma, cos_sigma)
            sin_alpha = math.cos(u_1) * math.cos(u_2) * math.sin(dlon) / \
                math.sin(sigma)
            alpha = math.asin(sin_alpha)
            cos2sigma_m = math.cos(sigma) - \
                (2 * math.sin(u_1) * math.sin(u_2) / pow(math.cos(alpha), 2))
            c = (f / 16) * pow(math.cos(alpha), 2) * \
                (4 + f * (4 - 3 * pow(math.cos(alpha), 2)))
            last_dlon = dlon
            dlon = omega + (1 - c) * f * math.sin(alpha) * \
                (sigma + c * math.sin(sigma) *
                    (cos2sigma_m + c * math.cos(sigma) *
                        (-1 + 2 * pow(cos2sigma_m, 2))))

            u2 = pow(math.cos(alpha), 2) * (a * a - b * b) / (b * b)
            _a = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 *
                                                  (320 - 175 * u2)))
            _b = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
            delta_sigma = _b * sin_sigma * \
                (cos2sigma_m + (_b / 4) *
                    (cos_sigma * (-1 + 2 * pow(cos2sigma_m, 2)) - (_b / 6) *
                        cos2sigma_m * (-3 + 4 * sqr_sin_sigma) *
                        (-3 + 4 * pow(cos2sigma_m, 2))))

            dist = b * _a * (sigma - delta_sigma)
            alpha12 = math.atan2(
                (math.cos(u_2) * math.sin(dlon)),
                (math.cos(u_1) * math.sin(u_2) -
                 math.sin(u_1) * math.cos(u_2) * math.cos(dlon)))
            alpha21 = math.atan2(
                (math.cos(u_1) * math.sin(dlon)),
                (-math.sin(u_1) * math.cos(u_2) +
                 math.cos(u_1) * math.sin(u_2) * math.cos(dlon)))
            iterlimit -= 1
            if iterlimit < 0:
                # iteration limit reached
                raise StopIteration
    except ValueError:
        # usually "math domain error"
        raise StopIteration

    if alpha12 < 0.0:
        alpha12 = alpha12 + (2.0 * math.pi)
    if alpha12 > (2.0 * math.pi):
        alpha12 = alpha12 - (2.0 * math.pi)

    alpha21 = alpha21 + math.pi

    if alpha21 < 0.0:
        alpha21 = alpha21 + (2.0 * math.pi)
    if alpha21 > (2.0 * math.pi):
        alpha21 = alpha21 - (2.0 * math.pi)

    # convert to degrees:
    alpha12 = alpha12 * 360 / (2.0 * math.pi)
    alpha21 = alpha21 * 360 / (2.0 * math.pi)

    return dist, alpha12, alpha21


def gps2dist_azimuth(lat1, lon1, lat2, lon2, a=WGS84_A, f=WGS84_F):
    """
    Computes the distance between two geographic points on the WGS84
    ellipsoid and the forward and backward azimuths between these points.

    :param lat1: Latitude of point A in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon1: Longitude of point A in degrees (positive for eastern,
        negative for western hemisphere)
    :param lat2: Latitude of point B in degrees (positive for northern,
        negative for southern hemisphere)
    :param lon2: Longitude of point B in degrees (positive for eastern,
        negative for western hemisphere)
    :param a: Radius of Earth in m. Uses the value for WGS84 by default.
    :param f: Flattening of Earth. Uses the value for WGS84 by default.
    :return: (Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees)

    .. note::
        This function will check if you have installed the Python module
        `geographiclib <http://geographiclib.sf.net>`_ - a very fast module
        for converting between geographic, UTM, UPS, MGRS, and geocentric
        coordinates, for geoid calculations, and for solving geodesic problems.
        Otherwise the locally implemented Vincenty's Inverse formulae
        (:func:`obspy.core.util.geodetics.calc_vincenty_inverse`) is used which
        has known limitations for two nearly antipodal points and is ca. 4x
        slower.
    """
    if HAS_GEOGRAPHICLIB:
        if lat1 > 90 or lat1 < -90:
            msg = "Latitude of Point 1 out of bounds! (-90 <= lat1 <=90)"
            raise ValueError(msg)
        if lat2 > 90 or lat2 < -90:
            msg = "Latitude of Point 2 out of bounds! (-90 <= lat2 <=90)"
            raise ValueError(msg)
        result = Geodesic(a=a, f=f).Inverse(lat1, lon1, lat2, lon2)
        azim = result['azi1']
        if azim < 0:
            azim += 360
        bazim = result['azi2'] + 180
        return (result['s12'], azim, bazim)
    else:
        try:
            values = calc_vincenty_inverse(lat1, lon1, lat2, lon2, a, f)
            if np.alltrue(np.isnan(values)):
                raise StopIteration
            return values
        except StopIteration:
            msg = ("Catching unstable calculation on antipodes. "
                   "The currently used Vincenty's Inverse formulae "
                   "has known limitations for two nearly antipodal points. "
                   "Install the Python module 'geographiclib' to solve this "
                   "issue.")
            warnings.warn(msg)
            return (20004314.5, 0.0, 0.0)
        except ValueError as e:
            raise e


def kilometers2degrees(kilometer, radius=6371):
    """
    Convenience function to convert kilometers to degrees assuming a perfectly
    spherical Earth.

    :type kilometer: float
    :param kilometer: Distance in kilometers
    :type radius: int, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in degrees as a floating point number.

    .. rubric:: Example

    >>> from obspy.geodetics import kilometers2degrees
    >>> kilometers2degrees(300)
    2.6979648177561915
    """
    return kilometer / (2.0 * radius * math.pi / 360.0)


kilometer2degrees = kilometers2degrees


def degrees2kilometers(degrees, radius=6371):
    """
    Convenience function to convert (great circle) degrees to kilometers
    assuming a perfectly spherical Earth.

    :type degrees: float
    :param degrees: Distance in (great circle) degrees
    :type radius: int, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in kilometers as a floating point number.

    .. rubric:: Example

    >>> from obspy.geodetics import degrees2kilometers
    >>> degrees2kilometers(1)
    111.19492664455873
    """
    return degrees * (2.0 * radius * math.pi / 360.0)


def locations2degrees(lat1, long1, lat2, long2):
    """
    Convenience function to calculate the great circle distance between two
    points on a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (https://github.com/geopy/geopy).

    :type lat1: float or :class:`numpy.ndarray`
    :param lat1: Latitude(s) of point 1 in degrees
    :type long1: float or :class:`numpy.ndarray`
    :param long1: Longitude(s) of point 1 in degrees
    :type lat2: float or :class:`numpy.ndarray`
    :param lat2: Latitude(s) of point 2 in degrees
    :type long2: float or :class:`numpy.ndarray`
    :param long2: Longitude(s) of point 2 in degrees
    :rtype: float or :class:`numpy.ndarray`
    :return: Distance in degrees as a floating point number,
        or numpy array of element-wise distances in degrees

    .. rubric:: Example

    >>> from obspy.geodetics import locations2degrees
    >>> locations2degrees(5, 5, 10, 10)
    7.0397014191753815
    """
    # broadcast explicitly here so it raises once instead of somewhere in the
    # middle if things can't be broadcast
    lat1, lat2, long1, long2 = np.broadcast_arrays(lat1, lat2, long1, long2)

    # Convert to radians.
    lat1 = np.radians(np.asarray(lat1))
    lat2 = np.radians(np.asarray(lat2))
    long1 = np.radians(np.asarray(long1))
    long2 = np.radians(np.asarray(long2))
    long_diff = long2 - long1
    gd = np.degrees(
        np.arctan2(
            np.sqrt((
                np.cos(lat2) * np.sin(long_diff)) ** 2 +
                (np.cos(lat1) * np.sin(lat2) - np.sin(lat1) *
                    np.cos(lat2) * np.cos(long_diff)) ** 2),
            np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) *
            np.cos(long_diff)))
    return gd


def mean_longitude(longitudes):
    """
    Compute sample mean longitude, assuming longitude in degrees from -180 to
    180.

    >>> lons = (-170.5, -178.3, 166)
    >>> np.mean(lons)  # doctest: +SKIP
    -60.933
    >>> mean_longitude(lons)  # doctest: +ELLIPSIS
    179.08509...

    :type longitudes: :class:`~numpy.ndarray` (or list, ..)
    :param longitudes: Geographical longitude values ranging from -180 to 180
        in degrees.
    """
    mean_longitude = circmean(np.array(longitudes), low=-180, high=180)
    while mean_longitude < -180:
        mean_longitude += 360
    while mean_longitude > 180:
        mean_longitude -= 360
    return mean_longitude


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
