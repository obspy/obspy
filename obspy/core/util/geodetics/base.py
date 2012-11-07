# -*- coding: utf-8 -*-
"""
Various geodetic utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import math
import numpy as np
import warnings


def calcVincentyInverse(lat1, lon1, lat2, lon2):
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

        * http://www.icsm.gov.au/gda/gdatm/index.html
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

    # Data on the WGS84 reference ellipsoid:
    a = 6378137.0          # semimajor axis in m
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)        # semiminor axis

    if (abs(lat1 - lat2) < 1e-8) and (abs(lon1 - lon2) < 1e-8):
        return 0.0, 0.0, 0.0

    # convert latitudes and longitudes to radians:
    lat1 = lat1 * 2.0 * math.pi / 360.
    lon1 = lon1 * 2.0 * math.pi / 360.
    lat2 = lat2 * 2.0 * math.pi / 360.
    lon2 = lon2 * 2.0 * math.pi / 360.

    TanU1 = (1 - f) * math.tan(lat1)
    TanU2 = (1 - f) * math.tan(lat2)

    U1 = math.atan(TanU1)
    U2 = math.atan(TanU2)

    dlon = lon2 - lon1
    last_dlon = -4000000.0  # an impossible value
    omega = dlon

    # Iterate until no significant change in dlon or iterlimit has been
    # reached (http://www.movable-type.co.uk/scripts/latlong-vincenty.html)
    iterlimit = 100
    try:
        while (last_dlon < -3000000.0 or dlon != 0 and
               abs((last_dlon - dlon) / dlon) > 1.0e-9):
            sqr_sin_sigma = pow(math.cos(U2) * math.sin(dlon), 2) + \
                pow((math.cos(U1) * math.sin(U2) - math.sin(U1) * \
                     math.cos(U2) * math.cos(dlon)), 2)
            Sin_sigma = math.sqrt(sqr_sin_sigma)
            Cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * \
                math.cos(U2) * math.cos(dlon)
            sigma = math.atan2(Sin_sigma, Cos_sigma)
            Sin_alpha = math.cos(U1) * math.cos(U2) * math.sin(dlon) / \
                math.sin(sigma)
            alpha = math.asin(Sin_alpha)
            Cos2sigma_m = math.cos(sigma) - (2 * math.sin(U1) * \
                math.sin(U2) / pow(math.cos(alpha), 2))
            C = (f / 16) * pow(math.cos(alpha), 2) * \
                (4 + f * (4 - 3 * pow(math.cos(alpha), 2)))
            last_dlon = dlon
            dlon = omega + (1 - C) * f * math.sin(alpha) * (sigma + C * \
                math.sin(sigma) * (Cos2sigma_m + C * math.cos(sigma) * \
                                   (-1 + 2 * pow(Cos2sigma_m, 2))))

            u2 = pow(math.cos(alpha), 2) * (a * a - b * b) / (b * b)
            A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
            B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
            delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B / 4) * \
                (Cos_sigma * (-1 + 2 * pow(Cos2sigma_m, 2)) - (B / 6) * \
                Cos2sigma_m * (-3 + 4 * sqr_sin_sigma) * \
                (-3 + 4 * pow(Cos2sigma_m, 2))))

            dist = b * A * (sigma - delta_sigma)
            alpha12 = math.atan2((math.cos(U2) * math.sin(dlon)),
                (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * \
                 math.cos(dlon)))
            alpha21 = math.atan2((math.cos(U1) * math.sin(dlon)),
                (-math.sin(U1) * math.cos(U2) + math.cos(U1) * math.sin(U2) * \
                 math.cos(dlon)))
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


def gps2DistAzimuth(lat1, lon1, lat2, lon2):
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
    :return: (Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees)

    .. note::
        This function will check if you have installed the Python module
        `geographiclib <http://geographiclib.sf.net>`_ - a very fast module
        for converting between geographic, UTM, UPS, MGRS, and geocentric
        coordinates, for geoid calculations, and for solving geodesic problems.
        Otherwise the locally implemented Vincenty's Inverse formulae
        (:func:`obspy.core.util.geodetics.calcVincentyInverse`) is used which
        has known limitations for two nearly antipodal points and is ca. 4x
        slower.
    """
    try:
        # try using geographiclib
        from geographiclib.geodesic import Geodesic
        result = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        azim = result['azi1']
        if azim < 0:
            azim += 360
        bazim = result['azi2'] + 180
        return (result['s12'], azim, bazim)
    except ImportError:
        pass
    try:
        values = calcVincentyInverse(lat1, lon1, lat2, lon2)
        if np.alltrue(np.isnan(values)):
            raise StopIteration
        return values
    except StopIteration:
        msg = "Catching unstable calculation on antipodes. " + \
              "The currently used Vincenty's Inverse formulae " + \
              "has known limitations for two nearly antipodal points. " + \
              "Install the Python module 'geographiclib' to solve this issue."
        warnings.warn(msg)
        return (20004314.5, 0.0, 0.0)
    except ValueError, e:
        raise e


def kilometer2degrees(kilometer, radius=6371):
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

    >>> from obspy.core.util import kilometer2degrees
    >>> kilometer2degrees(300)
    2.6979648177561915
    """
    return kilometer / (2.0 * radius * math.pi / 360.0)


def locations2degrees(lat1, long1, lat2, long2):
    """
    Convenience function to calculate the great distance between two points on
    a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (http://code.google.com/p/geopy/).

    :type lat1: float
    :param lat1: Latitude of point 1 in degrees
    :type long1: float
    :param long1: Longitude of point 1 in degrees
    :type lat2: float
    :param lat2: Latitude of point 2 in degrees
    :type long2: float
    :param long2: Longitude of point 2 in degrees
    :rtype: float
    :return: Distance in degrees as a floating point number.

    .. rubric:: Example

    >>> from obspy.core.util import locations2degrees
    >>> locations2degrees(5, 5, 10, 10)
    7.0397014191753815
    """
    # Convert to radians.
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    long1 = math.radians(long1)
    long2 = math.radians(long2)
    long_diff = long2 - long1
    gd = math.degrees(math.atan2(math.sqrt((math.cos(lat2) * \
        math.sin(long_diff)) ** 2 + (math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(long_diff)) ** 2),
        math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * \
        math.cos(long_diff)))
    return gd


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
