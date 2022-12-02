# -*- coding: utf-8 -*-
"""
Various geodetic utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import math
import warnings

import numpy as np

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


def _check_latitude(latitude, variable_name='latitude'):
    """
    Check whether latitude is in the -90 to +90 range.
    """
    if latitude is None:
        return
    if latitude > 90 or latitude < -90:
        msg = '{} out of bounds! (-90 <= {} <=90)'.format(
            variable_name, variable_name)
        raise ValueError(msg)


def _normalize_longitude(longitude):
    """
    Normalize longitude in the -180 to +180 range.
    """
    if longitude is None:
        return
    while longitude > 180:
        longitude -= 360
    while longitude < -180:
        longitude += 360
    return longitude


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

        * https://www.icsm.gov.au/publications
        * https://www.icsm.gov.au/publications/gda2020-technical-manual-v16

        It states::

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
    _check_latitude(lat1, 'lat1')
    lon1 = _normalize_longitude(lon1)
    _check_latitude(lat2, 'lat2')
    lon2 = _normalize_longitude(lon2)

    b = a * (1 - f)  # semiminor axis

    if math.isclose(lat1, lat2) and math.isclose(lon1, lon2):
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
                sin_sigma

            sqr_cos_alpha = 1 - sin_alpha * sin_alpha
            if math.isclose(sqr_cos_alpha, 0):
                # Equatorial line
                cos2sigma_m = 0
            else:
                cos2sigma_m = cos_sigma - \
                    (2 * math.sin(u_1) * math.sin(u_2) / sqr_cos_alpha)

            c = (f / 16) * sqr_cos_alpha * (4 + f * (4 - 3 * sqr_cos_alpha))
            last_dlon = dlon
            dlon = omega + (1 - c) * f * sin_alpha * \
                (sigma + c * sin_sigma *
                    (cos2sigma_m + c * cos_sigma *
                        (-1 + 2 * pow(cos2sigma_m, 2))))

            iterlimit -= 1
            if iterlimit < 0:
                # iteration limit reached
                raise StopIteration
    except ValueError:
        # usually "math domain error"
        raise StopIteration

    u2 = sqr_cos_alpha * (a * a - b * b) / (b * b)
    _a = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
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
        (:func:`obspy.geodetics.base.calc_vincenty_inverse`) is used which
        has known limitations for two nearly antipodal points and is ca. 4x
        slower.
    """
    if HAS_GEOGRAPHICLIB:
        _check_latitude(lat1, 'lat1')
        _check_latitude(lat2, 'lat2')
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


def kilometers2degrees(kilometer, radius=6371.0):
    """
    Convenience function to convert kilometers to degrees assuming a perfectly
    spherical Earth.

    :type kilometer: float
    :param kilometer: Distance in kilometers
    :type radius: float, optional
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


def degrees2kilometers(degrees, radius=6371.0):
    """
    Convenience function to convert (great circle) degrees to kilometers
    assuming a perfectly spherical Earth.

    :type degrees: float
    :param degrees: Distance in (great circle) degrees
    :type radius: float, optional
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
    >>> locations2degrees(5, 5, 10, 10) # doctest: +ELLIPSIS
    7.03970141917538...
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
    from scipy.stats import circmean
    mean_longitude = circmean(np.array(longitudes), low=-180, high=180)
    mean_longitude = _normalize_longitude(mean_longitude)
    return mean_longitude


def inside_geobounds(obj, minlatitude=None, maxlatitude=None,
                     minlongitude=None, maxlongitude=None,
                     latitude=None, longitude=None,
                     minradius=None, maxradius=None):
    """
    Check whether an object is within a given latitude and/or longitude range,
    or within a given distance range from a reference geographic point.

    The object must have ``latitude`` and ``longitude`` attributes, expressed
    in degrees.

    :type obj: object
    :param obj: An object with `latitude` and `longitude` attributes.
    :type minlatitude: float
    :param minlatitude: Minimum latitude in degrees.
    :type maxlatitude: float
    :param maxlatitude: Maximum latitude in degrees. If this value is smaller
        than ``minlatitude``, then 360 degrees are added to this value (i.e.,
        wrapping around latitude of +/- 180 degrees)
    :type minlongitude: float
    :param minlongitude: Minimum longitude in degrees.
    :type maxlongitude: float
    :param maxlongitude: Minimum longitude in degrees.
    :type latitude: float
    :param latitude: Latitude of the reference point, in degrees, for distance
        range selection.
    :type longitude: float
    :param longitude: Longitude of the reference point, in degrees, for
        distance range selection.
    :type minradius: float
    :param minradius: Minimum distance, in degrees, from the reference
        geographic point defined by the latitude and longitude parameters.
    :type maxradius: float
    :param maxradius: Maximum distance, in degrees, from the reference
        geographic point defined by the latitude and longitude parameters.
    :return: ``True`` if the object is within the given range, ``False``
        otherwise.

    .. rubric:: Example

    >>> from obspy.geodetics import inside_geobounds
    >>> from obspy import read_events
    >>> ev = read_events()[0]
    >>> orig = ev.origins[0]
    >>> inside_geobounds(orig, minlatitude=40, maxlatitude=42)
    True
    >>> inside_geobounds(orig, minlatitude=40, maxlatitude=42,
    ...                  minlongitude=78, maxlongitude=79)
    False
    >>> inside_geobounds(orig, latitude=40, longitude=80,
    ...                  minradius=1, maxradius=10)
    True
    """
    if not hasattr(obj, 'latitude') or not hasattr(obj, 'longitude'):
        raise AttributeError(
            'Object must have "latitude" and "longitude" attributes.')
    olatitude = obj.latitude
    _check_latitude(olatitude, 'obj.latitude')
    _check_latitude(minlatitude, 'minlatitude')
    _check_latitude(maxlatitude, 'maxlatitude')
    _check_latitude(latitude, 'latitude')
    # Make sure longitudes are between -180 to 180 degrees
    olongitude = _normalize_longitude(obj.longitude)
    minlongitude = _normalize_longitude(minlongitude)
    maxlongitude = _normalize_longitude(maxlongitude)
    longitude = _normalize_longitude(longitude)
    if minlatitude is not None:
        if olatitude is None or olatitude < minlatitude:
            return False
    if maxlatitude is not None:
        if olatitude is None or olatitude > maxlatitude:
            return False
    # Wrap longitude around +/- 180°, if necessary
    if None not in [minlongitude, maxlongitude] \
            and maxlongitude < minlongitude:
        maxlongitude += 360
        if olongitude is not None and olongitude < minlongitude:
            olongitude += 360
    if minlongitude is not None:
        if olongitude is None or olongitude < minlongitude:
            return False
    if maxlongitude is not None:
        if olongitude is None or olongitude > maxlongitude:
            return False
    if all([coord is not None for coord in
           (latitude, longitude, olatitude, olongitude)]):
        distance = locations2degrees(latitude, longitude,
                                     olatitude, olongitude)
        if minradius is not None and distance < minradius:
            return False
        if maxradius is not None and distance > maxradius:
            return False
    return True


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
