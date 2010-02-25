#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: rotate.py
#  Purpose: Various Seismogram Rotation Functions
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megies
#---------------------------------------------------------------------
"""
Various Seismogram Rotation Functions


:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from numpy import sin, cos
from math import sqrt, pi, asin, tan, atan, atan2

def rotate_NE_RT(n, e, ba):
    """Rotates horizontal components of a seismogram:
  
    The North- and East-Component of a seismogram will be rotated in Radial
    and Transversal Component. The angle is given as the back-azimuth, that is
    defined as the angle measured between the vector pointing from the station
    to the source and the vector pointing from the station to the north.
    
    :param n: Data of the North component of the seismogram.
    :param e: Data of the East component of the seismogram.
    :param ba: The back azimuth from station to source in degrees.
    :return: Radial and Transversal component of seismogram.
    """
    if len(n) != len(e):
        raise TypeError("North and East component have different length!?!")
    if ba < 0 or ba > 360:
        raise ValueError("Back Azimuth should be between 0 and 360 degrees!")
    r = e * sin((ba + 180) * 2 * pi / 360) + n * cos((ba + 180) * 2 * pi / 360)
    t = e * cos((ba + 180) * 2 * pi / 360) - n * sin((ba + 180) * 2 * pi / 360)
    return r, t


def gps2DistAzimuth(lat1, lon1, lat2, lon2) :
    """
    Computes the distance between two geographic points on the WGS84
    ellipsoid and the forward and backward azimuths between these points.

    Latitudes should be positive for eastern/northern hemispheres and
    negative for western/southern hemispheres respectively.
    
    This code is based on an implementation incorporated in 
    Matplotlib Basemap Toolkit 0.9.5
    http://sourceforge.net/projects/matplotlib/files/
    (basemap-0.9.5/lib/matplotlib/toolkits/basemap/greatcircle.py)

    Algorithm from Geocentric Datum of Australia Technical Manual.
    http://www.icsm.gov.au/gda/gdatm/index.html
    http://www.icsm.gov.au/gda/gdatm/gdav2.3.pdf

    It states::

        Computations on the Ellipsoid

        There are a number of formulae that are available to calculate accurate
        geodetic positions, azimuths and distances on the ellipsoid.

        Vincenty's formulae (Vincenty, 1975) may be used for lines ranging from
        a few cm to nearly 20,000 km, with millimetre accuracy. The formulae
        have been extensively tested for the Australian region, by comparison
        with results from other formulae (Rainsford, 1955 & Sodano, 1965).

        * Inverse problem: azimuth and distance from known latitudes and
            longitudes
        * Direct problem: Latitude and longitude from known position, azimuth
            and distance.

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
    """
    #Check inputs
    if lat1 > 90 or lat1 < -90:
        msg = "Latitude of Point 1 out of bounds! (-90 <= lat1 <=90)"
        raise ValueError(msg)
    if lon1 > 180 or lon1 < -180:
        msg = "Longitude of Point 1 out of bounds! (-180 <= lon1 <= +180)"
        raise ValueError(msg)
    if lat2 > 90 or lat2 < -90:
        msg = "Latitude of Point 2 out of bounds! (-90 <= lat2 <=90)"
        raise ValueError(msg)
    if lon2 > 180 or lon2 < -180:
        msg = "Longitude of Point 2 out of bounds! (-180 <= lon2 <= +180)"
        raise ValueError(msg)

    #Data on the WGS84 reference ellipsoid:
    a = 6378137.0       #semimajor axis in m
    f = 1 / 298.257223563 #flattening
    b = a * (1 - f)         #semiminor axis

    if (abs(lat1 - lat2) < 1e-8) and (abs(lon1 - lon2) < 1e-8):
        return 0.0, 0.0, 0.0

    #convert latitudes and longitudes to radians:
    lat1 = lat1 * 2.0 * pi / 360.
    lon1 = lon1 * 2.0 * pi / 360.
    lat2 = lat2 * 2.0 * pi / 360.
    lon2 = lon2 * 2.0 * pi / 360.

    TanU1 = (1 - f) * tan(lat1)
    TanU2 = (1 - f) * tan(lat2)

    U1 = atan(TanU1)
    U2 = atan(TanU2)

    dlon = lon2 - lon1
    last_dlon = -4000000.0                # an impossibe value
    omega = dlon

    # Iterate until there is no significant change in dlon
    while (last_dlon < -3000000.0 or dlon != 0 and
           abs((last_dlon - dlon) / dlon) > 1.0e-9):
        sqr_sin_sigma = pow(cos(U2) * sin(dlon), 2) + \
            pow((cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(dlon)), 2)
        Sin_sigma = sqrt(sqr_sin_sigma)
        Cos_sigma = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(dlon)
        sigma = atan2(Sin_sigma, Cos_sigma)
        Sin_alpha = cos(U1) * cos(U2) * sin(dlon) / sin(sigma)
        alpha = asin(Sin_alpha)
        Cos2sigma_m = cos(sigma) - (2 * sin(U1) * sin(U2) / pow(cos(alpha), 2))
        C = (f / 16) * pow(cos(alpha), 2) * \
            (4 + f * (4 - 3 * pow(cos(alpha), 2)))
        last_dlon = dlon
        dlon = omega + (1 - C) * f * sin(alpha) * (sigma + C * sin(sigma) * \
            (Cos2sigma_m + C * cos(sigma) * (-1 + 2 * pow(Cos2sigma_m, 2))))

        u2 = pow(cos(alpha), 2) * (a * a - b * b) / (b * b)
        A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B / 4) * (Cos_sigma * \
            (-1 + 2 * pow(Cos2sigma_m, 2)) - (B / 6) * Cos2sigma_m * \
            (-3 + 4 * sqr_sin_sigma) * (-3 + 4 * pow(Cos2sigma_m, 2))))

        dist = b * A * (sigma - delta_sigma)
        alpha12 = atan2((cos(U2) * sin(dlon)),
                        (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(dlon)))
        alpha21 = atan2((cos(U1) * sin(dlon)),
                        (-sin(U1) * cos(U2) + cos(U1) * sin(U2) * cos(dlon)))

    if (alpha12 < 0.0) :
        alpha12 = alpha12 + (2.0 * pi)
    if (alpha12 > (2.0 * pi)) :
        alpha12 = alpha12 - (2.0 * pi)

    alpha21 = alpha21 + pi

    if (alpha21 < 0.0) :
        alpha21 = alpha21 + (2.0 * pi)
    if (alpha21 > (2.0 * pi)) :
        alpha21 = alpha21 - (2.0 * pi)

    #convert to degrees:
    alpha12 = alpha12 * 360 / (2.0 * pi)
    alpha21 = alpha21 * 360 / (2.0 * pi)

    return dist, alpha12, alpha21
