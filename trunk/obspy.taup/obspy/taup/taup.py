# -*- coding: utf-8 -*-

from obspy.taup import __path__
from util import flibtaup as lib
import ctypes as C
import math
import numpy as np
import os


AVAILABLE_PHASES = ['P', "P'P'ab", "P'P'bc", "P'P'df", 'PKKPab', 'PKKPbc',
                    'PKKPdf', 'PKKSab', 'PKKSbc', 'PKKSdf', 'PKPab', 'PKPbc',
                    'PKPdf', 'PKPdiff', 'PKSab', 'PKSbc', 'PKSdf', 'PKiKP',
                    'PP', 'PS', 'PcP', 'PcS', 'Pdiff', 'Pn', 'PnPn', 'PnS',
                    'S', "S'S'ac", "S'S'df", 'SKKPab', 'SKKPbc', 'SKKPdf',
                    'SKKSac', 'SKKSdf', 'SKPab', 'SKPbc', 'SKPdf', 'SKSac',
                    'SKSdf', 'SKiKP', 'SP', 'SPg', 'SPn', 'SS', 'ScP', 'ScS',
                    'Sdiff', 'Sn', 'SnSn', 'pP', 'pPKPab', 'pPKPbc', 'pPKPdf',
                    'pPKPdiff', 'pPKiKP', 'pPdiff', 'pPn', 'pS', 'pSKSac',
                    'pSKSdf', 'pSdiff', 'sP', 'sPKPab', 'sPKPbc', 'sPKPdf',
                    'sPKPdiff', 'sPKiKP', 'sPb', 'sPdiff', 'sPg', 'sPn', 'sS',
                    'sSKSac', 'sSKSdf', 'sSdiff', 'sSn']


def getTravelTimes(delta, depth, model='iasp91'):
    """
    Returns the travel times calculated by the iaspei-tau travel time table
    package.

    :param delta: float
        Distance in degrees.
    :param depth: float
        Depth in kilometer.
    :param model: string, optional
        Either 'iasp91' or 'ak135' velocity model. Defaults to 'iasp91'.
    :return:
        A list of phase arrivals given in time order. Each phase is represented
        by a dictionary containing phase name, travel time in seconds, take-off
        angle, and various derivatives (travel time with respect to distance,
        travel time with respect to depth and the second derivative of travel
        time with respect to distance).
    """
    model_path = os.path.join(__path__[0], 'tables', model)
    if not os.path.exists(model_path + os.path.extsep + 'hed') or \
       not os.path.exists(model_path + os.path.extsep + 'tbl'):
        msg = 'Model %s not found' % model
        raise ValueError(msg)

    # Distance in degree
    delta = C.c_float(delta)
    # Depth in kilometer.
    depth = C.c_float(depth)

    # Max number of phases. Hard coded in the Fortran code. Do not change!
    max = 60

    phase_names = (C.c_char * 8 * max)()

    modnam = (C.c_char * 500)()
    modnam.value = os.path.join(__path__[0], 'tables', model)

    flags = ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']
    # Some arrays...
    tt = np.zeros(60, 'float32', flags)
    toang = np.zeros(60, 'float32', flags)
    dtdd = np.zeros(60, 'float32', flags)
    dtdh = np.zeros(60, 'float32', flags)
    dddp = np.zeros(60, 'float32', flags)

    flags = ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']

    lib.ttimes(C.byref(delta), C.byref(depth), modnam,
               phase_names,
               tt.ctypes.data_as(C.POINTER(C.c_float)),
               toang.ctypes.data_as(C.POINTER(C.c_float)),
               dtdd.ctypes.data_as(C.POINTER(C.c_float)),
               dtdh.ctypes.data_as(C.POINTER(C.c_float)),
               dddp.ctypes.data_as(C.POINTER(C.c_float)))

    phases = []
    for _i in xrange(max):
        phase_name = phase_names[_i].value.strip()
        if not phase_name:
            break
        time_dict = {
            'phase_name': phase_name,
            'time': tt[_i],
            'take-off angle': toang[_i],
            'dT/dD': dtdd[_i],
            'dT/dh': dtdh[_i],
            'd2T/dD2': dddp[_i]}
        phases.append(time_dict)
    return phases


def kilometer2degrees(kilometer, radius=6371):
    """
    Convenience function to convert kilometers to degrees assuming a perfectly
    spherical Earth.

    :param kilometer: float
        Distance in kilometers
    :param radius: int, optional
        Radius of the Earth used for the calculation.
    :return:
        Distance in degrees as a floating point number.
    """
    return kilometer / (2.0 * radius * math.pi / 360.0)


def locations2degrees(lat1, long1, lat2, long2):
    """
    Convenience function to calculate the great distance between two points on
    a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (http://code.google.com/p/geopy/).

    :param lat1: float
        Latitude of point 1 in degrees
    :param long1: float
        Longitude of point 1 in degrees
    :param lat2: float
        Latitude of point 2 in degrees
    :param long2: float
        Longitude of point 2 in degrees
    :return:
        Distance in degrees as a floating point number.
    """
    # Convert to radians.
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    long1 = math.radians(long1)
    long2 = math.radians(long2)
    long_diff = long2 - long1
    return math.degrees(math.atan2(math.sqrt((math.cos(lat2) * \
                                math.sin(long_diff)) ** 2 \
                                + (math.cos(lat1) * math.sin(lat2) - \
                                math.sin(lat1) * math.cos(lat2) * \
                                math.cos(long_diff)) ** 2),
                                math.sin(lat1) * math.sin(lat2) + \
                                math.cos(lat1) * math.cos(lat2) * \
                                math.cos(long_diff)))


def travelTimePlot(min_degree=0, max_degree=360, npoints=1000,
                   phases=AVAILABLE_PHASES, depth=100, model='iasp91'):
    """
    Basic travel time plotting function.

    :param min_degree: float, optional
        Minimum distance in degree used in plot. Defaults to 0.
    :param max_degree: float, optional
        Maximum distance in degree used in plot. Defaults to 360.
    :param npoints: int, optional
        Number of points to plot. Defaults to 1000.
    :param phases: list of strings, optional
        List of phase names which should be used within the plot. Defaults to
        all phases if not explicit set.
    :param depth: float, optional
        Depth in kilometer. Defaults to 100.
    :param model: string, optional
        Either 'iasp91' or 'ak135' velocity model. Defaults to 'iasp91'.
    """
    import matplotlib.pylab as plt

    data = {}
    for phase in phases:
        data[phase] = [[], []]

    degrees = np.linspace(min_degree, max_degree, npoints)
    # Loop over all degrees.
    for degree in degrees:
        tt = getTravelTimes(degree, depth, model)
        # Mirror if necessary.
        if degree > 180:
            degree = 180 - (degree - 180)
        for item in tt:
            phase = item['phase_name']
            try:
                data[phase][1].append(item['time'] / 60.0)
                data[phase][0].append(degree)
            except:
                data[phase][1].append(np.NaN)
                data[phase][0].append(degree)

    # Plot and some formatting.
    for key, value in data.iteritems():
        plt.plot(value[0], value[1], '.', label=key)
    plt.grid()
    plt.xlabel('Distance (degrees)')
    plt.ylabel('Time (minutes)')
    if max_degree <= 180:
        plt.xlim(min_degree, max_degree)
    else:
        plt.xlim(min_degree, 180)
    plt.legend()
    plt.show()


if __name__ == '__main__':  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
