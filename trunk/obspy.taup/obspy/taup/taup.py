# -*- coding: utf-8 -*-

from obspy.taup import __path__
from util import flibtaup as lib
import ctypes as C
import math
import numpy as np
import os


AVAILABLE_PHASES = [
    'P', "P'P'ab", "P'P'bc", "P'P'df", 'PKKPab', 'PKKPbc', 'PKKPdf', 'PKKSab',
    'PKKSbc', 'PKKSdf', 'PKPab', 'PKPbc', 'PKPdf', 'PKPdiff', 'PKSab', 'PKSbc',
    'PKSdf', 'PKiKP', 'PP', 'PS', 'PcP', 'PcS', 'Pdiff', 'Pn', 'PnPn', 'PnS',
    'S', "S'S'ac", "S'S'df", 'SKKPab', 'SKKPbc', 'SKKPdf', 'SKKSac', 'SKKSdf',
    'SKPab', 'SKPbc', 'SKPdf', 'SKSac', 'SKSdf', 'SKiKP', 'SP', 'SPg', 'SPn',
    'SS', 'ScP', 'ScS', 'Sdiff', 'Sn', 'SnSn', 'pP', 'pPKPab', 'pPKPbc',
    'pPKPdf', 'pPKPdiff', 'pPKiKP', 'pPdiff', 'pPn', 'pS', 'pSKSac', 'pSKSdf',
    'pSdiff', 'sP', 'sPKPab', 'sPKPbc', 'sPKPdf', 'sPKPdiff', 'sPKiKP', 'sPb',
    'sPdiff', 'sPg', 'sPn', 'sS', 'sSKSac', 'sSKSdf', 'sSdiff', 'sSn']


def getTravelTimes(delta, depth, model='iasp91'):
    """
    Returns the travel times calculated by the iaspei-tau, a travel time
    library by Arthur Snoke (http://www.iris.edu/pub/programs/iaspei-tau/).

    :type delta: float
    :param delta: Distance in degrees.
    :type depth: float
    :param depth: Depth in kilometer.
    :type model: string, optional
    :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model. Defaults
        to ``'iasp91'``.
    :rtype: list of dicts
    :return:
        A list of phase arrivals given in time order. Each phase is represented
        by a dictionary containing phase name, travel time in seconds, take-off
        angle, and various derivatives (travel time with respect to distance,
        travel time with respect to depth and the second derivative of travel
        time with respect to distance).

    .. rubric:: Example

    >>> from obspy.taup.taup import getTravelTimes
    >>> tt = getTravelTimes(delta=52.474, depth=611.0, model='ak135')
    >>> len(tt)
    24
    >>> tt[0]  #doctest: +SKIP
    {'phase_name': 'P', 'dT/dD': 7.1050525, 'take-off angle': 45.169445,
     'time': 497.53741, 'd2T/dD2': -0.0044748308, 'dT/dh': -0.070258446}
    """
    model_path = os.path.join(__path__[0], 'tables', model)
    if not os.path.exists(model_path + os.path.extsep + 'hed') or \
       not os.path.exists(model_path + os.path.extsep + 'tbl'):
        msg = 'Model %s not found' % model
        raise ValueError(msg)

    # Distance in degree
    delta = C.c_float(delta)
    # Depth in kilometer.
    depth = C.c_float(abs(depth))

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

    :type kilometer: float
    :param kilometer: Distance in kilometers
    :type radius: int, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in degrees as a floating point number.

    .. rubric:: Example

    >>> from obspy.taup.taup import kilometer2degrees
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

    >>> from obspy.taup.taup import locations2degrees
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


def travelTimePlot(min_degree=0, max_degree=360, npoints=1000,
                   phases=None, depth=100, model='iasp91'):
    """
    Basic travel time plotting function.

    :type min_degree: float, optional
    :param min_degree: Minimum distance in degree used in plot.
        Defaults to ``0``.
    :type max_degree: float, optional
    :param max_degree: Maximum distance in degree used in plot.
        Defaults to ``360``.
    :type npoints: int, optional
    :param npoints: Number of points to plot. Defaults to ``1000``.
    :type phases: list of strings, optional
    :param phases: List of phase names which should be used within the plot.
        Defaults to all phases if not explicit set.
    :type depth: float, optional
    :param depth: Depth in kilometer. Defaults to ``100``.
    :type model: string, optional
    :param model: Either ``'iasp91'`` or ``'ak135'`` velocity model.
        Defaults to ``'iasp91'``.
    :return: None

    .. rubric:: Example

    >>> from obspy.taup.taup import travelTimePlot
    >>> travelTimePlot(min_degree=0, max_degree=50, phases=['P', 'S', 'PP'],
    ...                depth=120, model='iasp91')  #doctest: +SKIP

    .. plot::

        from obspy.taup.taup import travelTimePlot
        travelTimePlot(min_degree=0, max_degree=50, phases=['P', 'S', 'PP'],
                       depth=120, model='iasp91')
    """
    import matplotlib.pylab as plt

    data = {}
    if not phases:
        phases = AVAILABLE_PHASES
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
            # Check if this phase should be plotted.
            if 'phase' in data:
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
