# -*- coding: utf-8 -*-
"""
obspy.taup - Travel time calculation tool
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import numpy as np
import os
import platform
from obspy.core.util.libnames import _get_lib_name, _load_CDLL


lib_name = _get_lib_name('tau', add_extension_suffix=False)

# Import libtau in a platform specific way.
try:
    # Linux / Mac using python import
    libtau = __import__('obspy.lib.' + lib_name, globals(), locals(),
                        ['ttimes'])
    ttimes = libtau.ttimes
except ImportError:
    # Windows using ctypes
    if platform.system() == "Windows":
        import ctypes as C
        libtau = _load_CDLL("tau")

        def ttimes(delta, depth, modnam):
            delta = C.c_float(delta)
            depth = C.c_float(abs(depth))
            # initialize some arrays...
            phase_names = (C.c_char * 8 * 60)()
            flags = ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE']
            tt = np.zeros(60, 'float32', flags)
            toang = np.zeros(60, 'float32', flags)
            dtdd = np.zeros(60, 'float32', flags)
            dtdh = np.zeros(60, 'float32', flags)
            dddp = np.zeros(60, 'float32', flags)

            libtau.ttimes_(C.byref(delta), C.byref(depth),
                           modnam.encode('ascii'), phase_names,
                           tt.ctypes.data_as(C.POINTER(C.c_float)),
                           toang.ctypes.data_as(C.POINTER(C.c_float)),
                           dtdd.ctypes.data_as(C.POINTER(C.c_float)),
                           dtdh.ctypes.data_as(C.POINTER(C.c_float)),
                           dddp.ctypes.data_as(C.POINTER(C.c_float)))
            phase_names = np.array([p.value for p in phase_names])
            return phase_names, tt, toang, dtdd, dtdh, dddp
    else:
        raise


# Directory of obspy.taup.
_taup_dir = \
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

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
    :type model: str, optional
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
    # Raise an error, otherwise libtau sends an EXIT signal. Depends on the
    # model but 800 km works for the included models.
    if depth > 800.00:
        raise ValueError("Source depth of %.2f km is too deep." % depth)
    model_path = os.path.join(_taup_dir, 'tables', model)
    if not os.path.exists(model_path + os.path.extsep + 'hed') or \
       not os.path.exists(model_path + os.path.extsep + 'tbl'):
        msg = 'Model %s not found' % model
        raise ValueError(msg)

    # Depth in kilometer.
    depth = abs(depth)

    # modnam is a string with 500 chars.
    modnam = os.path.join(_taup_dir, 'tables', model).ljust(500)

    phase_names, tt, toang, dtdd, dtdh, dddp = ttimes(delta, depth, modnam)

    phases = []
    for _i, phase in enumerate(phase_names):
        # An empty returned string will contain "\x00".
        phase_name = phase.tostring().strip().\
            replace(b"\x00", b"").decode()
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
    :type phases: list of str, optional
    :param phases: List of phase names which should be used within the plot.
        Defaults to all phases if not explicit set.
    :type depth: float, optional
    :param depth: Depth in kilometer. Defaults to ``100``.
    :type model: str, optional
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
            if phase in data:
                try:
                    data[phase][1].append(item['time'] / 60.0)
                    data[phase][0].append(degree)
                except:
                    data[phase][1].append(np.NaN)
                    data[phase][0].append(degree)
    # Plot and some formatting.
    for key, value in data.items():
        plt.plot(value[0], value[1], '.', label=key)
    plt.grid()
    plt.xlabel('Distance (degrees)')
    plt.ylabel('Time (minutes)')
    if max_degree <= 180:
        plt.xlim(min_degree, max_degree)
    else:
        plt.xlim(min_degree, 180)
    plt.legend(numpoints=1)
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
