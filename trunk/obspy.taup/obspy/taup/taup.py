# -*- coding: utf-8 -*-

from obspy.taup import __path__
from util import flibtaup as lib
import ctypes as C
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


def getTravelTimes(distance, depth, model='iasp91'):
    """
    Returns the travel times calculated by the iaspei-tau traveltime table
    package
    """
    model_path = os.path.join(__path__[0], 'tables', model)
    if not os.path.exists(model_path + os.path.extsep + 'hed') or \
       not os.path.exists(model_path + os.path.extsep + 'tbl'):
        msg = 'Model %s not found' % model
        raise ValueError(msg)

    # Distance in degree
    delta = C.c_float(distance)
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

    lib.ttimes_(C.byref(delta), C.byref(depth), modnam,
              phase_names,
              tt.ctypes.data_as(C.POINTER(C.c_float)),
              toang.ctypes.data_as(C.POINTER(C.c_float)),
              dtdd.ctypes.data_as(C.POINTER(C.c_float)),
              dtdh.ctypes.data_as(C.POINTER(C.c_float)),
              dddp.ctypes.data_as(C.POINTER(C.c_float)))

    phases = {}

    for _i in xrange(max):
        phase_name = phase_names[_i].value.strip()
        if not phase_name:
            break
        time_dict = {
            'phase_name': phase_name,
            'tt': tt[_i],
            'toang': toang[_i],
            'dtdd': dtdd[_i],
            'dtdh': dtdh[_i],
            'dddp': dddp[_i]}
        phases[phase_name] = time_dict
    return phases


def travelTimePlot(min_degree=0, max_degree=360, npoints=1000,
                   phases=AVAILABLE_PHASES):
    """
    Travel time plot.
    """
    import matplotlib.pylab as plt

    data = {}
    for phase in phases:
        data[phase] = []

    degrees = np.linspace(min_degree, max_degree, npoints)
    x_values = []
    # Loop over all degrees.
    for degree in degrees:
        tt = getTravelTimes(degree, 100)
        # Mirror if necessary.
        if degree > 180:
            degree = 180 - (degree - 180)
        x_values.append(degree)
        for phase in phases:
            try:
                data[phase].append(tt[phase]['tt'] / 60.0)
            except:
                data[phase].append(np.NaN)

    # Plot and some formatting.
    for key, value in data.iteritems():
        plt.plot(x_values, value, label=key)
    plt.grid()
    plt.xlabel('Distance (degrees)')
    plt.ylabel('Time (minutes)')
    if max_degree <= 180:
        plt.xlim(min_degree, max_degree)
    else:
        plt.xlim(min_degree, 180)
    plt.legend()
    plt.show()
