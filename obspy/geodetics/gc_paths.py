#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import sys
import os
import numpy as np

# checking for geographiclib
try:
    import geographiclib  # @UnusedImport # NOQA
    from geographiclib.geodesic import Geodesic
    HAS_GEOGRAPHICLIB = True
except ImportError:
    HAS_GEOGRAPHICLIB = False


def get_ray_paths(evlat, evlon, evdepth, inventory, coordinate_system='RTP'):
    """
    This function returns lat, lon, depth coordinates from an event location
    to all stations in the inventory object
    """
    if not HAS_GEOGRAPHICLIB:
        raise ImportError('greatcircle routines need geographiclib')

    # extract all stations and their location
    stlats = []
    stlons = []
    labels = []
    for network in inventory:
        for station in network:
            if station.latitude is None or station.longitude is None:
                msg = ("Station '%s' does not have latitude/longitude "
                       "information and will not be plotted." % label)
                warnings.warn(msg)
                continue
            label_ = "   " + ".".join((network.code, station.code))
            stlats.append(station.latitude)
            stlons.append(station.longitude)
            labels.append(label_)

    # initialize taup model if necessary
    from obspy.taup import TauPyModel
    taup_model = TauPyModel(model="iasp91")

    # now loop through all stations and compute the greatcircles
    greatcircles = []
    for stlat, stlon, stlabel in zip(stlats, stlons, labels):
        geod = Geodesic.WGS84.Inverse(evlat, evlon, stlat, stlon)
        line = Geodesic.WGS84.Line(evlat, evlon, geod['azi1'])
        arrivals = taup_model.get_ray_paths(evdepth, geod['s12'] * 1e-3,
                                            phase_list=('P', 'Pdiff', 'PKP'))
        if len(arrivals) == 0:
            continue

        r_earth = arrivals.model.radiusOfEarth
        for arr in arrivals:
            npoints = len(arr.path)
            gcircle = np.empty((npoints, 3)) # 3 dimensions x npoints
            for ipoint, (p, time, dist, depth) in enumerate(arr.path):
                gcircle[ipoint, 0] = line.Position(r_earth * 1e3 * dist)['lat2']
                gcircle[ipoint, 1] = line.Position(r_earth * 1e3 * dist)['lon2']
                gcircle[ipoint, 2] = depth
    
            if coordinate_system == 'RTP':
                radius = (r_earth - gcircle[:, 2])/r_earth
                theta = np.radians(90. - gcircle[:, 0])
                phi = np.radians(gcircle[:, 1])
                gcircle[:, 0] = radius
                gcircle[:, 1] = theta
                gcircle[:, 2] = phi
    
            if coordinate_system == 'XYZ':
                radius = (r_earth - gcircle[:, 2])/r_earth
                theta = np.radians(90. - gcircle[:, 0])
                phi = np.radians(gcircle[:, 1])
                gcircle[:, 0] = radius * np.sin(theta) * np.cos(phi)
                gcircle[:, 1] = radius * np.sin(theta) * np.sin(phi)
                gcircle[:, 2] = radius * np.cos(theta)
    
            greatcircles.append((gcircle, arr.name, stlabel))

    return greatcircles


def plot_rays(evlat, evlon, evdepth, inventory):
    """
    plots raypaths between an event and and inventory. This could be extended
    to plot all rays between a catalogue and an inventory
    """
    # use mayavi if possible.
    try:
        from mayavi import mlab
    except Exception as err:
        print(err)
        msg = "obspy failed to import mayavi. " +\
              "You need to install the mayavi module " +\
              "(e.g. conda install mayavi, pip install mayavi). " +\
              "If it is installed and still doesn't work, " +\
              "try setting the environmental variable QT_API to " +\
              "pyqt (e.g. export QT_API=pyqt) before running the " +\
              "code. Another option is to avoid mayavi and " +\
              "directly use kind='vtk' for vtk file output of the " +\
              "radiation pattern that can be used by external " +\
              "software like paraview"
        raise ImportError(msg)

    greatcircles = get_ray_paths(evlat, evlon, evdepth, inventory,
                                 coordinate_system='XYZ')

    mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
    colordict = {'P':(0., 0.5, 0.), 'PKP':(0.5, 0., 0.), 'Pdiff':(0., 0., 0.5)}
    for gcircle, name, stlabel in greatcircles:
        color = colordict[name]
        mlab.plot3d(*gcircle.T[:, ::3], color=color, tube_radius=0.004)

    # make surface
    data_source = mlab.pipeline.open('data/coastlines.vtk')
    surface = mlab.pipeline.surface(data_source, opacity=0.3)

    # make CMB
    rad = 0.55
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    x = rad * sin(phi) * cos(theta)
    y = rad * sin(phi) * sin(theta)
    z = rad * cos(phi)
    mlab.mesh(x, y, z, color=(0, 0, 0.3), opacity=0.4)

    mlab.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
