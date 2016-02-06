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

    evcoords = (evlat, evlon, evdepth)
    greatcircles = inventory.get_ray_paths(evcoords, coordinate_system='XYZ')

    fig = mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
    colordict = {'P':(0., 0.5, 0.), 'PKP':(0.5, 0., 0.), 'Pdiff':(0., 0., 0.5)}
    fig.scene.disable_render = True # Super duper trick
    for gcircle, name, stlabel in greatcircles:
        color = colordict[name]
        # use only every third point for plotting
        mlab.plot3d(*gcircle[:, ::3], color=color, tube_sides=3,
                    tube_radius=0.004)
        mlab.points3d(gcircle[0, -1], gcircle[1, -1], gcircle[2, -1],
                      scale_factor=0.01, color=(0.8, 0.8, 0.8))
        mlab.text3d(gcircle[0, -1], gcircle[1, -1], gcircle[2, -1], stlabel,
                    scale=(0.01, 0.01, 0.01), color=(0.8, 0.8, 0.8))
    fig.scene.disable_render = False

    # make surface
    data_source = mlab.pipeline.open('data/coastlines.vtk')
    surface = mlab.pipeline.surface(data_source, opacity=1.0, color=(0.5,0.5,0.5))

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
