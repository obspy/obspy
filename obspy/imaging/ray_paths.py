# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: ray_paths.py
#  Purpose: ray paths
#   Author: Matthias Meschede
#    Email: sippl@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Matthias Meschede
# --------------------------------------------------------------------
"""
Plotting spectrogram of seismograms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport


import numpy as np


def plot_rays(inventory=None, catalog=None, stlat=None, stlon=None, evlat=None,
              evlon=None, evdepth_km=None, phase_list=('P'), kind='mayavi',
              colorscheme='default'):
    """
    plots raypaths between an event and and inventory. This could be
    extended to plot all rays between a catalogue and an inventory
    """
    # use mayavi if possible.
    if kind == 'mayavi':
        _plot_rays_mayavi(
            inventory=inventory, catalog=catalog, evlat=evlat, evlon=evlon,
            evdepth_km=evdepth_km, stlat=stlat, stlon=stlon,
            phase_list=phase_list, colorscheme=colorscheme)
    elif kind == 'vtkfiles':
        _write_vtk_files(
            inventory=inventory, catalog=catalog, evlat=evlat, evlon=evlon,
            evdepth_km=evdepth_km, stlat=stlat, stlon=stlon,
            phase_list=phase_list)


def _write_vtk_files(inventory=None, catalog=None, stlat=None, stlon=None,
                     evlat=None, evlon=None, evdepth_km=None,
                     phase_list=('P')):

    # define file names
    fname_paths = 'paths.vtk'
    fname_events = 'events.vtk'
    fname_stations = 'stations.vtk'

    # get 3d paths for all station/event combinations
    greatcircles = get_ray_paths(
        inventory=inventory, catalog=catalog, stlat=stlat, stlon=stlon,
        evlat=evlat, evlon=evlon, evdepth_km=evdepth_km, phase_list=phase_list,
        coordinate_system='XYZ')

    # now assemble all points, stations and connectivity
    stations = []  # unique list of stations
    events = []  # unique list of events
    lines = []  # contains the points that constitute each ray
    npoints_tot = 0  # this is the total number of points of all rays
    istart_ray = 0  # this is the first point of each ray in the loop
    points = []
    for gcircle, name, stlabel, evlabel in greatcircles:
        points_ray = gcircle[:, ::3]  # use every third point
        ndim, npoints_ray = points_ray.shape
        iend_ray = istart_ray + npoints_ray
        connect_ray = np.arange(istart_ray, iend_ray, dtype=int)

        lines.append(connect_ray)
        points.append(points_ray)
        if stlabel not in stations:
            stations.append((gcircle[:, -1], stlabel))

        if evlabel not in events:
            events.append((gcircle[:, 0], evlabel))

        npoints_tot += npoints_ray
        istart_ray += npoints_ray

    # write the ray paths in one file
    with open(fname_paths, 'w') as vtk_file:
        # write some header information
        vtk_header = ('# vtk DataFile Version 2.0\n'
                      '3d ray paths\n'
                      'ASCII\n'
                      'DATASET UNSTRUCTURED_GRID\n'
                      'POINTS {:d} float\n'.format(npoints_tot))
        vtk_file.write(vtk_header)

        # write a long list of all points
        all_points = np.hstack(points)
        for x, y, z in np.hstack(points).T:
            vtk_file.write('{:.4e} {:.4e} {:.4e}\n'.format(x, y, z))

        # now write connectivity
        nlines = len(lines)
        npoints_connect = npoints_tot + nlines
        vtk_file.write('CELLS {:d} {:d}\n'.format(nlines, npoints_connect))
        for line in lines:
            vtk_file.write('{:d} '.format(len(line)))
            for ipoint in line:
                if ipoint % 30 == 29:
                    vtk_file.write('\n')
                vtk_file.write('{:d} '.format(ipoint))

        # cell types. 4 means cell type is a poly_line
        vtk_file.write('\nCELL_TYPES {:d}\n'.format(nlines))
        for line in lines:
            vtk_file.write('4\n')

    # write the stations in another file
    with open(fname_stations, 'w') as vtk_file:
        # write some header information
        vtk_header = ('# vtk DataFile Version 2.0\n'
                      'station locations\n'
                      'ASCII\n'
                      'DATASET UNSTRUCTURED_GRID\n'
                      'POINTS {:d} float\n'.format(len(stations)))
        vtk_file.write(vtk_header)

        # write a long list of all points
        for location, stlabel in stations:
            vtk_file.write('{:.4e} {:.4e} {:.4e}\n'.format(*location))

    # write the events in another file
    with open(fname_events, 'w') as vtk_file:
        # write some header information
        vtk_header = ('# vtk DataFile Version 2.0\n'
                      'event locations\n'
                      'ASCII\n'
                      'DATASET UNSTRUCTURED_GRID\n'
                      'POINTS {:d} float\n'.format(len(events)))
        vtk_file.write(vtk_header)

        # write a long list of all points
        for location, evlabel in events:
            vtk_file.write('{:.4e} {:.4e} {:.4e}\n'.format(*location))


def _plot_rays_mayavi(inventory=None, catalog=None, stlat=None, stlon=None,
                      evlat=None, evlon=None, evdepth_km=None,
                      phase_list=('P'), colorscheme='default'):
    try:
        from mayavi import mlab
    except Exception as err:
        print(err)
        msg = ("obspy failed to import mayavi. "
               "You need to install the mayavi module "
               "(e.g. conda install mayavi, pip install mayavi). "
               "If it is installed and still doesn't work, "
               "try setting the environmental variable QT_API to "
               "pyqt (e.g. export QT_API=pyqt) before running the "
               "code. Another option is to avoid mayavi and "
               "directly use kind='vtk' for vtk file output of the "
               "radiation pattern that can be used by external "
               "software like paraview")
        raise ImportError(msg)

    greatcircles = get_ray_paths(
        inventory=inventory, catalog=catalog, stlat=stlat, stlon=stlon,
        evlat=evlat, evlon=evlon, evdepth_km=evdepth_km, phase_list=phase_list,
        coordinate_system='XYZ')

    # define colors and style
    if colorscheme == 'dark' or colorscheme == 'default':
        # colors:
        colordict = {'P': (0., 0.5, 0.), 'PKP': (0.5, 0., 0.),
                     'Pdiff': (0., 0., 0.5), 'PKiKP': (0.5, 0.5, 0.),
                     'PKIKP': (0., 0.5, 0.5), 'PPP': (0.3, 0.8, 0.2),
                     'PcP': (0.8, 0.2, 0.3)}
        labelcolor = (1.0, 0.7, 0.7)
        continentcolor = (0.1, 0.1, 0.1)
        eventcolor = (0.7, 1.0, 0.7)
        cmbcolor = (0.0, 0.0, 0.2)
        bgcolor = (0, 0, 0)
        # sizes:
        tube_width = 0.001
        sttextsize = (0.01, 0.01, 0.01)
        stmarkersize = 0.01
        evtextsize = (0.01, 0.01, 0.01)
        evmarkersize = 0.01
    elif colorscheme == 'bright':
        # colors:
        colordict = {'P': (0., 0.3, 0.), 'PKP': (0.3, 0., 0.),
                     'Pdiff': (0., 0., 0.3), 'PKiKP': (0.3, 0.3, 0.),
                     'PKIKP': (0., 0.3, 0.3), 'PPP': (0.3, 0.8, 0.2)}
        labelcolor = (0.2, 0.0, 0.0)
        continentcolor = (0.9, 0.9, 0.9)
        eventcolor = (0.0, 0.2, 0.0)
        cmbcolor = (0.7, 0.7, 1.0)
        bgcolor = (1, 1, 1)
        # sizes:
        # everything has to be larger in bright background plot because it
        # is more difficult to read
        tube_width = 0.003
        sttextsize = (0.02, 0.02, 0.02)
        stmarkersize = 0.02
        evtextsize = (0.02, 0.02, 0.02)
        evmarkersize = 0.06
    else:
        raise ValueError('colorscheme {:s} not recognized'.format(colorscheme))

    fig = mlab.figure(size=(800, 800), bgcolor=bgcolor)

    # loop through, and plot all paths and their labels
    fig.scene.disable_render = True  # faster rendering trick (?)
    plotted_stations = []
    plotted_events = []
    for gcircle, name, stlabel, evlabel in greatcircles:
        color = colordict[name]
        # use only every third point for plotting
        mlab.plot3d(*gcircle[:, ::10], color=color, tube_sides=3,
                    tube_radius=tube_width)

        if stlabel not in plotted_stations:
            mlab.points3d(gcircle[0, -1], gcircle[1, -1], gcircle[2, -1],
                          scale_factor=stmarkersize, color=labelcolor)
            mlab.text3d(gcircle[0, -1], gcircle[1, -1], gcircle[2, -1],
                        stlabel, scale=sttextsize,
                        color=labelcolor)
            plotted_stations.append(stlabel)

        if evlabel not in plotted_events:
            mlab.points3d(gcircle[0, 0], gcircle[1, 0], gcircle[2, 0],
                          scale_factor=evmarkersize, color=eventcolor,
                          mode='2dtriangle')
            mlab.text3d(gcircle[0, 0], gcircle[1, 0], gcircle[2, 0],
                        evlabel, scale=evtextsize,
                        color=eventcolor)
            plotted_events.append(evlabel)

    fig.scene.disable_render = False

    # make surface
    data_source = mlab.pipeline.open('data/coastlines.vtk')
    mlab.pipeline.surface(data_source, opacity=1.0, color=continentcolor)

    # make CMB sphere
    rad = 0.55
    phi, theta = np.mgrid[0:np.pi:51j, 0:2 * np.pi:51j]

    x = rad * np.sin(phi) * np.cos(theta)
    y = rad * np.sin(phi) * np.sin(theta)
    z = rad * np.cos(phi)
    mlab.mesh(x, y, z, color=cmbcolor, opacity=0.4)

    mlab.show()


def get_ray_paths(inventory=None, catalog=None, stlat=None, stlon=None,
                  evlat=None, evlon=None, evdepth_km=None, phase_list=('P'),
                  coordinate_system='XYZ'):
    """
    This function returns lat, lon, depth coordinates from an event
    location to all stations in the inventory object
    """
    # make a big list of station coordinates and names
    stlats = []
    stlons = []
    stlabels = []
    if inventory is not None:
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
                stlabels.append(label_)
    elif stlat is not None and stlon is not None:
        stlats.append(stlat)
        stlons.append(stlon)
    else:
        raise ValueError("either inventory or stlat and stlon have to be set")

    # make a big list of event coordinates and names
    evlats = []
    evlons = []
    evdepths = []
    evlabels = []
    if catalog is not None:
        for event in catalog:
            if not event.origins:
                msg = ("Event '%s' does not have an origin and will not be "
                       "plotted." % str(event.resource_id))
                warnings.warn(msg)
                continue
            if not event.magnitudes:
                msg = ("Event '%s' does not have a magnitude and will not be "
                       "plotted." % str(event.resource_id))
                warnings.warn(msg)
                continue
            origin = event.preferred_origin() or event.origins[0]
            evlats.append(origin.latitude)
            evlons.append(origin.longitude)
            evdepths.append(origin.get('depth') * 1e-3)
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            mag = magnitude.mag
            evlabels.append('  %.1f' % mag)
    elif evlat is not None and evlon is not None and evdepth_km is not None:
        evlats.append(evlat)
        evlons.append(evlon)
        evdepths.append(evdepth_km)
        evlabels.append('')
    else:
        raise ValueError("either catalog or evlat, evlon and evdepth_km have "
                         "to be set")

    # initialize taup model
    from obspy.taup import TauPyModel
    model = TauPyModel(model="iasp91")

    # now loop through all stations and source combinations
    greatcircles = []
    for stlat, stlon, stlabel in zip(stlats, stlons, stlabels):
        for evlat, evlon, evdepth_km, evlabel in zip(evlats, evlons, evdepths,
                                                     evlabels):
            arrivals = model.get_ray_paths_geo(
                    evlat, evlon, evdepth_km, stlat, stlon,
                    phase_list=phase_list)
            if len(arrivals) == 0:
                continue
    
            r_earth = arrivals.model.radiusOfEarth
            for arr in arrivals:
                if coordinate_system == 'RTP':
                    radii = (r_earth - arr.path['depth']) / r_earth
                    thetas = np.radians(90. - arr.path['lat'])
                    phis = np.radians(arr.path['lon'])
                    gcircle = np.array([radii, thetas, phis])
    
                if coordinate_system == 'XYZ':
                    radii = (r_earth - arr.path['depth']) / r_earth
                    thetas = np.radians(90. - arr.path['lat'])
                    phis = np.radians(arr.path['lon'])
                    gcircle = np.array([radii * np.sin(thetas) * np.cos(phis),
                                        radii * np.sin(thetas) * np.sin(phis),
                                        radii * np.cos(thetas)])
    
                greatcircles.append((gcircle, arr.name, stlabel, evlabel))

    return greatcircles
