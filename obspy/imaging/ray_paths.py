#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting 3D ray paths.

:author:
    Matthias Meschede
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import warnings
import numpy as np

import obspy.geodetics.base as geodetics


def _write_vtk_files(inventory, catalog,
                     phase_list=('P'), taup_model='iasp91'):
    """
    internal vtk output routine. Check out the plot_rays routine
    for more information on the parameters
    """
    # define file names
    fname_paths = 'paths.vtk'
    fname_events = 'events.vtk'
    fname_stations = 'stations.vtk'

    # get 3d paths for all station/event combinations
    greatcircles = get_ray_paths(
        inventory, catalog, phase_list=phase_list,
        coordinate_system='XYZ', taup_model=taup_model)

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


def get_ray_paths(inventory, catalog, phase_list=['P'],
                  coordinate_system='XYZ', taup_model='iasp91'):
    """
    This function returns lat, lon, depth coordinates from an event
    location to all stations in the inventory object

    :param inventory: an obspy station inventory
    :param catalog: an obspy event catalog
    :param phase_list: a list of seismic phase names that is passed to taup
    :param coordinate_system: can be either 'XYZ' or 'RTP'.
    :param taup_model: the taup model for which the greatcircle paths are
                  computed
    :returns: a list of tuples
              [(gcircle, phase_name, value, station_label, event_label), ...]
              gcircle is a [3, npoints] array with the path coordinates.
              phase_name is the name of the seismic phase,
              value is a certain value that can be chosen, like the radiation
              pattern intensity.
              station_label is the name of the station that belongs to the path
              event_label is the name of the event that belongs to the path
    """
    # GEOGRAPHICLIB is mandatory for this function
    if not geodetics.HAS_GEOGRAPHICLIB:
        raise ImportError('Geographiclib not found but required by ray path '
                          'routine')

    stlats = []
    stlons = []
    stlabels = []
    for network in inventory:
        for station in network:
            label_ = "   " + ".".join((network.code, station.code))
            if station.latitude is None or station.longitude is None:
                msg = ("Station '%s' does not have latitude/longitude "
                       "information and will not be plotted." % label_)
                warnings.warn(msg)
                continue
            stlats.append(station.latitude)
            stlons.append(station.longitude)
            stlabels.append(label_)

    # make a big list of event coordinates and names
    # this part should be included as a subroutine of catalog that extracts
    # a list of event properties. E.g. catalog.extract(['latitiude',
    # 'longitude', 'depth', 'mag', 'focal_mechanism') The same should be done
    # for an inventory with stations
    evlats = []
    evlons = []
    evdepths = []
    evlabels = []
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
        if not origin.get('depth'):
            origin.depth = 0.
        evdepths.append(origin.get('depth') * 1e-3)
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        mag = magnitude.mag
        label = '  {:s} | M{:.1f}'.format(str(origin.time.date), mag)
        evlabels.append(label)

    # initialize taup model if it is not provided
    if isinstance(taup_model, str):
        from obspy.taup import TauPyModel
        model = TauPyModel(model=taup_model)
    else:
        model = taup_model

    # now loop through all stations and source combinations
    r_earth = model.model.radius_of_planet
    greatcircles = []
    for stlat, stlon, stlabel in zip(stlats, stlons, stlabels):
        for evlat, evlon, evdepth_km, evlabel in zip(evlats, evlons, evdepths,
                                                     evlabels):
            arrivals = model.get_ray_paths_geo(
                    evdepth_km, evlat, evlon, stlat, stlon,
                    phase_list=phase_list, resample=True)
            if len(arrivals) == 0:
                continue

            for arr in arrivals:
                radii = (r_earth - arr.path['depth']) / r_earth
                thetas = np.radians(90. - arr.path['lat'])
                phis = np.radians(arr.path['lon'])

                if coordinate_system == 'RTP':
                    gcircle = np.array([radii, thetas, phis])

                if coordinate_system == 'XYZ':
                    gcircle = np.array([radii * np.sin(thetas) * np.cos(phis),
                                        radii * np.sin(thetas) * np.sin(phis),
                                        radii * np.cos(thetas)])

                value = 0.
                greatcircles.append((gcircle, arr.name, value, stlabel,
                                     evlabel))

    return greatcircles


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
