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

from colorsys import hls_to_rgb, rgb_to_hls

import warnings
import numpy as np
from matplotlib.colors import hex2color

import ..geodetics.base as geodetics


def plot_rays(inventory, catalog, phase_list=('P',),
              kind='mayavi', colorscheme='default', animate=False,
              savemovie=False, figsize=(800, 800), coastlines='internal',
              taup_model='iasp91', icol=0, event_labels=True,
              station_labels=True, fname_out=None, view_dict=None):
    """
    Ray path plotting routine

    :param inventory: an obspy station inventory
    :param catalog: an obspy event catalog
    :param phase_list: a list of seismic phase names that is passed to taup
    :param kind: selects the plot type.
                 'mayavi' uses the mayavi library for plotting
                 'vtkfiles' outputs vtk data files that can be used by
                 third party software for visualization to the current
                 working directory.
    :param colorscheme: one of 'default', 'dark', or 'bright'
    :param animate: if set to True, mayavi is going to rotate the Earth
              around it's axis
    :param savemovie: if set to True and animate is set to True as well,
              mayavi is going to write png files of each frame to the current
              working directory.
    :param figsize: the figure size of the mayavi plot
    :param coastlines: if 'internal' mayavi uses internal coastlines.
              Alternatively it can be set to the path of a vtk file with
              higher resolution coastlines (e.g. data/coastlines.vtk)
    :param taup_model: the taup model for which the greatcircle paths are
                  computed
    :param icol: integer that changes the color. It cycles through the
              internal colorcycle of the different phases.
    :param event_labels: if True, plots a little label next to the events
    :param station_labels: if True, plots a little label next to each station
    :param fname_out: file name under which the plot is saved. Uses offscreen
              rendering which can be useful if mayavi or OpenGl make problems
    :param view_dict: mayaview view dictionary that defines the initial
                      perspective of the 3d camera. E.g.:
                     {'elevation': 80, 'azimuth': -20, 'distance': 4.,
                     'roll': 0, '', 'focalpoint': (0., 0., 0.)}

    .. rubric:: Examples

    >>> from obspy import read_events, read_inventory
    >>> from obspy.imaging.ray_paths import plot_rays
    >>> inventory = read_inventory("/path/to/IU_stations.txt")
    >>> catalog = read_events()
    >>> plot_rays(inventory, catalog, phase_list=['Pdiff'])  # doctest: +SKIP

    .. figure:: /_images/expensive_plots/mayavi_ray_paths.png
    """

    if kind == 'mayavi':
        _plot_rays_mayavi(
            inventory, catalog, phase_list=phase_list,
            colorscheme=colorscheme, animate=animate, savemovie=savemovie,
            figsize=figsize, taup_model='iasp91', coastlines=coastlines,
            icol=icol, event_labels=event_labels,
            station_labels=station_labels, fname_out=fname_out,
            view_dict=view_dict)
    elif kind == 'vtkfiles':
        _write_vtk_files(
            inventory, catalog,
            phase_list=phase_list)
    else:
        raise NotImplementedError


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


def _plot_rays_mayavi(inventory, catalog, phase_list=['P'],
                      colorscheme='default', animate=False,
                      savemovie=False, figsize=(800, 800), taup_model='iasp91',
                      coastlines='internal', icol=0, event_labels=True,
                      station_labels=True, fname_out=None, view_dict=None):
    """
    internal mayavi plotting routine. Check out the plot_rays routine
    for more information on the parameters
    """
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

    if isinstance(taup_model, str):
        from obspy.taup import TauPyModel
        model = TauPyModel(model=taup_model)
    else:
        model = taup_model

    if fname_out is not None:
        offscreen = True
    else:
        offscreen = False

    nphases = len(phase_list)

    greatcircles = get_ray_paths(
        inventory=inventory, catalog=catalog, phase_list=phase_list,
        coordinate_system='XYZ', taup_model=model)

    if len(greatcircles) == 0:
        raise ValueError('no paths found for the input stations and events')

    # define colorschemes
    if colorscheme == 'dark' or colorscheme == 'default':
        # we use the color set that is used in taup, but adjust the lightness
        # to get nice shiny rays that are well visible in the dark 3d plots
        from obspy.taup.tau import COLORS
        ncolors = len(COLORS)
        color_hues = [rgb_to_hls(*hex2color(col))[0] for col in COLORS]
        # swap green and red to start with red:
        color_hues[2], color_hues[1] = color_hues[1], color_hues[2]
        # first color is for the continents etc:
        continent_hue = color_hues[0]
        event_hue = color_hues[-1]
        # the remaining colors are for the rays:
        ray_hues = color_hues[1: -1]

        # now convert all of the hues to rgb colors:
        ray_light = 0.45
        ray_sat = 1.0
        raycolors = [hls_to_rgb(ray_hues[(iphase + icol) % (ncolors - 2)],
                     ray_light, ray_sat) for iphase in range(nphases)]
        stationcolor = hls_to_rgb(continent_hue, 0.7, 0.7)
        continentcolor = hls_to_rgb(continent_hue, 0.3, 0.2)
        eventcolor = hls_to_rgb(event_hue, 0.7, 0.7)
        cmbcolor = continentcolor
        bgcolor = (0, 0, 0)

        # sizes:
        sttextsize = (0.015, 0.015, 0.015)
        stmarkersize = 0.01
        evtextsize = (0.015, 0.015, 0.015)
        evmarkersize = 0.05

    elif colorscheme == 'bright':
        # colors (a distinct colors for each phase):
        lightness = 0.2
        saturation = 1.0
        ncolors = nphases + 2  # two extra colors for continents and events
        hues = np.linspace(0., 1. - 1./ncolors, ncolors)

        raycolors = [hls_to_rgb(hue, lightness, saturation) for hue
                     in hues[2:]]

        stationcolor = hls_to_rgb(hues[0], 0.2, 0.5)
        continentcolor = hls_to_rgb(hues[0], 0.6, 0.2)
        eventcolor = hls_to_rgb(hues[1], 0.2, 0.5)
        cmbcolor = continentcolor
        bgcolor = (1, 1, 1)
        # sizes:
        # everything has to be larger in bright background plot because it
        # is more difficult to read
        sttextsize = (0.02, 0.02, 0.02)
        stmarkersize = 0.02
        evtextsize = (0.02, 0.02, 0.02)
        evmarkersize = 0.06
    else:
        raise ValueError('colorscheme {:s} not recognized'.format(colorscheme))

    # assemble each phase and all stations/events to plot them in a single call
    stations_loc = []
    stations_lab = []
    events_loc = []
    events_lab = []
    phases = [[] for iphase in range(nphases)]
    for gcircle, name, value, stlabel, evlabel in greatcircles:
        iphase = phase_list.index(name)
        phases[iphase].append(gcircle)

        if stlabel not in stations_lab:
            x, y, z = gcircle[0, -1], gcircle[1, -1], gcircle[2, -1]
            stations_loc.append((x, y, z))
            stations_lab.append(stlabel)

        if evlabel not in events_lab:
            x, y, z = gcircle[0, 0], gcircle[1, 0], gcircle[2, 0]
            events_loc.append((x, y, z))
            events_lab.append(evlabel)

    # now begin mayavi plotting
    if offscreen:
        mlab.options.offscreen = True
    fig = mlab.figure(size=figsize, bgcolor=bgcolor)

    # define connectivity of each ray and add it to the scene
    for iphase, phase in enumerate(phases):
        index = 0
        connects = []
        for ray in phase:
            ndim, npoints = ray.shape
            connects.append(np.vstack(
                            [np.arange(index,   index + npoints - 1.5),
                             np.arange(index + 1, index + npoints - .5)]).T)
            index += npoints

        # collapse all points of the phase into a long array
        if len(phase) > 1:
            points = np.hstack(phase)
        elif len(phase) == 1:
            points = phase[0]
        else:
            continue
        connects = np.vstack(connects)

        # Create the points
        src = mlab.pipeline.scalar_scatter(*points)

        # Connect them
        src.mlab_source.dataset.lines = connects

        # The stripper filter cleans up connected lines
        lines = mlab.pipeline.stripper(src)

        color = raycolors[iphase]
        mlab.pipeline.surface(lines, line_width=0.5, color=color)

    # plot all stations
    fig.scene.disable_render = True  # Super duper trick
    stxs, stys, stzs = zip(*stations_loc)
    mlab.points3d(stxs, stys, stzs, scale_factor=stmarkersize,
                  color=stationcolor)

    if station_labels:
        for loc, stlabel in zip(stations_loc, stations_lab):
            mlab.text3d(loc[0], loc[1], loc[2], stlabel, scale=sttextsize,
                        color=stationcolor)

    # plot all events
    evxs, evys, evzs = zip(*events_loc)
    evsource = mlab.pipeline.vector_scatter(
        evxs, evys, evzs, -np.array(evxs), -np.array(evys), -np.array(evzs))
    evmarkers = mlab.pipeline.glyph(
            evsource, scale_factor=evmarkersize, scale_mode='none',
            color=eventcolor, mode='cone', resolution=8)
    evmarkers.glyph.glyph_source.glyph_position = 'head'

    if event_labels:
        for loc, evlabel in zip(events_loc, events_lab):
            mlab.text3d(loc[0], loc[1], loc[2], evlabel, scale=evtextsize,
                        color=eventcolor)
    fig.scene.disable_render = False  # Super duper trick

    # read and plot coastlines
    if coastlines == 'internal':
        from mayavi.sources.builtin_surface import BuiltinSurface
        data_source = BuiltinSurface(source='earth', name="Continents")
        data_source.data_source.on_ratio = 1
    else:
        data_source = mlab.pipeline.open(coastlines)
    coastmesh = mlab.pipeline.surface(data_source, opacity=1.0, line_width=0.5,
                                      color=continentcolor)
    coastmesh.actor.actor.scale = np.array([1.02, 1.02, 1.02])

    # plot block sphere that hides the backside of the continents
    rad = 0.99
    phi, theta = np.mgrid[0:np.pi:51j, 0:2 * np.pi:51j]

    x = rad * np.sin(phi) * np.cos(theta)
    y = rad * np.sin(phi) * np.sin(theta)
    z = rad * np.cos(phi)
    blocksphere = mlab.mesh(x, y, z, color=bgcolor)
    blocksphere.actor.property.frontface_culling = True  # front not rendered

    # make CMB sphere
    r_earth = model.model.radius_of_planet
    r_cmb = r_earth - model.model.cmb_depth
    rad = r_cmb / r_earth
    phi, theta = np.mgrid[0: np.pi: 201j, 0: 2 * np.pi: 201j]

    x = rad * np.sin(phi) * np.cos(theta)
    y = rad * np.sin(phi) * np.sin(theta)
    z = rad * np.cos(phi)
    cmb = mlab.mesh(x, y, z, color=cmbcolor, opacity=0.3, line_width=0.5)
    cmb.actor.property.interpolation = 'gouraud'
    # cmb.actor.property.interpolation = 'flat'

    # make ICB sphere
    r_iocb = r_earth - model.model.iocb_depth
    rad = r_iocb / r_earth
    phi, theta = np.mgrid[0:np.pi:101j, 0:2 * np.pi:101j]

    x = rad * np.sin(phi) * np.cos(theta)
    y = rad * np.sin(phi) * np.sin(theta)
    z = rad * np.cos(phi)
    icb = mlab.mesh(x, y, z, color=cmbcolor, opacity=0.3, line_width=0.5)
    icb.actor.property.interpolation = 'gouraud'
    if view_dict is None:
        view_dict = {'azimuth': 0., 'elevation': 90., 'distance': 4.,
                     'focalpoint': (0., 0., 0.)}
    mlab.view(**view_dict)

    # to make a movie from the image files, you can use the command:
    # avconv -qscale 5 -r 20 -b 9600 -i %05d.png -vf scale=800:752 movie.mp4
    if animate and not offscreen:
        @mlab.show
        @mlab.animate(delay=20)
        def anim():
            iframe = 0
            while 1:
                if savemovie and iframe < 360:
                    mlab.savefig('{:05d}.png'.format(iframe))
                # camera moves from East to West opposite of Earth's rotation
                fig.scene.camera.azimuth(-1.)
                fig.scene.render()
                iframe += 1
                yield
        anim()  # Starts the animation.
    else:
        if offscreen:
            mlab.savefig(fname_out)
        else:
            mlab.show()


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
