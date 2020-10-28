# -*- coding: utf-8 -*-
"""
Calculations for 3D ray paths.

:author:
    Matthias Meschede
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import warnings
import numpy as np

import obspy.geodetics.base as geodetics


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
        ``[(gcircle, phase_name, station_label, event_timestamp,
        event_magnitude, event_id, origin_id), ...]``. ``gcircle`` is an array
        of shape ``[3, npoints]`` with the path coordinates. ``phase_name`` is
        the name of the seismic phase, ``station_label`` is the name of the
        station and network that belongs to the path. ``event_timestamp``,
        ``event_magnitude``, ``event_id`` and ``origin_id`` describe the event
        that belongs to the path.
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
            label_ = ".".join((network.code, station.code))
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
    event_ids = []
    origin_ids = []
    magnitudes = []
    times = []
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
            # XXX do we really want to include events without depth???
            origin.depth = 0.
        evdepths.append(origin.get('depth') * 1e-3)
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        mag = magnitude.mag
        event_ids.append(str(event.resource_id))
        origin_ids.append(str(origin.resource_id))
        magnitudes.append(mag)
        times.append(origin.time.timestamp)

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
        for evlat, evlon, evdepth_km, time, magnitude, event_id, origin_id \
                in zip(evlats, evlons, evdepths, times, magnitudes, event_ids,
                       origin_ids):
            arrivals = model.get_ray_paths_geo(
                evdepth_km, evlat, evlon, stlat, stlon, phase_list=phase_list,
                resample=True)
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

                greatcircles.append((gcircle, arr.name, stlabel, time,
                                     magnitude, event_id, origin_id))

    return greatcircles
