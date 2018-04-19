# Author: Tom Eulenfeld
# Year: 2018
"""
SeismicHandler evt file bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from collections import defaultdict
from math import cos, pi
from warnings import warn

from obspy.core.event import (Arrival, Catalog, Event,
                              Magnitude, Origin, OriginQuality,
                              OriginUncertainty, Pick, ResourceIdentifier,
                              StationMagnitude, WaveformStreamID)
from obspy.core.event.header import EvaluationMode, EventType, PickOnset
from obspy.io.sh.core import to_utcdatetime


def _is_evt(filename):
    try:
        with open(filename, 'rb') as f:
            temp = f.read(20)
    except Exception:
        return False
    return b'Event ID' in temp


MAP = {
    'pick': {
        'onset time': 'time',
        'phase flags': 'phase_hint',
        'onset type': 'onset',
        'pick type': 'evaluation_mode',
        'applied filter': 'filter_id',
        'sign': 'polarity'
    },
    'arrival': {
        'phase name': 'phase',
        'distance (deg)': 'distance',
        'theo. azimuth (deg)': 'azimuth',
        'weight': 'time weight'
    },
    'origin': {
        'origin time': 'time',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'depth (km)': 'depth',
        'error in origin time': 'time_errors',
        'error in latitude (km)': 'latitude_errors',
        'error in longitude (km)': 'longitude_errors',
        'error in depth (km)': 'depth_errors',
        'no. of stations used': 'quality',
        'source region': 'region'
    },
    'origin_uncertainty': {
        'error ellipse major': 'max_horizontal_uncertainty',
        'error ellipse minor': 'min_horizontal_uncertainty',
        'error ellipse strike': 'azimuth_max_horizontal_uncertainty'
    },
    'event': {
        'event type': 'event_type'
    }
    # no dict for magnitudes, these are handled by function _mag
}


def _km2m(km):
    return 1000 * float(km)


def _km2deg(km):
    return float(km) / 111.195


def _event_type(et):
    if et == 'local_quake':
        et = 'earthquake'
    return EventType(et)


MAG_MAP = {'ml': 'ML',
           'ms': 'MS',
           'mb': 'Mb',
           'mw': 'Mw'}


CONVERT = {
    # pick
    'onset time': to_utcdatetime,
    'theo. backazimuth (deg)': float,
    'phase flags': str,
    'onset type': PickOnset,
    'pick type': EvaluationMode,
    'applied filter': ResourceIdentifier,
    'sign': str,
    # arrival
    'phase name': str,
    'distance (deg)': float,
    'theo. azimuth (deg)': float,
    'weight': float,
    # origin
    'origin time': to_utcdatetime,
    'latitude': float,
    'longitude': float,
    'depth (km)': _km2m,
    'error in origin time': float,
    'error in latitude (km)': _km2deg,
    'error in longitude (km)': _km2deg,  # correction for lat in _read_evt
    'error in depth (km)': _km2m,
    'no. of stations used': lambda x: OriginQuality(
        used_station_count=int(x)),
    'source region': str,
    # Origin uncertainty'
    'error ellipse major': _km2m,
    'error ellipse minor': _km2m,
    'error ellipse strike': float,
    # event
    'event type': _event_type
}


def _kw(obj, obj_name):
    kws = {}
    for source_key, dest_key in MAP[obj_name].items():
        try:
            val = CONVERT[source_key](obj[source_key])
        except KeyError:
            pass
        except ValueError as ex:
            warn(str(ex))
        else:
            kws[dest_key] = val
    return kws


def _mags(obj, evid, stamag=False):
    mags = []
    pm = None
    for magtype1, magtype2 in MAG_MAP.items():
        magkey = 'mean ' * (not stamag) + 'magnitude ' + magtype1
        if magkey in obj:
            magv = obj[magkey]
            if 'inf' in magv:
                warn('invalid value for magnitude: %s (event id %s)'
                     % (magv, evid))
            else:
                mag = StationMagnitude if stamag else Magnitude
                mags.append(mag(magnitude_type=magtype2, mag=float(magv)))
    if len(mags) == 1:
        pm = mags[0].resource_id
    return mags, pm


def _seed_id_map(inventory=None, id_map=None, id_default='.{}..{}'):
    ret = {}
    if id_map is None:
        id_map = {}
    if inventory is not None:
        for net in inventory:
            for sta in net:
                if len(sta) == 0:
                    temp = id_map.get(sta.code, id_default)
                    temp.split('.', 2)[-1]
                else:
                    cha = sta[0]
                    temp = cha.location_code + '.' + cha.code[:-1] + '{}'
                ret[sta.code] = net.code + '.{}.' + temp
    return ret


def _read_evt(filename, inventory=None, id_map=None, id_default='.{}..{}'):
    """
    Read a SeismicHandler EVT file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    See http://www.seismic-handler.org/wiki/ShmDocFileEvt

    :type filename: str
    :param filename: File or file-like object in text mode.
    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inventory: Inventory used to retrieve network code, location code
        and channel code of stations (SEED id).
    :type id_map: dict
    :param id_map: If channel information was not found in inventory,
        it will be looked up in this dictionary
        (example: `id_map={'MOX': 'GR.{}..HH{}'`).
        The values must contain three dots and two `{}` which are
        substituted by station code and component.
    :type id_default: str
    :param id_default: Default SEED id expression.
        The value must contain three dots and two `{}` which are
        substituted by station code and component.

    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.
    """
    seed_map = _seed_id_map(inventory, id_map, id_default)
    with open(filename, 'r') as f:
        temp = f.read()
    # first create phases and phases_o dictionaries for different phases
    # and phases with origin information
    phases = defaultdict(list)
    phases_o = {}
    phase = {}
    evid = None
    for line in temp.splitlines():
        if 'End of Phase' in line:
            if 'origin time' in phase.keys():
                if evid in phases_o:
                    # found more than one origin
                    pass
                phases_o[evid] = phase
            phases[evid].append(phase)
            phase = {}
            evid = None
        elif line.strip() != '':
            try:
                key, value = line.split(':', 1)
            except ValueError:
                continue
            key = key.strip().lower()
            value = value.strip()
            if key == 'event id':
                evid = value
            elif value != '':
                phase[key] = value
    assert evid is None

    # now create obspy Events from phases and phases_o dictionaries
    events = []
    for evid in phases:
        picks = []
        arrivals = []
        stamags = []
        origins = []
        po = None
        magnitudes = []
        pm = None
        for p in phases[evid]:
            try:
                sta = p['station code']
            except KeyError:
                sta = ''
            try:
                comp = p['component']
            except KeyError:
                comp = ''
            try:
                wid = seed_map[sta]
            except KeyError:
                wid = id_default
            wid = WaveformStreamID(seed_string=wid.format(sta, comp))
            pick = Pick(waveform_id=wid, **_kw(p, 'pick'))
            arrival = Arrival(pick_id=pick.resource_id, **_kw(p, 'arrival'))
            picks.append(pick)
            arrivals.append(arrival)
            stamags_temp, _ = _mags(p, evid, stamag=True)
            stamags.extend(stamags_temp)
        if evid in phases_o:
            o = phases_o[evid]
            uncertainty = OriginUncertainty(**_kw(o, 'origin_uncertainty'))
            origin = Origin(arrivals=arrivals, origin_uncertainty=uncertainty,
                            **_kw(o, 'origin'))
            if origin.latitude is None or origin.longitude is None:
                warn('latitude or longitude not set for event %s' % evid)
            else:
                if origin.longitude_errors.uncertainty is not None:
                    origin.longitude_errors.uncertainty *= cos(
                        origin.latitude / 180 * pi)
                origins = [origin]
                po = origin.resource_id
            magnitudes, pm = _mags(o, evid)
        else:
            o = p
        event = Event(resource_id=ResourceIdentifier(evid),
                      picks=picks,
                      origins=origins,
                      magnitudes=magnitudes,
                      station_magnitudes=stamags,
                      preferred_origin_id=po,
                      preferred_magnitude_id=pm,
                      **_kw(o, 'event')
                      )
        events.append(event)
    return Catalog(events,
                   description='Created from SeismicHandler EVT format')
