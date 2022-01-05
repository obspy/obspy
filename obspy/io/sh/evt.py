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
from collections import defaultdict
import io
from math import cos, pi
from warnings import warn

from obspy.core.event import (Arrival, Catalog, Event,
                              Magnitude, Origin, OriginQuality,
                              OriginUncertainty, Pick, ResourceIdentifier,
                              StationMagnitude, WaveformStreamID)
from obspy.core.event.header import EvaluationMode, EventType, PickOnset
from obspy.core.inventory.util import _resolve_seedid, _add_resolve_seedid_doc
from obspy.io.sh.core import to_utcdatetime


def _is_evt(filename):
    try:
        with open(filename, 'rb') as f:
            temp = f.read(20)
    except Exception:
        return False
    return b'Event ID' in temp


def _km2m(km):
    return 1000 * float(km)


def _km2deg(km):
    return float(km) / 111.195


def _event_type(et):
    if 'quake' in et:
        et = 'earthquake'
    return EventType(et)


MAG_MAP = {'ml': 'ML',
           'ms': 'MS',
           'mb': 'Mb',
           'mw': 'Mw'}


MAP = {
    'pick': {
        'onset time': ('time', to_utcdatetime),
        'phase flags': ('phase_hint', str),
        'onset type': ('onset', PickOnset),
        'pick type': ('evaluation_mode', EvaluationMode),
        'applied filter': ('filter_id', ResourceIdentifier),
        'sign': ('polarity', str)
    },
    'arrival': {
        'phase name': ('phase', str),
        'distance (deg)': ('distance', float),
        'theo. azimuth (deg)': ('azimuth', float),
        'weight': ('time_weight', float),
    },
    'origin': {
        'origin time': ('time', to_utcdatetime),
        'latitude': ('latitude', float),
        'longitude': ('longitude', float),
        'depth (km)': ('depth', _km2m),
        'error in origin time': ('time_errors', float),
        'error in latitude (km)': ('latitude_errors', _km2deg),
        'error in longitude (km)': ('longitude_errors', _km2deg),
        # (correction for lat in _read_evt)
        'error in depth (km)': ('depth_errors', _km2m),
        'no. of stations used': (
            'quality', lambda x: OriginQuality(used_station_count=int(x))),
        'source region': ('region', str)
    },
    'origin_uncertainty': {
        'error ellipse major': ('max_horizontal_uncertainty', _km2m),
        'error ellipse minor': ('min_horizontal_uncertainty', _km2m),
        'error ellipse strike': ('azimuth_max_horizontal_uncertainty', float)
    },
    'event': {
        'event type': ('event_type', _event_type)
    }
    # no dict for magnitudes, these are handled by function _mag
}


# define supported keys just for documentation
SUPPORTED_KEYS = ['event id', 'station code', 'component', 'magnitude (M?)',
                  'mean magnitude (M?)']
SUPPORTED_KEYS = sorted([key for obj in MAP.values() for key in obj] +
                        SUPPORTED_KEYS)


def _kw(obj, obj_name):
    kws = {}
    for source_key, (dest_key, convert) in MAP[obj_name].items():
        try:
            val = convert(obj[source_key])
        except KeyError:
            pass
        except ValueError as ex:
            warn(str(ex))
        else:
            kws[dest_key] = val
    return kws


def _mags(obj, evid, stamag=False, wid=None):
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
                magv = float(magv)
                mag = (StationMagnitude(station_magnitude_type=magtype2,
                                        mag=magv, waveform_id=wid)
                       if stamag else
                       Magnitude(magnitude_type=magtype2, mag=magv))
                mags.append(mag)
    if len(mags) == 1:
        pm = mags[0].resource_id
    return mags, pm


@_add_resolve_seedid_doc
def _read_evt(filename, encoding='utf-8', **kwargs):
    """
    Read a SeismicHandler EVT file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    :param str encoding: encoding used (default: utf-8)

    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.

    .. note::
        The following fields are supported by this function: %s.

        Compare with http://www.seismic-handler.org/wiki/ShmDocFileEvt
    """
    with io.open(filename, 'r', encoding=encoding) as f:
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
            sta = p.get('station code', '')
            comp = p.get('component', '')
            pick_kwargs = _kw(p, 'pick')
            widargs = _resolve_seedid(
                    sta, comp, time=pick_kwargs['time'], **kwargs)
            wid = WaveformStreamID(*widargs)
            pick = Pick(waveform_id=wid, **pick_kwargs)
            arrival = Arrival(pick_id=pick.resource_id, **_kw(p, 'arrival'))
            picks.append(pick)
            arrivals.append(arrival)
            stamags_temp, _ = _mags(p, evid, stamag=True, wid=wid)
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


if _read_evt.__doc__ is not None and '%s' in _read_evt.__doc__:
    _read_evt.__doc__ = _read_evt.__doc__ % (SUPPORTED_KEYS,)
