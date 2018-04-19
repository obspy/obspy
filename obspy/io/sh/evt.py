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
from math import cos, pi
from warnings import warn

from obspy.core.event import (Arrival, Catalog, Event,
                              Magnitude, Origin, OriginQuality,
                              Pick, ResourceIdentifier, StationMagnitude)
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
        'Onset time': 'time',
        'Phase Flags': 'phase_hint',
        'Onset Type': 'onset',
        'Pick Type': 'evaluation_mode',
        'Applied filter': 'filter_id'
        # not used:
        # Quality number         : 2
    },
    'arrival': {
        'Phase name': 'phase',
        'Distance (deg)': 'distance',
        'Theo. Azimuth (deg)': 'azimuth',
        'Weight': 'time weight',
        # not used:
        # Distance (km)
        # Theo. Backazimuth (deg)
    },
    'origin': {
        'Origin time': 'time',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Depth (km)': 'depth',
        'Error in Origin Time': 'time_errors',
        'Error in Latitude (km)': 'latitude_errors',
        'Error in Longitude (km)': 'longitude_errors',
        'Error in Depth (km)': 'depth_errors',
        'No. of Stations used': 'quality',
        'Source region': 'region'
        # not used:
        # Depth type             : ( ) free
        # Region Table           : GEO_REG
        # Region ID              : 5538
        # Reference Location Name: CENTRE
        # Error Ellipse Major    :   0.02
        # Error Ellipse Minor    :   0.01
        # Error Ellipse Strike   :  68.60
    },
    'event': {
        'Event type': 'event_type'
    }
    # no dict for magnitudes, these are handled by function _mag
}


def KM2M(km):
    return 1000 * float(km)


def KM2DEG(km):
    return float(km) / 111.195


def _event_type(et):
    if et == 'local_quake':
        et = 'earthquake'
    return EventType(et)


CONVERT = {
    # pick
    'Onset time': to_utcdatetime,
    'Theo. Backazimuth (deg)': float,
    'Phase Flags': str,
    'Onset Type': PickOnset,
    'Pick Type': EvaluationMode,
    'Applied filter ': ResourceIdentifier,
    # arrival
    'Phase name': str,
    'Distance (deg)': float,
    'Theo. Azimuth (deg)': float,
    'Weight': float,
    # station magnitudes
    'Magnitude ml': float,
    # origin
    'Origin time': to_utcdatetime,
    'Latitude': float,
    'Longitude': float,
    'Depth (km)': KM2M,
    'Error in Origin Time': float,
    'Error in Latitude (km)': KM2DEG,
    'Error in Longitude (km)': KM2DEG,  # still needs correction for lat
    'Error in Depth (km)': KM2M,
    'No. of Stations used': lambda x: OriginQuality(
        used_station_count=int(x)),
    'Source region': str,
    # event
    'Event type': _event_type
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


def _mag(obj, evid, stamag=False):
    magkey = 'Mean ' * (not stamag) + 'Magnitude ml'
    if magkey in obj:
        magv = obj[magkey]
        if 'inf' in magv:
            warn('invalid value for magnitude: %s (event id %s)'
                 % (magv, evid))
        else:
            Mag = StationMagnitude if stamag else Magnitude
            return Mag(magnitude_type='ML', mag=float(magv))


def _read_evt(filename, waveform_id='.{}..{}'):
    # first create phases and phases_o dictionaries for different phases
    # and phases with origin information
    with open(filename, 'r') as f:
        temp = f.read()
    phases = defaultdict(list)
    phases_o = {}
    phase = {}
    evid = None
    for line in temp.splitlines():
        if 'End of Phase' in line:
            if 'Origin time' in phase.keys():
                if evid in phases_o:
                    warn(('Found two or more origins for event %s '
                          '-> take first') % evid)
                else:
                    phases_o[evid] = phase
            phases[evid].append(phase)
            phase = {}
            evid = None
        elif line.strip() != '':
            try:
                key, value = line.split(':', 1)
            except ValueError:
                continue
            key = key.strip()
            value = value.strip()
            if key == 'Event ID':
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
                sta = p['Station code']
            except KeyError:
                sta = ''
            try:
                comp = p['Component']
            except KeyError:
                comp = ''
            try:
                wid = waveform_id[sta]
            except TypeError:
                wid = waveform_id
            pick = Pick(waceform_id=wid.format(sta, comp), **_kw(p, 'pick'))
            arrival = Arrival(pick_id=pick.resource_id, **_kw(p, 'arrival'))
            picks.append(pick)
            arrivals.append(arrival)
            stamag = _mag(p, evid, stamag=True)
            if stamag is not None:
                stamags.append(stamag)
        if evid in phases_o:
            o = phases_o[evid]
            origin = Origin(arrivals=arrivals, **_kw(o, 'origin'))
            if origin.latitude is None or origin.longitude is None:
                warn('latitude or longitude not set for event %s' % evid)
            else:
                if origin.longitude_errors.uncertainty is not None:
                    origin.longitude_errors.uncertainty *= cos(
                        origin.latitude / 180 * pi)
                origins = [origin]
                po = origin.resource_id
            mag = _mag(o, evid)
            if mag is not None:
                magnitudes = [mag]
                pm = mag.resource_id
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
