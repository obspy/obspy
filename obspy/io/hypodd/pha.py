# -*- coding: utf-8 -*-
"""
HypoDD PHA read support.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
from math import cos
from numpy import deg2rad
from warnings import warn

from obspy import UTCDateTime
from obspy.core.event import (
    Catalog, Event, Origin, Magnitude, Pick, WaveformStreamID, Arrival,
    OriginQuality)
from obspy.core.inventory.util import (
    _add_resolve_seedid_doc, _add_resolve_seedid_ph2comp_doc, _resolve_seedid)


DEG2KM = 111.2


def _block2event(block, eventid_map, **kwargs):
    """
    Read HypoDD event block
    """
    lines = block.strip().splitlines()
    yr, mo, dy, hr, mn, sc, la, lo, dp, mg, eh, ez, rms, id_ = lines[0].split()
    if eventid_map is not None and id_ in eventid_map:
        id_ = eventid_map[id_]
    time = UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mn), float(sc),
                       strict=False)
    laterr = None if float(eh) == 0 else float(eh) / DEG2KM
    lonerr = (None if laterr is None or float(la) > 89 else
              laterr / cos(deg2rad(float(la))))
    ez = None if float(ez) == 0 else float(ez) * 1000
    rms = None if float(rms) == 0 else float(rms)
    picks = []
    arrivals = []
    for line in lines[1:]:
        sta, reltime, weight, phase = line.split()
        widargs = _resolve_seedid(sta, '', time=time, phase=phase, **kwargs)
        wid = WaveformStreamID(*widargs)
        pick = Pick(waveform_id=wid, phase_hint=phase,
                    time=time + float(reltime))
        arrival = Arrival(phase=phase, pick_id=pick.resource_id,
                          time_weight=float(weight))
        picks.append(pick)
        arrivals.append(arrival)
    qu = OriginQuality(associated_phase_count=len(picks), standard_error=rms)
    origin = Origin(arrivals=arrivals,
                    resource_id="smi:local/origin/" + id_,
                    quality=qu,
                    latitude=float(la),
                    longitude=float(lo),
                    depth=1000 * float(dp),
                    latitude_errors=laterr,
                    longitude_errors=lonerr,
                    depth_errors=ez,
                    time=time)
    if mg.lower() == 'nan':
        magnitudes = []
        preferred_magnitude_id = None
    else:
        magnitude = Magnitude(mag=mg, resource_id="smi:local/magnitude/" + id_)
        magnitudes = [magnitude]
        preferred_magnitude_id = magnitude.resource_id
    event = Event(resource_id="smi:local/event/" + id_,
                  picks=picks,
                  origins=[origin],
                  magnitudes=magnitudes,
                  preferred_origin_id=origin.resource_id,
                  preferred_magnitude_id=preferred_magnitude_id)
    return event


def _is_pha(filename):
    try:
        with open(filename, 'rb') as f:
            line = f.readline()
        assert line.startswith(b'#')
        assert len(line.split()) == 15
        yr, mo, dy, hr, mn, sc = line.split()[1:7]
        UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mn), float(sc),
                    strict=False)
    except Exception:
        return False
    else:
        return True


@_add_resolve_seedid_ph2comp_doc
@_add_resolve_seedid_doc
def _read_pha(filename, eventid_map=None, encoding='utf-8', **kwargs):
    """
    Read a HypoDD PHA file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    :param str filename: File or file-like object in text mode.
    :param dict ph2comp: mapping of phases to components
        (default: {'P': 'Z', 'S': 'N'})
    :param dict eventid_map: Desired mapping of hypodd event ids (dict values)
        to event resource ids (dict keys).
        The returned dictionary of the HYPODDPHA writing operation can be used.
        By default, ids are not mapped.
    :param str encoding: encoding used (default: utf-8)

    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.
    """
    if eventid_map is not None:
        eventid_map = {v: k for k, v in eventid_map.items()}
    with io.open(filename, 'r', encoding=encoding) as f:
        text = f.read()
    events = [_block2event(block, eventid_map, **kwargs)
              for block in text.split('#')[1:]]
    return Catalog(events)


PHA1 = ('# {o.time.year} {o.time.month:>2} {o.time.day:>2} {o.time.hour:>2}'
        ' {o.time.minute:>2} {o.time.second:>2}.{o.time.microsecond:06d}  '
        '{o.latitude} {o.longitude} {depth}  {mag}  {he} {ve} {rms}'
        '   {evid:>9}\n')
PHA2 = '{p.waveform_id.station_code:6}  {relt:.4f}  {weight}  {p.phase_hint}\n'


def _map_eventid(evid, eventid_map, used_ids, counter):
    idpha = evid
    if evid in eventid_map:
        idpha = eventid_map[evid]
        if not idpha.isdigit() or len(idpha) > 9:
            msg = ('Invalid value in eventid_map, pha event id has to be '
                   'digit with max 9 digits')
            raise ValueError(msg)
        return idpha
    if not idpha.isdigit():
        idpha = ''.join(char for char in idpha if char.isdigit())
    if len(idpha) > 9:
        idpha = idpha[:9]
    while idpha == '' or idpha in used_ids:
        idpha = str(counter[0])
        counter[0] += 1
    if idpha != evid:
        eventid_map[evid] = idpha
    used_ids.add(idpha)
    return idpha


def _write_pha(catalog, filename, eventid_map=None,
               **kwargs):  # @UnusedVariable
    """
    Write a HypoDD PHA file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file-like object
    :param filename: Filename to write or open file-like object.
    :param dict eventid_map: Desired mapping of event resource ids (dict keys)
        to hypodd event ids (dict values).
        HYPODD expects integer event ids with maximal 9 digits. If the event
        resource id is not present in the mapping,
        the event resource id is stripped of all non-digit characters and
        truncated to a length of 9 chars. If this method does not generate a
        valid hypodd event id, a counter starting at 1000 is used.

    :returns: Dictionary eventid_map with mapping of event resource id to
        hypodd event id. Items are only present if both ids are different.
    """
    if len(catalog) >= 10**10:
        warn('Writing a very large catalog will use event ids that might not '
             'be readable by HypoDD.')
    lines = []
    if eventid_map is None:
        eventid_map = {}
    args_map_eventid = (eventid_map, set(eventid_map.values()), [1])
    for event in catalog:
        try:
            ori = event.preferred_origin() or event.origins[0]
        except IndexError:
            warn(f'Skipping writing event with missing origin: {event}')
            continue
        try:
            mag = event.preferred_magnitude() or event.magnitudes[0]
        except IndexError:
            warn('Missing magnitude will be set to 0.0')
            mag = 0.
        else:
            mag = mag.mag
        evid = event.resource_id.id
        evid = evid.split('/')[-1] if '/' in evid else evid
        evid = _map_eventid(evid, *args_map_eventid)
        rms = (ori.quality.standard_error if 'quality' in ori and ori.quality
               else None)
        rms = rms if rms is not None else 0.0
        he1 = ori.latitude_errors.uncertainty if ori.latitude_errors else None
        he2 = (ori.longitude_errors.uncertainty if ori.longitude_errors
               else None)
        shortening = cos(deg2rad(ori.latitude))
        he = max(0. if he1 is None else he1 * DEG2KM,
                 0. if he2 is None else he2 * DEG2KM * shortening)
        ve = ori.depth_errors.uncertainty if ori.depth_errors else None
        ve = 0. if ve is None else ve / 1000
        line = PHA1.format(o=ori, depth=ori.depth / 1000, mag=mag,
                           he=he, ve=ve, rms=rms,
                           evid=evid)
        lines.append(line)
        weights = {str(arrival.pick_id): arrival.time_weight
                   for arrival in ori.arrivals if arrival.time_weight}
        for pick in event.picks:
            weight = weights.get(str(pick.resource_id), 1.)
            line = PHA2.format(p=pick, relt=pick.time - ori.time,
                               weight=weight)
            lines.append(line)
    data = ''.join(lines)
    try:
        with open(filename, 'w') as fh:
            fh.write(data)
    except TypeError:
        filename.write(data)
    return None if len(eventid_map) == 0 else eventid_map


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
