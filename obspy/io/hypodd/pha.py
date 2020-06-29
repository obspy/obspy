# -*- coding: utf-8 -*-
"""
HypoDD PHA read support.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
import io

from obspy import UTCDateTime
from obspy.core.event import (
    Catalog, Event, Origin, Magnitude, Pick, WaveformStreamID, Arrival,
    OriginQuality)
from obspy.core.util.misc import _seed_id_map


def _block2event(block, seed_map, id_default, ph2comp):
    """
    Read HypoDD event block
    """
    lines = block.strip().splitlines()
    yr, mo, dy, hr, mn, sc, la, lo, dp, mg, eh, ez, rms, id_ = lines[0].split()
    time = UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mn), float(sc),
                       strict=False)
    picks = []
    arrivals = []
    for line in lines[1:]:
        sta, reltime, weight, phase = line.split()
        comp = ph2comp.get(phase, '')
        wid = seed_map.get(sta, id_default)
        _waveform_id = WaveformStreamID(seed_string=wid.format(sta, comp))
        pick = Pick(waveform_id=_waveform_id, phase_hint=phase,
                    time=time + float(reltime))
        arrival = Arrival(phase=phase, pick_id=pick.resource_id,
                          time_weight=float(weight))
        picks.append(pick)
        arrivals.append(arrival)
    qu = None if rms == '0.0' else OriginQuality(standard_error=float(rms))
    origin = Origin(arrivals=arrivals,
                    resource_id="smi:local/origin/" + id_,
                    quality=qu,
                    latitude=float(la),
                    longitude=float(lo),
                    depth=1000 * float(dp),
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


def _read_pha(filename, inventory=None, id_map=None, id_default='.{}..{}',
              ph2comp={'P': 'Z', 'S': 'N'}, encoding='utf-8'):
    """
    Read a HypoDD PHA file and returns an ObsPy Catalog object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.read_events` function, call this
        instead.

    The optional parameters all deal with the problem, that the PHA format
    only stores station names for the picks, but the Pick object expects
    a SEED id. A SEED id template is retrieved for each station by the
    following procedure:

    1. look at id_map for a direct station name match and use the specified
       template
    2. if 1 did not succeed, look if the station is present in inventory and
       use its first channel as template
    3. if 1 and 2 did not succeed, use specified default template (id_default)

    :param str filename: File or file-like object in text mode.
    :type inventory: :class:`~obspy.core.inventory.inventory.Inventory`
    :param inventory: Inventory used to retrieve network code, location code
        and channel code of stations (SEED id).
    :param dict id_map: Default templates for each station
        (example: `id_map={'MOX': 'GR.{}..HH{}'`).
        The values must contain three dots and two `{}` which are
        substituted by station code and component.
    :param str id_default: Default SEED id template.
        The value must contain three dots and two `{}` which are
        substituted by station code and component.
    :param dict ph2comp: mapping of phases to components
        (default: {'P': 'Z', 'S': 'N'})
    :param str encoding: encoding used (default: utf-8)

    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.
    """
    seed_map = _seed_id_map(inventory, id_map)
    with io.open(filename, 'r', encoding=encoding) as f:
        text = f.read()
    events = [_block2event(block, seed_map, id_default, ph2comp)
              for block in text.split('#')[1:]]
    return Catalog(events)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
