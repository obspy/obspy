#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NonLinLoc file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from math import sqrt
import warnings
from obspy import UTCDateTime, Catalog, __version__
from obspy.core.event import Event, Origin, OriginQuality, OriginUncertainty, \
    Pick, Arrival, WaveformStreamID, CreationInfo, Comment
from obspy.core.util.geodetics import kilometer2degrees


ONSETS = {"I": "impulsive", "E": "emergent"}
ONSETS_REVERSE = {"impulsive": "i", "emergent": "e"}
POLARITIES = {"C": "positive", "U": "positive", "D": "negative"}
POLARITIES_REVERSE = {"positive": "u", "negative": "d"}


def is_nlloc_hyp(filename):
    """
    Checks that a file is actually a NonLinLoc Hypocenter-Phase file.
    """
    try:
        with open(filename, 'rb') as fh:
            temp = fh.read(6)
    except:
        return False
    if temp != b'NLLOC ':
        return False
    return True


def read_nlloc_hyp(filename, coordinate_converter=None, **kwargs):
    """
    Reads a NonLinLoc Hypocenter-Phase file to a
    :class:`~obspy.core.event.Catalog` object.

    .. note::

        Coordinate conversion from coordinate frame of NonLinLoc model files /
        location run to WGS84 has to be specified explicitly by the user if
        necessary.

    :param filename: File or file-like object in text mode.
    :type coordinate_converter: func
    :param coordinate_converter: Function to convert (x, y, z)
        coordinates of NonLinLoc output to geographical coordinates and depth
        in meters (longitude, latitude, depth in kilometers).
        If left `None` NonLinLoc (x, y, z) output is left unchanged (e.g. if
        it is in geographical coordinates already like for NonLinLoc in
        global mode).
        The function should accept three arguments x, y, z and return a
        tuple of three values (lon, lat, depth in kilometers).
    """
    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rt") as fh:
                data = fh.read()
        except:
            try:
                data = filename.decode()
            except:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode()

    lines = data.splitlines()

    # determine indices of block start/end of the NLLOC output file
    indices_hyp = [None, None]
    indices_phases = [None, None]
    for i, line in enumerate(lines):
        if line.startswith("NLLOC "):
            indices_hyp[0] = i
        elif line.startswith("END_NLLOC"):
            indices_hyp[1] = i
        elif line.startswith("PHASE "):
            indices_phases[0] = i
        elif line.startswith("END_PHASE"):
            indices_phases[1] = i
    if any([i is None for i in indices_hyp]):
        msg = ("NLLOC HYP file seems corrupt,"
               " could not detect 'NLLOC' and 'END_NLLOC' lines.")
        raise Exception(msg)
    # strip any other lines around NLLOC block
    lines = lines[indices_hyp[0]:indices_hyp[1]]

    # extract PHASES lines (if any)
    if any(indices_phases):
        if not all(indices_phases):
            msg = ("NLLOC HYP file seems corrupt, 'PHASE' block is corrupt.")
            raise Exception(msg)
        i1, i2 = indices_phases
        lines, phases_lines = lines[:i1] + lines[i2 + 1:], lines[i1 + 1:i2]
    else:
        phases_lines = []

    lines = dict([line.split(None, 1) for line in lines])
    line = lines["SIGNATURE"]

    line = line.rstrip().split('"')[1]
    signature, version, date, time = line.rsplit(" ", 3)
    creation_time = UTCDateTime().strptime(date + time, str("%d%b%Y%Hh%Mm%S"))

    # maximum likelihood origin location info line
    line = lines["HYPOCENTER"]

    x, y, z = map(float, line.split()[1:7:2])

    if coordinate_converter:
        x, y, z = coordinate_converter(x, y, z)

    # origin time info line
    line = lines["GEOGRAPHIC"]

    year, month, day, hour, minute = map(int, line.split()[1:6])
    seconds = float(line.split()[6])
    time = UTCDateTime(year, month, day, hour, minute, seconds)

    # distribution statistics line
    line = lines["STATISTICS"]
    covariance_ZZ = float(line.split()[17])
    stats_info_string = (
        "Note: Depth error is calculated from covariance matrix as 1D "
        "marginal while OriginUncertainty min/max horizontal errors "
        "are calculated from 2D error ellipsoid and are therefore seemingly "
        "higher compared to depth errors. Error estimates can be "
        "reconstructed from the following original NonLinLoc error "
        "statistics line:\nSTATISTICS " +
        lines["STATISTICS"])

    # goto location quality info line
    line = lines["QML_OriginQuality"].split()

    (assoc_phase_count, used_phase_count, assoc_station_count,
     used_station_count, depth_phase_count) = map(int, line[1:11:2])
    stderr, az_gap, sec_az_gap = map(float, line[11:17:2])
    gt_level = line[17]
    min_dist, max_dist, med_dist = map(float, line[19:25:2])

    # goto location quality info line
    line = lines["QML_OriginUncertainty"]

    hor_unc, min_hor_unc, max_hor_unc, hor_unc_azim = \
        map(float, line.split()[1:9:2])

    # assign origin info
    event = Event()
    cat = Catalog(events=[event])
    o = Origin()
    event.origins = [o]
    o.origin_uncertainty = OriginUncertainty()
    o.quality = OriginQuality()
    ou = o.origin_uncertainty
    oq = o.quality
    o.comments.append(Comment(text=stats_info_string))

    cat.creation_info.creation_time = UTCDateTime()
    cat.creation_info.version = "ObsPy %s" % __version__
    event.creation_info = CreationInfo(creation_time=creation_time,
                                       version=version)
    event.creation_info.version = version
    o.creation_info = CreationInfo(creation_time=creation_time,
                                   version=version)

    o.longitude = x
    o.latitude = y
    o.depth = z * 1e3  # meters!
    o.depth_errors.uncertainty = sqrt(covariance_ZZ) * 1e3  # meters!
    o.depth_errors.confidence_level = 68
    o.depth_type = "from location"
    o.time = time

    ou.horizontal_uncertainty = hor_unc
    ou.min_horizontal_uncertainty = min_hor_unc
    ou.max_horizontal_uncertainty = max_hor_unc
    # values of -1 seem to be used for unset values, set to None
    for field in ("horizontal_uncertainty", "min_horizontal_uncertainty",
                  "max_horizontal_uncertainty"):
        if ou.get(field, -1) == -1:
            ou[field] = None
        else:
            ou[field] *= 1e3  # meters!
    ou.azimuth_max_horizontal_uncertainty = hor_unc_azim
    ou.preferred_description = "uncertainty ellipse"
    ou.confidence_level = 68  # NonLinLoc in general uses 1-sigma (68%) level

    oq.standard_error = stderr
    oq.azimuthal_gap = az_gap
    oq.secondary_azimuthal_gap = sec_az_gap
    oq.used_phase_count = used_phase_count
    oq.used_station_count = used_station_count
    oq.associated_phase_count = assoc_phase_count
    oq.associated_station_count = assoc_station_count
    oq.depth_phase_count = depth_phase_count
    oq.ground_truth_level = gt_level
    oq.minimum_distance = kilometer2degrees(min_dist)
    oq.maximum_distance = kilometer2degrees(max_dist)
    oq.median_distance = kilometer2degrees(med_dist)

    # go through all phase info lines
    for line in phases_lines:
        line = line.split()
        pick = Pick()
        event.picks.append(pick)
        arrival = Arrival(pick_id=pick.resource_id)
        o.arrivals.append(arrival)
        station = str(line[0])
        wid = WaveformStreamID(station_code=station)
        pick.waveform_id = wid
        phase = str(line[4])
        date, hourmin, sec = map(str, line[6:9])
        t = UTCDateTime().strptime(date + hourmin, "%Y%m%d%H%M") + float(sec)
        pick.time = t
        pick.time_errors.uncertainty = float(line[10])
        pick.phase_hint = phase
        arrival.phase = phase
        arrival.distance = kilometer2degrees(float(line[21]))
        arrival.azimuth = float(line[23])
        arrival.takeoff_angle = float(line[24])
        pick.onset = ONSETS.get(line[3].upper(), None)
        pick.polarity = POLARITIES.get(line[5].upper(), None)
        arrival.time_residual = float(line[16])
        arrival.time_weight = float(line[17])

    return cat


def write_nlloc_obs(catalog, filename, **kwargs):
    """
    Reads a NonLinLoc Phase file (NLLOC_OBS) to a
    :class:`~obspy.core.event.Catalog` object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.stream.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    """
    info = []

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "write"):
        file_opened = True
        fh = open(filename, "wb")
    else:
        file_opened = False
        fh = filename

    if len(catalog) > 1:
        msg = ("Writing NonLinLoc Phase file is only supported for Catalogs "
               "with a single Event in it (use a for loop over the catalog "
               "and provide an output file name for each event).")
        raise ValueError(msg)

    fmt = '%s %s %s %s %s %s %s %s %7.4f GAU %9.2e %9.2e %9.2e %9.2e'
    for pick in catalog[0].picks:
        wid = pick.waveform_id
        station = wid.station_code or "?"
        component = wid.channel_code and wid.channel_code[-1].upper() or "?"
        if component not in "ZNEH":
            component = "?"
        onset = ONSETS_REVERSE.get(pick.onset, "?")
        phase_type = pick.phase_hint or "?"
        polarity = POLARITIES_REVERSE.get(pick.polarity, "?")
        date = pick.time.strftime("%Y%m%d")
        hourminute = pick.time.strftime("%H%M")
        seconds = pick.time.second + pick.time.microsecond * 1e-6
        time_error = pick.time_errors.uncertainty or -1
        if time_error == -1:
            try:
                time_error = (pick.time_errors.upper_uncertainty +
                              pick.time_errors.lower_uncertainty) / 2.0
            except:
                pass
        info_ = fmt % (station.ljust(6), "?".ljust(4), component.ljust(4),
                       onset.ljust(1), phase_type.ljust(6), polarity.ljust(1),
                       date, hourminute, seconds, time_error, -1, -1, -1)
        info.append(info_)

    if info:
        info = "\n".join(sorted(info) + [""])
    else:
        msg = "No pick information, writing empty NLLOC OBS file."
        warnings.warn(msg)
    fh.write(info.encode())

    # Close if a file has been opened by this function.
    if file_opened is True:
        fh.close()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
