# -*- coding: utf-8 -*-
"""
NonLinLoc file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings
from math import sqrt

import numpy as np

from obspy import Catalog, UTCDateTime, __version__
from obspy.core.event import (Arrival, Comment, CreationInfo, Event, Origin,
                              OriginQuality, OriginUncertainty, Pick,
                              WaveformStreamID)
from obspy.core.inventory.util import (
    _add_resolve_seedid_doc, _add_resolve_seedid_ph2comp_doc, _resolve_seedid)
from obspy.geodetics import kilometer2degrees


ONSETS = {"i": "impulsive", "e": "emergent"}
ONSETS_REVERSE = {"impulsive": "i", "emergent": "e"}
POLARITIES = {"c": "positive", "u": "positive", "d": "negative"}
POLARITIES_REVERSE = {"positive": "u", "negative": "d"}


def is_nlloc_hyp(filename):
    """
    Checks that a file is actually a NonLinLoc Hypocenter-Phase file.
    """
    try:
        with open(filename, 'rb') as fh:
            temp = fh.read(6)
    except Exception:
        return False
    if temp != b'NLLOC ':
        return False
    return True


@_add_resolve_seedid_ph2comp_doc
@_add_resolve_seedid_doc
def read_nlloc_hyp(filename, coordinate_converter=None, picks=None, **kwargs):
    """
    Reads a NonLinLoc Hypocenter-Phase file to a
    :class:`~obspy.core.event.catalog.Catalog` object.

    .. note::

        Coordinate conversion from coordinate frame of NonLinLoc model files /
        location run to WGS84 has to be specified explicitly by the user if
        necessary.

    .. note::

        An example can be found on the :mod:`~obspy.io.nlloc` submodule front
        page in the documentation pages.

    :param filename: File or file-like object in text mode.
    :type coordinate_converter: callable
    :param coordinate_converter: Function to convert (x, y, z)
        coordinates of NonLinLoc output to geographical coordinates and depth
        in meters (longitude, latitude, depth in kilometers).
        If left ``None``, the geographical coordinates in the "GEOGRAPHIC" line
        of NonLinLoc output are used.
        The function should accept three arguments x, y, z (each of type
        :class:`numpy.ndarray`) and return a tuple of three
        :class:`numpy.ndarray` (lon, lat, depth in kilometers).
    :type picks: list of :class:`~obspy.core.event.origin.Pick`
    :param picks: Original picks used to generate the NonLinLoc location.
        If provided, the output event will include the original picks and the
        arrivals in the output origin will link to them correctly (with their
        ``pick_id`` attribute). If not provided, the output event will include
        (the rather basic) pick information that can be reconstructed from the
        NonLinLoc hypocenter-phase file.
    :rtype: :class:`~obspy.core.event.catalog.Catalog`
    """
    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rb") as fh:
                data = fh.read()
            data = data.decode("UTF-8")
        except Exception:
            try:
                data = filename.decode("UTF-8")
            except Exception:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode("UTF-8")

    # split lines and remove empty ones
    lines = [line for line in data.splitlines() if line.strip()]

    # remember picks originally used in location, if provided
    original_picks = picks
    if original_picks is None:
        original_picks = []

    cat = Catalog()
    lines_start = [i for i, line in enumerate(lines)
                   if line.startswith("NLLOC ")]
    lines_end = [i for i, line in enumerate(lines)
                 if line.startswith("END_NLLOC")]
    if len(lines_start) != len(lines_end):
        msg = ("NLLOC HYP file '{}' seems corrupt, number of 'NLLOC' lines "
               "does not match number of 'END_NLLOC' lines").format(filename)
        raise Exception(msg)
    start_end_indices = []
    for start, end in zip(lines_start, lines_end):
        start_end_indices.append(start)
        start_end_indices.append(end)
    if any(np.diff(start_end_indices) < 1):
        msg = ("NLLOC HYP file '{}' seems corrupt, inconsistent "
               "positioning of 'NLLOC' and 'END_NLLOC' lines "
               "detected.").format(filename)
        raise Exception(msg)
    for start, end in zip(lines_start, lines_end):
        event = _read_single_hypocenter(
            lines[start:end + 1], coordinate_converter=coordinate_converter,
            original_picks=original_picks, **kwargs)
        cat.append(event)
    cat.creation_info.creation_time = UTCDateTime()
    cat.creation_info.version = "ObsPy %s" % __version__
    return cat


def _read_single_hypocenter(lines, coordinate_converter, original_picks,
                            **kwargs):
    """
    Given a list of lines (starting with a 'NLLOC' line and ending with a
    'END_NLLOC' line), parse them into an Event.
    """
    nlloc_file_format_version = None
    try:
        # some paranoid checks..
        assert lines[0].startswith("NLLOC ")
        assert lines[-1].startswith("END_NLLOC")
        for line in lines[1:-1]:
            assert not line.startswith("NLLOC ")
            assert not line.startswith("END_NLLOC")
    except Exception:
        msg = ("This should not have happened, please report this as a bug at "
               "https://github.com/obspy/obspy/issues.")
        raise Exception(msg)

    indices_phases = [None, None]
    for i, line in enumerate(lines):
        if line.startswith("PHASE "):
            indices_phases[0] = i
            # determine whether it's old format or new one which has one
            # additional item in middle. use position of magic character to
            # find out.
            separator_position = line.split().index('>')
            if separator_position == 15:
                nlloc_file_format_version = 1
            elif separator_position == 16:
                nlloc_file_format_version = 2
            else:
                msg = ("This should not happen, please open a ticket on "
                       "github and supply your nonlinloc file for debugging")
                raise NotImplementedError(msg)
        elif line.startswith("END_PHASE"):
            indices_phases[1] = i

    # extract PHASES lines (if any)
    if any(indices_phases):
        if not all(indices_phases):
            msg = ("NLLOC HYP file seems corrupt, 'PHASE' block is corrupt.")
            raise RuntimeError(msg)
        i1, i2 = indices_phases
        lines, phases_lines = lines[:i1] + lines[i2 + 1:], lines[i1 + 1:i2]
    else:
        phases_lines = []

    lines = dict([line.split(None, 1) for line in lines[:-1]])
    line = lines["SIGNATURE"]

    line = line.rstrip().split('"')[1]
    signature, version, date, time = line.rsplit(" ", 3)
    # new NLLoc > 6.0 seems to add prefix 'run:' before date
    if date.startswith('run:'):
        date = date[4:]
    signature = signature.strip()
    creation_time = UTCDateTime.strptime(date + time, str("%d%b%Y%Hh%Mm%S"))

    if coordinate_converter:
        # maximum likelihood origin location in km info line
        line = lines["HYPOCENTER"]
        x, y, z = coordinate_converter(*map(float, line.split()[1:7:2]))
    else:
        # maximum likelihood origin location lon lat info line
        line = lines["GEOGRAPHIC"]
        y, x, z = map(float, line.split()[8:13:2])

    # maximum likelihood origin time info line
    line = lines["GEOGRAPHIC"]

    year, mon, day, hour, min = map(int, line.split()[1:6])
    seconds = float(line.split()[6])
    time = UTCDateTime(year, mon, day, hour, min, seconds, strict=False)

    # distribution statistics line
    line = lines["STATISTICS"]
    covariance_xx = float(line.split()[7])
    covariance_yy = float(line.split()[13])
    covariance_zz = float(line.split()[17])
    stats_info_string = str(
        "Note: Depth/Latitude/Longitude errors are calculated from covariance "
        "matrix as 1D marginal (Lon/Lat errors as great circle degrees) "
        "while OriginUncertainty min/max horizontal errors are calculated "
        "from 2D error ellipsoid and are therefore seemingly higher compared "
        "to 1D errors. Error estimates can be reconstructed from the "
        "following original NonLinLoc error statistics line:\nSTATISTICS " +
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

    if "COMMENT" in lines:
        comment = lines["COMMENT"].strip()
        comment = comment.strip('\'"')
        comment = comment.strip()

    hor_unc, min_hor_unc, max_hor_unc, hor_unc_azim = \
        map(float, line.split()[1:9:2])

    nlloc_info_line = 'NLLOC ' + lines['NLLOC']

    # assign origin info
    event = Event()
    o = Origin()
    event.origins = [o]
    event.preferred_origin_id = o.resource_id
    o.origin_uncertainty = OriginUncertainty()
    o.quality = OriginQuality()
    ou = o.origin_uncertainty
    oq = o.quality
    o.comments.append(Comment(text=stats_info_string, force_resource_id=False))
    o.comments.append(Comment(text=nlloc_info_line, force_resource_id=False))
    event.comments.append(Comment(text=comment, force_resource_id=False))
    event.comments.append(Comment(text=nlloc_info_line,
                                  force_resource_id=False))

    # SIGNATURE field's first item is LOCSIG, which is supposed to be
    # 'Identification of an individual, institiution or other entity'
    # according to
    # http://alomax.free.fr/nlloc/soft6.00/control.html#_NLLoc_locsig_
    # so use it as author in creation info
    event.creation_info = CreationInfo(creation_time=creation_time,
                                       version=version,
                                       author=signature)
    o.creation_info = CreationInfo(creation_time=creation_time,
                                   version=version,
                                   author=signature)

    # nlloc writes location status in "NLLOC" line
    # char location status LOCATED, ABORTED, IGNORED, REJECTED
    # set evaluation status to "rejected" if it is anything but LOCATED
    nlloc_location_status = lines['NLLOC'].split()[1].strip('\'"')
    if nlloc_location_status in ('ABORTED', 'IGNORED', 'REJECTED'):
        o.evaluation_status = 'rejected'

    # negative values can appear on diagonal of covariance matrix due to a
    # precision problem in NLLoc implementation when location coordinates are
    # large compared to the covariances.
    o.longitude = x
    try:
        o.longitude_errors.uncertainty = kilometer2degrees(sqrt(covariance_xx))
    except ValueError:
        if covariance_xx < 0:
            msg = ("Negative value in XX value of covariance matrix, not "
                   "setting longitude error (epicentral uncertainties will "
                   "still be set in origin uncertainty).")
            warnings.warn(msg)
        else:
            raise
    o.latitude = y
    try:
        o.latitude_errors.uncertainty = kilometer2degrees(sqrt(covariance_yy))
    except ValueError:
        if covariance_yy < 0:
            msg = ("Negative value in YY value of covariance matrix, not "
                   "setting longitude error (epicentral uncertainties will "
                   "still be set in origin uncertainty).")
            warnings.warn(msg)
        else:
            raise
    o.depth = z * 1e3  # meters!
    o.depth_errors.uncertainty = sqrt(covariance_zz) * 1e3  # meters!
    o.depth_errors.confidence_level = 68
    o.depth_type = str("from location")
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
    ou.preferred_description = str("uncertainty ellipse")
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
        arrival = Arrival()
        o.arrivals.append(arrival)
        station = line[0]
        channel = line[2]
        phase = line[4]
        arrival.phase = phase
        if nlloc_file_format_version == 1:
            arrival.distance = kilometer2degrees(float(line[21]))
            arrival.azimuth = float(line[22])
            # do not read in take off angle, if the ray takeoff quality is
            # given as "0" for "unreliable", see #3224
            if int(line[25]) != 0:
                arrival.takeoff_angle = float(line[24])
            arrival.time_residual = float(line[16])
            arrival.time_weight = float(line[17])
        elif nlloc_file_format_version == 2:
            arrival.distance = kilometer2degrees(float(line[22]))
            arrival.azimuth = float(line[23])
            # do not read in take off angle, if the ray takeoff quality is
            # given as "0" for "unreliable", see #3224
            if int(line[26]) != 0:
                arrival.takeoff_angle = float(line[25])
            arrival.time_residual = float(line[17])
            arrival.time_weight = float(line[18])
        else:
            msg = ("This should not happen, please open a ticket on "
                   "github and supply your nonlinloc file for debugging")
            raise NotImplementedError(msg)
        pick = Pick()
        # have to split this into ints for overflow to work correctly
        date, hourmin, sec = map(str, line[6:9])
        ymd = [int(date[:4]), int(date[4:6]), int(date[6:8])]
        hm = [int(hourmin[:2]), int(hourmin[2:4])]
        t = UTCDateTime(*(ymd + hm), strict=False) + float(sec)
        pick.time = t
        # network codes are not used by NonLinLoc, so they can not be known
        # when reading the .hyp file.. if an inventory is provided, a lookup
        # is done
        widargs = _resolve_seedid(
            station=station, component=channel, time=t, phase=phase,
            unused_kwargs=True, **kwargs)
        wid = WaveformStreamID(*widargs)
        pick.waveform_id = wid
        pick.time_errors.uncertainty = float(line[10])
        pick.phase_hint = phase
        pick.onset = ONSETS.get(line[3].lower(), None)
        pick.polarity = POLARITIES.get(line[5].lower(), None)
        # try to determine original pick for each arrival
        for pick_ in original_picks:
            wid = pick_.waveform_id
            if station == wid.station_code and phase == pick_.phase_hint:
                pick = pick_
                break
        else:
            # warn if original picks were specified and we could not associate
            # the arrival correctly
            if original_picks:
                msg = ("Could not determine corresponding original pick for "
                       "arrival. "
                       "Falling back to pick information in NonLinLoc "
                       "hypocenter-phase file.")
                warnings.warn(msg)
        event.picks.append(pick)
        arrival.pick_id = pick.resource_id

    event.scope_resource_ids()

    return event


def write_nlloc_obs(catalog, filename, **kwargs):
    """
    Write a NonLinLoc Phase file (NLLOC_OBS) from a
    :class:`~obspy.core.event.catalog.Catalog` object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.Catalog` object, call this instead.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file-like object
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
        component = wid.channel_code or "?"
        onset = ONSETS_REVERSE.get(pick.onset, "?")
        phase_type = pick.phase_hint or "?"
        polarity = POLARITIES_REVERSE.get(pick.polarity, "?")
        date = pick.time.strftime("%Y%m%d")
        hourminute = pick.time.strftime("%H%M")
        seconds = pick.time.second + pick.time.microsecond * 1e-6
        if pick.time_errors.upper_uncertainty is not None and \
                pick.time_errors.lower_uncertainty is not None:
            time_error = (pick.time_errors.upper_uncertainty +
                          pick.time_errors.lower_uncertainty) / 2.0
        elif pick.time_errors.uncertainty is not None:
            time_error = pick.time_errors.uncertainty
        else:
            # see discussion in #2371
            msg = ("Writing pick without time uncertainty. Time uncertainty "
                   "will be written as '0.0'")
            warnings.warn(msg)
            time_error = 0.0
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
