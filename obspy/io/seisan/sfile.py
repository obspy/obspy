#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEISAN SFILE file format support for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import re

from obspy.geodetics import FlinnEngdahl

from obspy import UTCDateTime
from obspy.core.event import (Catalog, Event, Origin, Magnitude, Pick,
                              WaveformStreamID, QuantityError)

_fe = FlinnEngdahl()


def _get_resource_id(cmtname, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/seisan_sfile/%s/%s" % (cmtname, res_type)
    if tag is not None:
        res_id += "#" + tag
    return res_id


def _buffer_proxy(filename_or_buf, function, reset_fp=True,
                  file_mode="rb", *args, **kwargs):
    """
    Calls a function with an open file or file-like object as the first
    argument. If the file originally was a filename, the file will be
    opened, otherwise it will just be passed to the underlying function.

    :param filename_or_buf: File to pass.
    :type filename_or_buf: str, open file, or file-like object.
    :param function: The function to call.
    :param reset_fp: If True, the file pointer will be set to the initial
        position after the function has been called.
    :type reset_fp: bool
    :param file_mode: Mode to open file in if necessary.
    """
    try:
        position = filename_or_buf.tell()
        is_buffer = True
    except AttributeError:
        is_buffer = False

    if is_buffer is True:
        ret_val = function(filename_or_buf, *args, **kwargs)
        if reset_fp:
            filename_or_buf.seek(position, 0)
        return ret_val
    else:
        with open(filename_or_buf, file_mode) as fh:
            return function(fh, *args, **kwargs)


def _is_seisan_sfile(filename_or_buf):
    """
    Checks if the file is a SEISAN SFILE file.

    :param filename_or_buf: File to test.
    :type filename_or_buf: str or file-like object.
    """
    try:
        return _buffer_proxy(filename_or_buf, _internal_is_seisan_sfile,
                             reset_fp=True)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except (OSError, FileNotFoundError):
        return False


def _internal_is_seisan_sfile(buf):
    """
    Checks if the file is a SEISAN SFILE file.

    :param buf: File to check.
    :type buf: Open file or open file like object.
    """
    # The file format is so simple. Just attempt to read the first event. If
    # it passes it will be read again but that has really no
    # significant performance impact.
    try:
        _internal_read_seisan_sfile(buf, limit=1)
        return True
    except:
        return False


def _read_seisan_sfile(filename_or_buf, limit=None, **kwargs):
    """
    Reads a SEISAN SFILE file to a :class:`~obspy.core.event.Catalog` object.

    :param filename_or_buf: File to read.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf,
                         _internal_read_seisan_sfile,
                         limit=limit,
                         **kwargs)


pattern = re.compile(r"^\s*"          # Might start with whitespaces
                     r"\d{4}"           # 4 digits for the year
                     r"\s+"           # One or more whitespaces
                     r"\d{1,2}"       # 1-2 digits for month
                     r"\s*"           # Might be one or more spaces
                     r"\d{1,2}"       # 1-2 digits for the day
                     r"\s*"           # Might be one or more spaces
                     r"\d{1,2}"       # 1-2 digits for hours
                     r"\s*"           # Might be one or more spaces
                     r"\d{1,2}"       # 1-2 digits for the minutes
                     r"\s*"           # Might be one or more spaces
                     r"[\d\.]{1,4}"   # The seconds
                     r".*$"
                     )


def parse_pick(line, yr, mo, dy, ev_time):
    if not line[10:13] == 'AML':
        sta = line[0:5]
        cha = str(line[5:6]) + 'H' + str(line[7:8])
        net = 'CM'
        wfid = WaveformStreamID()
        wfid.station_code = sta
        wfid.channel_code = cha
        wfid.network_code = net
        pha = line[9:10]
        hr_pick = int(line[17:19])
        mn_pick = int(line[19:21])
        sc_pick = float(line[22:28])
        if not sc_pick == 60.0:
            if not hr_pick == 24:
                pick_time = UTCDateTime(
                    yr, mo, dy, hr_pick, mn_pick, sc_pick)
            else:
                hr_pick = 23
                pick_time = UTCDateTime(
                    yr, mo, dy, hr_pick, mn_pick, sc_pick) + 3600
        else:
            sc_pick = 59.9
            if not hr_pick == 24:
                pick_time = UTCDateTime(
                    yr, mo, dy, hr_pick, mn_pick, sc_pick) + 0.1
            else:
                hr_pick = 23
                pick_time = UTCDateTime(
                    yr, mo, dy, hr_pick, mn_pick, sc_pick) + 3600.1
        if pick_time < ev_time:
            pick_time = pick_time + 86400
        pick = Pick()
        pick.time = pick_time
        pick.phase_hint = pha
        pick.waveform_id = wfid
        return pick
    else:
        return 1


def _new_events_start_with_line(line):
    return bool(pattern.match(line))


def yield_events(buf):
    lines_for_event = []

    for line in buf:
        line = line.decode().strip()
        if not line:
            continue
        if _new_events_start_with_line(line):
            if lines_for_event:
                yield lines_for_event
                lines_for_event = []

        lines_for_event.append(line)
    yield lines_for_event


def _internal_read_seisan_sfile(buf, limit=None):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.event.Catalog` object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    cat = Catalog()

    for _i, lines in enumerate(yield_events(buf)):
        if limit and _i >= limit:
            return cat
        # We already know the first line is valid.
        yr = int(lines[0][0:4])
        mo = int(lines[0][5:7])
        dy = int(lines[0][7:9])
        hr = int(lines[0][10:12])
        mn = int(lines[0][12:14])
        sc = float(lines[0][15:19])
        if not sc == 60.0:
            ev_time = UTCDateTime(yr, mo, dy, hr, mn, sc)
        else:
            ev_time = UTCDateTime(yr, mo, dy, hr, mn, 59.9) + 0.1
        la = float(lines[0][22:29])
        lo = float(lines[0][29:37])
        dp = float(lines[0][37:42]) * 1e3
        ml = float(lines[0][55:58])

        event = Event()
        origin = Origin()

        cont_mag_stas = 0
        for line in lines[1:]:
            # Error line.
            if line[-1] == 'E':
                er_la = float(line[24:29])
                er_lo = float(line[29:37])
                er_dp = float(line[37:42])
                er_tm = float(line[15:19])
            # Check for pick.
            elif len(line) == 78:
                pick = parse_pick(line=line, yr=yr, mo=mo, dy=dy,
                                  ev_time=ev_time)
                if pick == 1:
                    cont_mag_stas += 1
                else:
                    event.picks.append(pick)
            # Ignore line.
            else:
                continue

        # ERRORS
        lat_er_qe = QuantityError()
        lon_er_qe = QuantityError()
        dep_er_qe = QuantityError()
        time_er_qe = QuantityError()
        lat_er_qe.uncertainty = er_la
        lon_er_qe.uncertainty = er_lo
        dep_er_qe.uncertainty = er_dp
        time_er_qe.uncertainty = er_tm
        # ORIGINS
        origin.longitude = lo
        origin.latitude = la
        origin.depth = dp
        origin.time = ev_time
        origin.latitude_errors = lat_er_qe
        origin.longitude_errors = lon_er_qe
        origin.depth_errors = dep_er_qe
        origin.time_errors = time_er_qe
        # MAGNITUDES
        magnitude = Magnitude()
        magnitude.mag = ml
        magnitude.magnitude_type = 'ML'
        magnitude.station_count = cont_mag_stas

        event.origins.append(origin)
        event.magnitudes.append(magnitude)
        cat.events.append(event)
    return cat
