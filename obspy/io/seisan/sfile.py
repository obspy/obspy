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

import math
import uuid
import warnings

from obspy import UTCDateTime
from obspy.core.event import (Catalog, Comment, Event, EventDescription,
                              Origin, Magnitude, FocalMechanism, MomentTensor,
                              Tensor, SourceTimeFunction)
from obspy.geodetics import FlinnEngdahl


from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude, Pick, WaveformStreamID, QuantityError

_fe = FlinnEngdahl()


def _get_resource_id(cmtname, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/cmtsolution/%s/%s" % (cmtname, res_type)
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
    Checks if the file is a CMTSOLUTION file.

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
    Checks if the file is a CMTSOLUTION file.

    :param buf: File to check.
    :type buf: Open file or open file like object.
    """
    # The file format is so simple. Just attempt to read the first event. If
    # it passes it will be read again but that has really no
    # significant performance impact.
    try:
        _internal_read_seisan_sfile(buf)
        return True
    except:
        return False


def _read_seisan_sfile(filename_or_buf, **kwargs):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.event.Catalog` object.

    :param filename_or_buf: File to read.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf,
                         _internal_read_seisan_sfile,
                         **kwargs)


def _internal_read_seisan_sfile(buf):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.event.Catalog` object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    cat = Catalog()
    lines = buf.readlines()
    n_lines = []
    for line in lines:
        line = line.decode()
        ls = line.split('\n')[0].split()
        if not ls == []:
            n_lines.append(line.split('\n')[0])
    indexes = []
    for i in range(len(n_lines)):
        if n_lines[i][79:80] == '1':
            indexes.append(i)
    ls_evs_grp = []
    prev = 0
    for index in indexes:
        ls_evs_grp.append(n_lines[prev:int(index)])
        prev = index
    ls_evs_grp.append(n_lines[indexes[-1]:])
    ls_evs_grp = [x for x in ls_evs_grp if len(x) != 0]
    for ev in ls_evs_grp:
        yr = int(ev[0][1:5])
        mo = int(ev[0][6:8])
        dy = int(ev[0][8:10])
        hr = int(ev[0][11:13])
        mn = int(ev[0][13:15])
        sc = float(ev[0][16:20])
        if not sc == 60.0:
            ev_time = UTCDateTime(yr, mo, dy, hr, mn, sc)
        else:
            ev_time = UTCDateTime(yr, mo, dy, hr, mn, 59.9) + 0.1
        la = float(ev[0][23:30])
        lo = float(ev[0][30:38])
        dp = float(ev[0][38:43]) * 1e3
        ml = float(ev[0][56:59])
    #       print(ev[0])
        event = Event()
        origin = Origin()

        cont_mag_stas = 0
        for ev2 in ev:
            if ev2[79:80] == 'E':
                #                       print(ev2)
                er_la = float(ev2[23:30])
                er_lo = float(ev2[30:38])
                er_dp = float(ev2[38:43])
                er_tm = float(ev2[16:20])
            # PICKS
            elif ev2[79:80] == ' ':
                if not ev2[11:14] == 'AML':
                    #                               print(ev2)
                    sta = ev2[0:6]
                    cha = str(ev2[6:7]) + 'H' + str(ev2[7:8])
                    net = 'CM'
                    wfid = WaveformStreamID()
                    wfid.station_code = sta
                    wfid.channel_code = cha
                    wfid.network_code = net
                    pha = ev2[10:11]
                    hr_pick = int(ev2[18:20])
                    mn_pick = int(ev2[20:22])
                    sc_pick = float(ev2[23:29])
                #       uncert = float(ev2[64:68])
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
                #       qe = QuantityError()
                #       qe.uncertainty = uncert
                    pick = Pick()
                    pick.time = pick_time
                #       pick.time_errors = qe
                    pick.phase_hint = pha
                    pick.waveform_id = wfid
                    event.picks.append(pick)
                else:
                    cont_mag_stas += 1
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

