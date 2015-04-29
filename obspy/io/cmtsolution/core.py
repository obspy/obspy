#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMTSOLUTION file format support for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import math
import warnings


from obspy import UTCDateTime
from obspy.core.event import (Catalog, Event, Origin, Magnitude,
                              FocalMechanism, MomentTensor, Tensor,
                              SourceTimeFunction)
from obspy.geodetics import FlinnEngdahl

fe = FlinnEngdahl()



def _buffer_proxy(filename_or_buf, function, reset_fp=True, *args, **kwargs):
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
        with open(filename_or_buf, "rb") as fh:
            return function(fh, *args, **kwargs)


def _is_cmtsolution(filename_or_buf):
    """
    Checks if the file is a CMTSOLUTION file.

    :param filename_or_buf: File to test.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf, __is_cmtsolution, reset_fp=True)


def __is_cmtsolution(buf):
    """
    Checks if the file is a CMTSOLUTION file.

    :param buf: File to check.
    :type buf: Open file or open file like object.
    """
    # The file format is so simple. Just attempt to read it. If it passes it
    # will be read again but that has really no performance impact.
    try:
        _read_cmtsolution(buf)
        return True
    except:
        return False


def _read_cmtsolution(filename_or_buf, **kwargs):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.stream.Stream` object.

    :param filename_or_buf: File to read.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf, __read_cmtsolution, **kwargs)


def __read_cmtsolution(buf, **kwargs):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.stream.Stream` object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    # The first line encodes the preliminary epicenter.
    line = buf.readline()
    origin_time = line[4:].strip().split()[:6]
    values = list(map(int, origin_time[:-1])) + \
             [float(origin_time[-1])]
    try:
        origin_time = UTCDateTime(*values)
    except (TypeError, ValueError):
        warnings.warn("Could not determine origin time from line: %s. Will "
                      "be set to zero." % line)
        origin_time = UTCDateTime(0)
    line = line[6:]
    latitude, longitude, depth, body_wave_mag, surface_wave_mag = \
        map(float, line[:5])

    preliminary_origin = Origin(
        time=origin_time,
        longitude=longitude,
        latitude=latitude,
        # Depth is in meters.
        depth=depth * 1000.0,
        origin_type="hypocenter",
        region=fe.get_region(longitude=longitude, latitude=latitude),
        evaluation_status="preliminary"
    )

    preliminary_bw_magnitude = Magnitude(
        mag=body_wave_mag, magnitude_type="Mb",
        evaluation_status="preliminary",
        origin_id=preliminary_origin.resource_id)

    preliminary_sw_magnitude = Magnitude(
        mag=surface_wave_mag, magnitude_type="MS",
        evaluation_status="preliminary",
        origin_id=preliminary_origin.resource_id)

    # The rest encodes the centroid solution.
    buf.readline()  # Skip event name.

    values = ["time_shift", "half_duration", "latitude", "longitude",
              "depth", "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    cmt_values = {_i: float(buf.readline().strip().split()[-1])
                  for _i in values}

    # Moment magnitude calculation in dyne * cm.
    m_0 = 1.0 / math.sqrt(2.0) * math.sqrt(
        cmt_values["m_rr"] ** 2 +
        cmt_values["m_tt"] ** 2 +
        cmt_values["m_pp"] ** 2 +
        2.0 * cmt_values["m_rt"] ** 2 +
        2.0 * cmt_values["m_rp"] ** 2 +
        2.0 * cmt_values["m_tp"] ** 2)
    m_w = 2.0 / 3.0 * (math.log10(m_0) - 16.1)

    # Convert to meters.
    cmt_values["depth"] *= 1000.0
    # Convert to Newton meter.
    values = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    for value in values:
        cmt_values[value] /= 1E7

    cmt_origin = Origin(
        time=origin_time + cmt_values["time_shift"],
        longitude=cmt_values["longitude"],
        latitude=cmt_values["latitude"],
        depth=cmt_values["depth"],
        origin_type="centroid",
        # Could rarely be different than the epicentral region.
        region=fe.get_region(longitude=cmt_values["longitude"],
                             latitude=cmt_values["latitude"])
        # No evaluation status as it could be any of several and the file
        # format does not provide that information.
    )

    cmt_mag = Magnitude(
        mag=m_w,
        magnitude_type="mw",
        origin_id=cmt_origin.resource_id
    )

    foc_mec = FocalMechanism(
        # The preliminary origin most likely triggered the focal mechanism
        # determination.
        triggering_origin_id=preliminary_origin.resource_id
    )

    tensor = Tensor(
        m_rr=cmt_values["m_rr"],
        m_pp=cmt_values["m_pp"],
        m_tt=cmt_values["m_tt"],
        m_rt=cmt_values["m_rt"],
        m_rp=cmt_values["m_rp"],
        m_tp=cmt_values["m_tp"]
    )

    # Source time function is a triangle, according to the SPECFEM manual.
    stf = SourceTimeFunction(
        type="triangle",
        # The duration is twice the half duration.
        duration=2.0 * cmt_values["half_duration"]
    )

    mt = MomentTensor(
        derived_origin_id=cmt_origin.resource_id,
        moment_magnitude_id=cmt_mag.resource_id,
        # In Nm.
        scalar_moment=m_0 / 1E7,
        tensor=tensor,
        source_time_function=stf
    )

    # Assemble everything.
    foc_mec.moment_tensor = mt

    ev = Event()
    ev.origins.append(cmt_origin)
    ev.origins.append(preliminary_origin)
    ev.magnitudes.append(cmt_mag)
    ev.magnitudes.append(preliminary_bw_magnitude)
    ev.magnitudes.append(preliminary_sw_magnitude)
    ev.focal_mechanisms.append(foc_mec)

    return Catalog(events=[ev])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
