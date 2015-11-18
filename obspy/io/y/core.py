# -*- coding: utf-8 -*-
"""
Y bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import re
import warnings
from struct import unpack

import numpy as np

from obspy import Stream
from obspy.core.compatibility import from_buffer
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict


INVALID_CHAR_MSG = (
    "Invalid non-ASCII characters in Y file header detected (%s). "
    "These were ignored.")


def _unpack_with_asciiz_and_decode(fmt, data):
    """
    Unpack binary data and decode ASCII bytestrings, stripping ASCIIZ
    bytestrings correctly as specified by Y format definition. In addition to
    format flags defined by :py:func:`struct.unpack`, "z" can be used to denote
    ASCIIZ fields.

    :param fmt: see :py:func:`struct.unpack`
    :param data: see :py:func:`struct.unpack`
    :returns: see :py:func:`struct.unpack` but with bytestrings being decoded
    """
    fmt_list = re.findall(b'[a-zA-Z]', fmt)
    z_positions = [pos for pos, fmt_ in enumerate(fmt_list) if fmt_ == b"z"]
    s_positions = [pos for pos, fmt_ in enumerate(fmt_list) if fmt_ == b"s"]

    parts = list(unpack(fmt.replace(b"z", b"s"), data))

    # special handling for ASCIIZ fields:
    # strip everything after first (if any) ASCII NULL character *before*
    # decoding (those need not be valid encoded ASCII bytes and should be
    # ignored)
    for i in z_positions:
        part = parts[i]
        terminal_index = part.find(b"\x00")
        if terminal_index != -1:
            parts[i] = part[:terminal_index]
    # decode all bytestrings from ASCII
    for i in z_positions + s_positions:
        part = parts[i]
        try:
            part = part.decode('ascii', errors="strict")
        except UnicodeError as e:
            warnings.warn(INVALID_CHAR_MSG % str(e), UserWarning)
            part = part.decode('ascii', errors="ignore")
        parts[i] = part
    # right-strip all BLANKPADDED fields
    for i in s_positions:
        parts[i] = parts[i].rstrip()

    return tuple(parts)


def __parse_tag(fh):
    """
    Reads and parses a single tag.

    returns endian, tag_type, next_tag, next_same
    """
    data = fh.read(16)
    # byte order format for this data. Uses letter “I” for Intel format
    # data (little endian) or letter “M” for Motorola (big endian) format
    format = unpack(b'=c', data[0:1])[0]
    if format == b'I':
        endian = b'<'
    elif format == b'M':
        endian = b'>'
    else:
        raise ValueError('Invalid tag: missing byte order information')
    # magic: check for magic number "31"
    magic = unpack(endian + b'B', data[1:2])[0]
    if magic != 31:
        raise ValueError('Invalid tag: missing magic number')
    # tag type: the type of data attached to this tag.
    tag_type = unpack(endian + b'H', data[2:4])[0]
    # NextTag is the offset in bytes from the end of this tag to the start of
    # the next tag. That means, the offset is the size of the data attached
    # to this tag.
    next_tag = unpack(endian + b'i', data[4:8])[0]
    # NextSame is the offset in bytes from the end of this tag to the start
    # of the next tag with the same type. If zero, there is no next tag with
    # the same type.
    next_same = unpack(endian + b'i', data[8:12])[0]
    return endian, tag_type, next_tag, next_same


def _is_y(filename):
    """
    Checks whether a file is a Nanometrics Y file or not.

    :type filename: str
    :param filename: Name of the Nanometrics Y file to be checked.
    :rtype: bool
    :return: ``True`` if a Nanometrics Y file.

    .. rubric:: Example

    >>> _is_y("/path/to/YAYT_BHZ_20021223.124800")  #doctest: +SKIP
    True
    """
    try:
        # get first tag (16 bytes)
        with open(filename, 'rb') as fh:
            _, tag_type, _, _ = __parse_tag(fh)
    except:
        return False
    # The first tag in a Y-file must be the TAG_Y_FILE tag (tag type 0)
    if tag_type != 0:
        return False
    return True


def _read_y(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a Nanometrics Y file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Nanometrics Y file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/YAYT_BHZ_20021223.124800")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    .AYT..BHZ | 2002-12-23T12:48:00.000100Z - ... | 100.0 Hz, 18000 samples
    """
    # The first tag in a Y-file must be the TAG_Y_FILE (0) tag. This must be
    # followed by the following tags, in any order:
    #   TAG_STATION_INFO (1)
    #   TAG_STATION_LOCATION (2)
    #   TAG_STATION_PARAMETERS (3)
    #   TAG_STATION_DATABASE (4)
    #   TAG_SERIES_INFO (5)
    #   TAG_SERIES_DATABASE (6)
    # The following tag is optional:
    #   TAG_STATION_RESPONSE (26)
    # The last tag in the file must be a TAG_DATA_INT32 (7) tag. This tag must
    # be followed by an array of LONG's. The number of entries in the array
    # must agree with what was described in the TAG_SERIES_INFO data.
    with open(filename, 'rb') as fh:
        trace = Trace()
        trace.stats.y = AttribDict()
        count = -1
        while True:
            endian, tag_type, next_tag, _next_same = __parse_tag(fh)
            if tag_type == 1:
                # TAG_STATION_INFO
                # UCHAR Update[8]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # UCHAR Station[5] (BLANKPAD)
                #   Station is the five letter SEED format station
                #   identification.
                # UCHAR Location[2] (BLANKPAD)
                #   Location Location is the two letter SEED format location
                #   identification.
                # UCHAR Channel[3] (BLANKPAD)
                #   Channel Channel is the three letter SEED format channel
                #   identification.
                # UCHAR NetworkID[51] (ASCIIZ)
                #   This is some descriptive text identifying the network.
                # UCHAR SiteName[61] (ASCIIZ)
                #   SiteName is some text identifying the site.
                # UCHAR Comment[31] (ASCIIZ)
                #   Comment is any comment for this station.
                # UCHAR SensorType[51] (ASCIIZ)
                #   SensorType is some text describing the type of sensor used
                #   at the station.
                # UCHAR DataFormat[7] (ASCIIZ)
                #   DataFormat is some text describing the data format recorded
                #   at the station.
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(
                    b'5s2s3s51z61z31z51z7z', data[8:])
                trace.stats.station = parts[0]
                trace.stats.location = parts[1]
                trace.stats.channel = parts[2]
                # extra
                params = AttribDict()
                params.network_id = parts[3]
                params.site_name = parts[4]
                params.comment = parts[5]
                params.sensor_type = parts[6]
                params.data_format = parts[7]
                trace.stats.y.tag_station_info = params
            elif tag_type == 2:
                # TAG_STATION_LOCATION
                # UCHAR Update[8]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # FLOAT Latitude
                #   Latitude in degrees of the location of the station. The
                #   latitude should be between -90 (South) and +90 (North).
                # FLOAT Longitude
                #   Longitude in degrees of the location of the station. The
                #   longitude should be between -180 (West) and +180 (East).
                # FLOAT Elevation
                #   Elevation in meters above sea level of the station.
                # FLOAT Depth
                #   Depth is the depth in meters of the sensor.
                # FLOAT Azimuth
                #   Azimuth of the sensor in degrees clockwise.
                # FLOAT Dip
                #   Dip is the dip of the sensor. 90 degrees is defined as
                #   vertical right way up.
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(
                    endian + b'ffffff', data[8:])
                params = AttribDict()
                params.latitude = parts[0]
                params.longitude = parts[1]
                params.elevation = parts[2]
                params.depth = parts[3]
                params.azimuth = parts[4]
                params.dip = parts[5]
                trace.stats.y.tag_station_location = params
            elif tag_type == 3:
                # TAG_STATION_PARAMETERS
                # UCHAR Update[16]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # REALTIME StartValidTime
                #   Time that the information in these records became valid.
                # REALTIME EndValidTime
                #   Time that the information in these records became invalid.
                # FLOAT Sensitivity
                #   Sensitivity of the sensor in nanometers per bit.
                # FLOAT SensFreq
                #   Frequency at which the sensitivity was measured.
                # FLOAT SampleRate
                #   This is the number of samples per second. This value can be
                #   less than 1.0. (i.e. 0.1)
                # FLOAT MaxClkDrift
                #   Maximum drift rate of the clock in seconds per sample.
                # UCHAR SensUnits[24] (ASCIIZ)
                #   Some text indicating the units in which the sensitivity was
                #   measured.
                # UCHAR CalibUnits[24] (ASCIIZ)
                #   Some text indicating the units in which calibration input
                #   was measured.
                # UCHAR ChanFlags[27] (BLANKPAD)
                #   Text indicating the channel flags according to the SEED
                #   definition.
                # UCHAR UpdateFlag
                #   This flag must be “N” or “U” according to the SEED
                #   definition.
                # UCHAR Filler[4]
                #   Filler Pads out the record to satisfy the alignment
                #   restrictions for reading data on a SPARC processor.
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(
                    endian + b'ddffff24z24z27sc4s', data[16:])
                trace.stats.sampling_rate = parts[4]
                # extra
                params = AttribDict()
                params.start_valid_time = parts[0]
                params.end_valid_time = parts[1]
                params.sensitivity = parts[2]
                params.sens_freq = parts[3]
                params.sample_rate = parts[4]
                params.max_clk_drift = parts[5]
                params.sens_units = parts[6]
                params.calib_units = parts[7]
                params.chan_flags = parts[8]
                params.update_flag = parts[9]
                trace.stats.y.tag_station_parameters = params
            elif tag_type == 4:
                # TAG_STATION_DATABASE
                # UCHAR Update[8]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # REALTIME LoadDate
                #   Date the information was loaded into the database.
                # UCHAR Key[16]
                #   Unique key that identifies this record in the database.
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(
                    endian + b'd16s', data[8:])
                params = AttribDict()
                params.load_date = parts[0]
                params.key = parts[1]
                trace.stats.y.tag_station_database = params
            elif tag_type == 5:
                # TAG_SERIES_INFO
                # UCHAR Update[16]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # REALTIME StartTime
                #   This is start time of the data in this series.
                # REALTIME EndTime
                #   This is end time of the data in this series.
                # ULONG NumSamples
                #   This is the number of samples of data in this series.
                # LONG DCOffset
                #   DCOffset is the DC offset of the data.
                # LONG MaxAmplitude
                #   MaxAmplitude is the maximum amplitude of the data.
                # LONG MinAmplitude
                #   MinAmplitude is the minimum amplitude of the data.
                # UCHAR Format[8] (ASCIIZ)
                #   This is the format of the data. This should always be
                #   “YFILE”.
                # UCHAR FormatVersion[8] (ASCIIZ)
                #   FormatVersion is the version of the format of the data.
                #   This should always be “5.0”
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(
                    endian + b'ddLlll8z8z', data[16:])
                trace.stats.starttime = UTCDateTime(parts[0])
                count = parts[2]
                # extra
                params = AttribDict()
                params.endtime = UTCDateTime(parts[1])
                params.num_samples = parts[2]
                params.dc_offset = parts[3]
                params.max_amplitude = parts[4]
                params.min_amplitude = parts[5]
                params.format = parts[6]
                params.format_version = parts[7]
                trace.stats.y.tag_series_info = params
            elif tag_type == 6:
                # TAG_SERIES_DATABASE
                # UCHAR Update[8]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # REALTIME LoadDate
                #   Date the information was loaded into the database.
                # UCHAR Key[16]
                #   Unique key that identifies this record in the database.
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(
                    endian + b'd16s', data[8:])
                params = AttribDict()
                params.load_date = parts[0]
                params.key = parts[1]
                trace.stats.y.tag_series_database = params
            elif tag_type == 26:
                # TAG_STATION_RESPONSE
                # UCHAR Update[8]
                #   This field is only used internally for administrative
                #   purposes.  It should always be set to zeroes.
                # UCHAR PathName[260]
                #  PathName is the full name of the file which contains the
                #  response information for this station.
                data = fh.read(next_tag)
                parts = _unpack_with_asciiz_and_decode(b'260s', data[8:])
                params = AttribDict()
                params.path_name = parts[0]
                trace.stats.y.tag_station_response = params
            elif tag_type == 7:
                # TAG_DATA_INT32
                trace.data = from_buffer(
                    fh.read(np.dtype(np.int32).itemsize * count),
                    dtype=np.int32)
                # break loop as TAG_DATA_INT32 should be the last tag in file
                break
            else:
                fh.seek(next_tag, 1)
    return Stream([trace])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
