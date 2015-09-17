# -*- coding: utf-8 -*-
"""
Reading of the K-NET and KiK-net ASCII format as defined on
http://www.kyoshin.bosai.go.jp.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from obspy import UTCDateTime, Stream, Trace
from obspy.core.trace import Stats
from datetime import datetime
import re
import numpy as np


class KnetFormatError(Exception):
    pass


class KnetDataError(Exception):
    pass


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


def _is_knet_ascii(filename_or_buf):
    """
    Checks if the file is a valid K-NET/KiK-net ASCII file.

    :param filename_or_buf: File to test.
    :type filename_or_buf: str or file-like object.
    """
    try:
        return _buffer_proxy(filename_or_buf, __is_knet_ascii, reset_fp=True)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except (OSError):
        return False


def __is_knet_ascii(buf):
    """
    Checks if the file is a valid K-NET/KiK-net ASCII file.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    first_string = buf.read(11)
    # File has less than 11 characters
    if len(first_string) != 11:
        return False
    if first_string == 'Origin Time':
        return True
    return False


def _prep_hdr_line(name, line):
    """
    Helper function to check the contents of a header line and split it.

    :param name: String that the line should start with.
    :type name: str
    :param line: Line to check and split.
    :type line: str
    """
    if not line.startswith(name):
        raise KnetFormatError("Expected line to start with %s but got %s " \
                              % (name, line))
    else:
        return line.split()


def _read_knet_hdr(hdrlines, convert_stnm=False, **kwargs):
    """
    Read the header values into a dictionary.

    :param hdrlines: List of the header lines of a a K-NET/KiK-net ASCII file
    :type hdrlines: list
    :param convert_stnm: For station names with 6 letters write the last two
    letters of the station code to the 'location' field
    :type convert_stnm: bool
    """
    hdrdict = {'knet':{}}
    hdrnames = ['Origin Time', 'Lat.', 'Long.', 'Depth. (km)', 'Mag.',
                'Station Code', 'Station Lat.', 'Station Long.',
                'Station Height(m)', 'Record Time', 'Sampling Freq(Hz)',
                'Duration Time(s)', 'Dir.', 'Scale Factor', 'Max. Acc. (gal)',
                'Last Correction', 'Memo.']
    _i = 0
    # Event information
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dt = flds[2] + ' ' + flds[3]
    dt = UTCDateTime(datetime.strptime(dt, '%Y/%m/%d %H:%M:%S'))
    # All times are in Japanese standard time which is 9 hours ahead of GMT
    dt -= 9 * 3600.
    hdrdict['knet']['evot'] = dt

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    lat = float(flds[1])
    hdrdict['knet']['evla'] = lat

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    lon = float(flds[1])
    hdrdict['knet']['evlo'] = lon

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dp = float(flds[2])
    hdrdict['knet']['evdp'] = dp

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    mag = float(flds[1])
    hdrdict['knet']['mag'] = mag

    # Station information
    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    # K-NET and KiK-Net station names can be more than 5 characters long
    # which will cause the station name to be truncated when writing the
    # the trace as miniSEED; if convert_stnm is enabled, the last two
    # letters of the station code are written to the 'location' field
    stnm = flds[2]
    location = ''
    convert_stnm = False
    if convert_stnm and len(stnm) > 5:
        location = stnm[-2:]
        stnm = stnm[:-2]
    if len(stnm) > 7:
        raise KnetFormatError("Station name can't be more than 7 characters\
        long!")
    hdrdict['station'] = stnm
    hdrdict['location'] = location

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['stla'] = float(flds[2])

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['stlo'] = float(flds[2])

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['stel'] = float(flds[2])

    # Data information
    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dt = flds[2] + ' ' + flds[3]
    # A 15 s delay is added to the record time by the
    # the K-NET and KiK-Net data logger
    dt = UTCDateTime(datetime.strptime(dt, '%Y/%m/%d %H:%M:%S')) - 15.0
    # All times are in Japanese standard time which is 9 hours ahead of GMT
    dt -= 9 * 3600.
    hdrdict['starttime'] = dt

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    freqstr = flds[2]
    m = re.search('[0-9]*', freqstr)
    freq = int(m.group())
    delta = 1.0 / freq
    hdrdict['delta'] = delta
    hdrdict['sampling_rate'] = freq

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['duration'] = float(flds[2])

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    channel = flds[1].replace('-', '')
    kiknetcomps = {'1':'NS1', '2':'EW1', '3':'UD1',
                   '4':'NS2', '5':'EW2', '6':'UD2'}
    if channel.strip() in kiknetcomps.keys():  # kiknet directions are 1-6
        channel = kiknetcomps[channel.strip()]
    hdrdict['channel'] = channel

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    eqn = flds[2]
    num, denom = eqn.split('/')
    num = float(re.search('[0-9]*', num).group())
    denom = float(denom)
    # convert the calibration from gal to m/s^2
    hdrdict['calib'] = 0.01 * num / denom

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    acc = float(flds[3])
    hdrdict['knet']['accmax'] = acc

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dt = flds[2] + ' ' + flds[3]
    dt = UTCDateTime(datetime.strptime(dt, '%Y/%m/%d %H:%M:%S'))
    # All times are in Japanese standard time which is 9 hours ahead of GMT
    dt -= 9 * 3600.
    hdrdict['knet']['last correction'] = dt

    # The comment ('Memo') field is optional
    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    if len(flds) > 1:
        hdrdict['knet']['comment'] = ' '.join(flds[1:])

    if len(hdrlines) != _i + 1:
        raise KnetFormatError("Expected %d header lines but got %d" \
                              % (_i + 1, len(hdrlines)))
    return hdrdict


def _read_knet_ascii(filename_or_buf, **kwargs):
    """
    Reads a K-NET/KiK-net ASCII file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: K-NET/KiK-net ASCII file to be read.
    :type filename: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf, __read_knet_ascii, **kwargs)


def __read_knet_ascii(buf, **kwargs):
    """
    Reads a K-NET/KiK-net ASCII file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    data = []
    hdrdict = {}

    cur_pos = buf.tell()
    buf.seek(0, 2)
    size = buf.tell()
    buf.seek(cur_pos, 0)

    # First read the headerlines
    headerlines = []
    while True:
        if buf.tell() >= size:
            break
        line = buf.readline()
        if line.startswith('Memo'):
            headerlines.append(line)
            hdrdict = _read_knet_hdr(headerlines, **kwargs)
            break
        headerlines.append(line)

    while True:
        if buf.tell() >= size:
            break
        line = buf.readline()
        parts = line.strip().split()
        data += [float(p) for p in parts]

    hdrdict['npts'] = len(data)
    elapsed = float(hdrdict['npts']) / float(hdrdict['sampling_rate'])
    hdrdict['endtime'] = hdrdict['starttime'] + elapsed
    # The FDSN network code for the National Research Institute for Earth
    # Science and Disaster Prevention (NEID JAPAN) is BO (Bosai-Ken Network)
    hdrdict['network'] = 'BO'

    data = np.array(data)
    stats = Stats(hdrdict)
    trace = Trace(data, header=stats)
    return Stream([trace])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
