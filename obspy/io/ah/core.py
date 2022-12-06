# -*- coding: utf-8 -*-
"""
AH bindings to ObsPy core module.

An AH file is used for the storage of binary seismic time series data.
The file is portable among machines of varying architecture by virtue of
its XDR implementation. It is composed of a variable-sized header containing
a number of values followed by the time series data.

.. seealso:: ftp://www.orfeus-eu.org/pub/software/mirror/ldeo.columbia/

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.util.attribdict import AttribDict
from obspy.io.ah import xdrlib

AH1_CODESIZE = 6
AH1_CHANSIZE = 6
AH1_STYPESIZE = 8
AH1_COMSIZE = 80
AH1_LOGSIZE = 202


def _is_ah(filename):
    """
    Checks whether a file is AH waveform data or not.

    :type filename: str
    :param filename: AH file to be checked.
    :rtype: bool
    :return: ``True`` if a AH waveform file.
    """
    if _get_ah_version(filename):
        return True
    return False


def _read_ah(filename, **kwargs):  # @UnusedVariable
    """
    Reads an AH waveform file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: AH file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream with Traces specified by given file.
    """
    version = _get_ah_version(filename)
    if version == '2.0':
        return _read_ah2(filename)
    else:
        return _read_ah1(filename)


def _get_ah_version(filename):
    """
    Returns version of AH waveform data.

    :type filename: str
    :param filename: AH v1 file to be checked.
    :rtype: str or False
    :return: version string of AH waveform data or ``False`` if unknown.
    """
    with open(filename, "rb") as fh:
        # read first 8 bytes with XDR library
        try:
            data = xdrlib.Unpacker(fh.read(8))
            # check for magic version number
            magic = data.unpack_int()
        except Exception:
            return False
        if magic == 1100:
            try:
                # get record length
                length = data.unpack_uint()
                # read first record
                fh.read(length)
            except Exception:
                return False
            # seems to be AH v2
            return '2.0'
        elif magic == 6:
            # AH1 has no magic variable :/
            # so we have to use some fixed values as indicators
            try:
                fh.seek(12)
                if xdrlib.Unpacker(fh.read(4)).unpack_int() != 6:
                    return False
                fh.seek(24)
                if xdrlib.Unpacker(fh.read(4)).unpack_int() != 8:
                    return False
                fh.seek(700)
                if xdrlib.Unpacker(fh.read(4)).unpack_int() != 80:
                    return False
                fh.seek(784)
                if xdrlib.Unpacker(fh.read(4)).unpack_int() != 202:
                    return False
            except Exception:
                return False
            return '1.0'
        else:
            return False


def _unpack_string(data):
    data = data.unpack_string().split(b'\x00', 1)[0].strip()
    try:
        data = data.decode("utf-8")
    except UnicodeDecodeError:
        msg = f'can not decode {data} as UTF-8, decoding with replacing errors'
        warnings.warn(msg)
        data = data.decode("utf-8", errors="replace")
    return data


def _read_ah1(filename):
    """
    Reads an AH v1 waveform file and returns a Stream object.

    :type filename: str
    :param filename: AH v1 file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream with Traces specified by given file.
    """

    def _unpack_trace(data):
        ah_stats = AttribDict({
            'version': '1.0',
            'event': AttribDict(),
            'station': AttribDict(),
            'record': AttribDict(),
            'extras': []
        })

        # station info
        ah_stats.station.code = _unpack_string(data)
        ah_stats.station.channel = _unpack_string(data)
        ah_stats.station.type = _unpack_string(data)
        ah_stats.station.latitude = data.unpack_float()
        ah_stats.station.longitude = data.unpack_float()
        ah_stats.station.elevation = data.unpack_float()
        ah_stats.station.gain = data.unpack_float()
        ah_stats.station.normalization = data.unpack_float()  # A0
        poles = []
        zeros = []
        for _i in range(0, 30):
            r = data.unpack_float()
            i = data.unpack_float()
            poles.append(complex(r, i))
            r = data.unpack_float()
            i = data.unpack_float()
            zeros.append(complex(r, i))
        # first value describes number of poles/zeros
        npoles = int(poles[0].real) + 1
        nzeros = int(zeros[0].real) + 1
        ah_stats.station.poles = poles[1:npoles]
        ah_stats.station.zeros = zeros[1:nzeros]

        # event info
        ah_stats.event.latitude = data.unpack_float()
        ah_stats.event.longitude = data.unpack_float()
        ah_stats.event.depth = data.unpack_float()
        ot_year = data.unpack_int()
        ot_mon = data.unpack_int()
        ot_day = data.unpack_int()
        ot_hour = data.unpack_int()
        ot_min = data.unpack_int()
        ot_sec = data.unpack_float()
        try:
            ot = UTCDateTime(ot_year, ot_mon, ot_day, ot_hour, ot_min, ot_sec)
        except Exception:
            ot = None
        ah_stats.event.origin_time = ot
        ah_stats.event.comment = _unpack_string(data)

        # record info
        ah_stats.record.type = dtype = data.unpack_int()  # data type
        ah_stats.record.ndata = ndata = data.unpack_uint()  # number of samples
        ah_stats.record.delta = data.unpack_float()  # sampling interval
        ah_stats.record.max_amplitude = data.unpack_float()
        at_year = data.unpack_int()
        at_mon = data.unpack_int()
        at_day = data.unpack_int()
        at_hour = data.unpack_int()
        at_min = data.unpack_int()
        at_sec = data.unpack_float()
        at = UTCDateTime(at_year, at_mon, at_day, at_hour, at_min, at_sec)
        ah_stats.record.start_time = at
        ah_stats.record.abscissa_min = data.unpack_float()
        ah_stats.record.comment = _unpack_string(data)
        ah_stats.record.log = _unpack_string(data)

        # extras
        ah_stats.extras = data.unpack_array(data.unpack_float)

        # unpack data using dtype from record info
        if dtype == 1:
            # float
            temp = data.unpack_farray(ndata, data.unpack_float)
        elif dtype == 6:
            # double
            temp = data.unpack_farray(ndata, data.unpack_double)
        else:
            # e.g. 3 (vector), 2 (complex), 4 (tensor)
            msg = 'Unsupported AH v1 record type %d'
            raise NotImplementedError(msg % (dtype))
        tr = Trace(np.array(temp))
        tr.stats.ah = ah_stats
        tr.stats.delta = ah_stats.record.delta
        tr.stats.starttime = ah_stats.record.start_time
        tr.stats.station = ah_stats.station.code
        tr.stats.channel = ah_stats.station.channel
        return tr

    st = Stream()
    with open(filename, "rb") as fh:
        # read with XDR library
        data = xdrlib.Unpacker(fh.read())
        # loop as long we can read records
        while True:
            try:
                tr = _unpack_trace(data)
                st.append(tr)
            except EOFError:
                break
        return st


def _write_ah1(stream, filename):
    """
    Writes a Stream object to an AH v1 waveform file.

    :type stream:
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: open file, or file-like object

    """
    packer = xdrlib.Packer()

    for tr in stream:
        if hasattr(tr.stats, 'ah'):
            packer = _pack_trace_with_ah_dict(
                tr, packer, AH1_CODESIZE, AH1_CHANSIZE, AH1_STYPESIZE,
                AH1_COMSIZE, AH1_LOGSIZE)
        else:
            packer = _pack_trace_wout_ah_dict(
                tr, packer, AH1_CODESIZE, AH1_CHANSIZE, AH1_STYPESIZE,
                AH1_COMSIZE, AH1_LOGSIZE)

    with open(filename, 'wb') as fh:
        fh.write(packer.get_buffer())


def _pack_trace_with_ah_dict(tr, packer, codesize, chansize,
                             stypesize, comsize, logsize):

    # station info
    packer.pack_int(codesize)
    packer.pack_fstring(codesize, tr.stats.station.encode('utf-8'))
    packer.pack_int(chansize)
    packer.pack_fstring(chansize, tr.stats.channel.encode('utf-8'))

    packer.pack_int(stypesize)
    packer.pack_fstring(stypesize, tr.stats.ah.station.type.encode('utf-8'))
    packer.pack_float(tr.stats.ah.station.latitude)
    packer.pack_float(tr.stats.ah.station.longitude)
    packer.pack_float(tr.stats.ah.station.elevation)
    packer.pack_float(tr.stats.ah.station.gain)
    packer.pack_float(tr.stats.ah.station.normalization)

    poles = tr.stats.ah.station.poles
    zeros = tr.stats.ah.station.zeros

    # Poles and Zeros
    packer.pack_float(len(poles))
    packer.pack_float(0)
    packer.pack_float(len(zeros))
    packer.pack_float(0)

    for _i in range(1, 30):
        try:
            r, i = poles[_i].real, poles[_i].imag
        except IndexError:
            r, i = 0, 0
        packer.pack_float(r)
        packer.pack_float(i)

        try:
            r, i = zeros[_i].real, zeros[_i].imag
        except IndexError:
            r, i = 0, 0
        packer.pack_float(r)
        packer.pack_float(i)

    # event info
    packer.pack_float(tr.stats.ah.event.latitude)
    packer.pack_float(tr.stats.ah.event.longitude)
    packer.pack_float(tr.stats.ah.event.depth)
    if tr.stats.ah.event.origin_time is not None:
        packer.pack_int(tr.stats.ah.event.origin_time.year)
        packer.pack_int(tr.stats.ah.event.origin_time.month)
        packer.pack_int(tr.stats.ah.event.origin_time.day)
        packer.pack_int(tr.stats.ah.event.origin_time.hour)
        packer.pack_int(tr.stats.ah.event.origin_time.minute)
        packer.pack_float(tr.stats.ah.event.origin_time.second)
    else:
        packer.pack_int(0)
        packer.pack_int(0)
        packer.pack_int(0)
        packer.pack_int(0)
        packer.pack_int(0)
        packer.pack_float(0)

    packer.pack_int(comsize)
    packer.pack_fstring(comsize, tr.stats.ah.event.comment.encode('utf-8'))

    # record info
    dtype = tr.stats.ah.record.type
    packer.pack_int(dtype)
    ndata = tr.stats.npts
    packer.pack_uint(ndata)
    packer.pack_float(tr.stats.ah.record.delta)
    packer.pack_float(tr.stats.ah.record.max_amplitude)
    packer.pack_int(tr.stats.ah.record.start_time.year)
    packer.pack_int(tr.stats.ah.record.start_time.month)
    packer.pack_int(tr.stats.ah.record.start_time.day)
    packer.pack_int(tr.stats.ah.record.start_time.hour)
    packer.pack_int(tr.stats.ah.record.start_time.minute)
    packer.pack_float(tr.stats.ah.record.start_time.second)
    packer.pack_float(tr.stats.ah.record.abscissa_min)
    packer.pack_int(comsize)
    packer.pack_fstring(comsize, tr.stats.ah.record.comment.encode('utf-8'))
    packer.pack_int(logsize)
    packer.pack_fstring(logsize, tr.stats.ah.record.log.encode('utf-8'))

    # # extras
    packer.pack_array(tr.stats.ah.extras, packer.pack_float)

    # pack data using dtype from record info
    if dtype == 1:
        # float
        packer.pack_farray(ndata, tr.data, packer.pack_float)
    elif dtype == 6:
        # double
        packer.pack_farray(ndata, tr.data, packer.pack_double)
    else:
        # e.g. 3 (vector), 2 (complex), 4 (tensor)
        msg = 'Unsupported AH v1 record type %d'
        raise NotImplementedError(msg % (dtype))

    return packer


def _pack_trace_wout_ah_dict(tr, packer, codesize, chansize,
                             stypesize, comsize, logsize):
    """
    Entry are packed in the same order as shown in
    _pack_trace_with_ah_dict .The missing information
    is replaced with zeros
    station info
    """
    packer.pack_int(codesize)
    packer.pack_fstring(codesize, tr.stats.station.encode('utf-8'))
    packer.pack_int(chansize)
    packer.pack_fstring(chansize, tr.stats.channel.encode('utf-8'))
    packer.pack_int(stypesize)
    packer.pack_fstring(stypesize, 'null'.encode('utf-8'))
    # There is no information about latitude, longitude, elevation,
    # gain and normalization in the basic stream object,  are set to 0
    packer.pack_float(0)
    packer.pack_float(0)
    packer.pack_float(0)
    packer.pack_float(0)
    packer.pack_float(0)

    # Poles and Zeros are not provided by stream object, are set to 0
    for _i in range(0, 30):
        packer.pack_float(0)
        packer.pack_float(0)
        packer.pack_float(0)
        packer.pack_float(0)

    # event info
    packer.pack_float(0)
    packer.pack_float(0)
    packer.pack_float(0)
    packer.pack_int(0)
    packer.pack_int(0)
    packer.pack_int(0)
    packer.pack_int(0)
    packer.pack_int(0)
    packer.pack_float(0)

    packer.pack_int(comsize)
    packer.pack_fstring(comsize, 'null'.encode('utf-8'))

    # record info
    dtype = type(tr.data[0])
    if '32' in str(dtype):
        dtype = 1
    elif '64' in str(dtype):
        dtype = 6

    packer.pack_int(dtype)
    ndata = tr.stats.npts
    packer.pack_uint(ndata)
    packer.pack_float(tr.stats.delta)
    packer.pack_float(max(tr.data))
    packer.pack_int(tr.stats.starttime.year)
    packer.pack_int(tr.stats.starttime.month)
    packer.pack_int(tr.stats.starttime.day)
    packer.pack_int(tr.stats.starttime.hour)
    packer.pack_int(tr.stats.starttime.minute)

    sec = tr.stats.starttime.second
    msec = tr.stats.starttime.microsecond
    starttime_second = float(str(sec) + '.' + str(msec))
    packer.pack_float(starttime_second)

    packer.pack_float(0)
    packer.pack_int(comsize)
    packer.pack_fstring(comsize, 'null'.encode('utf-8'))
    packer.pack_int(logsize)
    packer.pack_fstring(logsize, 'null'.encode('utf-8'))

    # # extras
    packer.pack_array(np.zeros(21).tolist(), packer.pack_float)

    # pack data using dtype from record info
    if dtype == 1:
        # float
        packer.pack_farray(ndata, tr.data, packer.pack_float)
    elif dtype == 6:
        # double
        packer.pack_farray(ndata, tr.data, packer.pack_double)
    else:
        # e.g. 3 (vector), 2 (complex), 4 (tensor)
        msg = 'Unsupported AH v1 record type %d'
        raise NotImplementedError(msg % (dtype))

    return packer


def _read_ah2(filename):
    """
    Reads an AH v2 waveform file and returns a Stream object.

    :type filename: str
    :param filename: AH v2 file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream with Traces specified by given file.
    """

    def _unpack_trace(data):
        ah_stats = AttribDict({
            'version': '2.0',
            'event': AttribDict(),
            'station': AttribDict(),
            'record': AttribDict(),
            'extras': []
        })

        # station info
        data.unpack_int()  # undocumented extra int?
        ah_stats.station.code = _unpack_string(data)
        data.unpack_int()  # here too?
        ah_stats.station.channel = _unpack_string(data)
        data.unpack_int()  # and again?
        ah_stats.station.type = _unpack_string(data)
        ah_stats.station.recorder = _unpack_string(data)
        ah_stats.station.sensor = _unpack_string(data)
        ah_stats.station.azimuth = data.unpack_float()  # degrees E from N
        ah_stats.station.dip = data.unpack_float()  # up = -90, down = +90
        ah_stats.station.latitude = data.unpack_double()
        ah_stats.station.longitude = data.unpack_double()
        ah_stats.station.elevation = data.unpack_float()
        ah_stats.station.gain = data.unpack_float()
        ah_stats.station.normalization = data.unpack_float()  # A0

        npoles = data.unpack_int()
        ah_stats.station.poles = []
        for _i in range(npoles):
            r = data.unpack_float()
            i = data.unpack_float()
            ah_stats.station.poles.append(complex(r, i))

        nzeros = data.unpack_int()
        ah_stats.station.zeros = []
        for _i in range(nzeros):
            r = data.unpack_float()
            i = data.unpack_float()
            ah_stats.station.zeros.append(complex(r, i))
        ah_stats.station.comment = _unpack_string(data)

        # event info
        ah_stats.event.latitude = data.unpack_double()
        ah_stats.event.longitude = data.unpack_double()
        ah_stats.event.depth = data.unpack_float()
        ot_year = data.unpack_int()
        ot_mon = data.unpack_int()
        ot_day = data.unpack_int()
        ot_hour = data.unpack_int()
        ot_min = data.unpack_int()
        ot_sec = data.unpack_float()
        try:
            ot = UTCDateTime(ot_year, ot_mon, ot_day, ot_hour, ot_min, ot_sec)
        except Exception:
            ot = None
        ah_stats.event.origin_time = ot
        data.unpack_int()  # and again?
        ah_stats.event.comment = _unpack_string(data)

        # record info
        ah_stats.record.type = dtype = data.unpack_int()  # data type
        ah_stats.record.ndata = ndata = data.unpack_uint()  # number of samples
        ah_stats.record.delta = data.unpack_float()  # sampling interval
        ah_stats.record.max_amplitude = data.unpack_float()
        at_year = data.unpack_int()
        at_mon = data.unpack_int()
        at_day = data.unpack_int()
        at_hour = data.unpack_int()
        at_min = data.unpack_int()
        at_sec = data.unpack_float()
        at = UTCDateTime(at_year, at_mon, at_day, at_hour, at_min, at_sec)
        ah_stats.record.start_time = at
        ah_stats.record.units = _unpack_string(data)
        ah_stats.record.inunits = _unpack_string(data)
        ah_stats.record.outunits = _unpack_string(data)
        data.unpack_int()  # and again?
        ah_stats.record.comment = _unpack_string(data)
        data.unpack_int()  # and again?
        ah_stats.record.log = _unpack_string(data)

        # user attributes
        nusrattr = data.unpack_int()
        ah_stats.usrattr = {}
        for _i in range(nusrattr):
            key = _unpack_string(data)
            value = _unpack_string(data)
            ah_stats.usrattr[key] = value

        # unpack data using dtype from record info
        if dtype == 1:
            # float
            temp = data.unpack_farray(ndata, data.unpack_float)
        elif dtype == 6:
            # double
            temp = data.unpack_farray(ndata, data.unpack_double)
        else:
            # e.g. 3 (vector), 2 (complex), 4 (tensor)
            msg = 'Unsupported AH v2 record type %d'
            raise NotImplementedError(msg % (dtype))

        tr = Trace(np.array(temp))
        tr.stats.ah = ah_stats
        tr.stats.delta = ah_stats.record.delta
        tr.stats.starttime = ah_stats.record.start_time
        tr.stats.station = ah_stats.station.code
        tr.stats.channel = ah_stats.station.channel
        return tr

    st = Stream()
    with open(filename, "rb") as fh:
        # loop as long we can read records
        while True:
            try:
                # read first 8 bytes with XDR library
                data = xdrlib.Unpacker(fh.read(8))
                # check magic version number
                magic = data.unpack_int()
            except EOFError:
                break
            if magic != 1100:
                raise Exception('Not a AH v2 file')
            try:
                # get record length
                length = data.unpack_uint()
                # read rest of record into XDR unpacker
                data = xdrlib.Unpacker(fh.read(length))
                tr = _unpack_trace(data)
                st.append(tr)
            except EOFError:
                break
        return st
