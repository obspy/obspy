# -*- coding: utf-8 -*-
"""
SH bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
from pathlib import Path
import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.compatibility import from_buffer


MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
          'OCT', 'NOV', 'DEC']

MONTHS_DE = ['JAN', 'FEB', 'MAR', 'APR', 'MAI', 'JUN', 'JUL', 'AUG', 'SEP',
             'OKT', 'NOV', 'DEZ']

SH_IDX = {
    'LENGTH': 'L001',
    'SIGN': 'I011',
    'EVENTNO': 'I012',
    'MARK': 'I014',
    'DELTA': 'R000',
    'CALIB': 'R026',
    'DISTANCE': 'R011',
    'AZIMUTH': 'R012',
    'SLOWNESS': 'R018',
    'INCI': 'R013',
    'DEPTH': 'R014',
    'MAGNITUDE': 'R015',
    'LAT': 'R016',
    'LON': 'R017',
    'SIGNOISE': 'R022',
    'PWDW': 'R023',
    'DCVREG': 'R024',
    'DCVINCI': 'R025',
    'COMMENT': 'S000',
    'STATION': 'S001',
    'OPINFO': 'S002',
    'FILTER': 'S011',
    'QUAL': 'S012',
    'COMP': 'C000',
    'CHAN1': 'C001',
    'CHAN2': 'C002',
    'BYTEORDER': 'C003',
    'START': 'S021',
    'P-ONSET': 'S022',
    'S-ONSET': 'S023',
    'ORIGIN': 'S024'
}

STANDARD_ASC_HEADERS = ['START', 'COMP', 'CHAN1', 'CHAN2', 'STATION', 'CALIB']

SH_KEYS_INT = [k for (k, v) in SH_IDX.items() if v.startswith('I')]
SH_KEYS_FLOAT = [k for (k, v) in SH_IDX.items() if v.startswith('R')]
INVERTED_SH_IDX = {v: k for k, v in SH_IDX.items()}


def _is_asc(filename):
    """
    Checks whether a file is a Seismic Handler ASCII file or not.

    :type filename: str
    :param filename: Name of the ASCII file to be checked.
    :rtype: bool
    :return: ``True`` if a Seismic Handler ASCII file.

    .. rubric:: Example

    >>> _is_asc("/path/to/QFILE-TEST-ASC.ASC")  #doctest: +SKIP
    True
    """
    # first six chars should contain 'DELTA:'
    try:
        with open(filename, 'rb') as f:
            temp = f.read(6)
    except Exception:
        return False
    if temp != b'DELTA:':
        return False
    return True


def _read_asc(filename, headonly=False, skip=0, delta=None, length=None,
              **kwargs):  # @UnusedVariable
    """
    Reads a Seismic Handler ASCII file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: ASCII file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type skip: int, optional
    :param skip: Number of lines to be skipped from top of file. If defined
        only one trace is read from file.
    :type delta: float, optional
    :param delta: If ``skip`` is used, ``delta`` defines sample offset in
        seconds.
    :type length: int, optional
    :param length: If ``skip`` is used, ``length`` defines the number of values
        to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/QFILE-TEST-ASC.ASC")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    .TEST..BHN | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
    .TEST..BHE | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
    .WET..HHZ  | 2010-01-01T01:01:05.999000Z - ... | 100.0 Hz, 4001 samples
    """
    fh = open(filename, 'rt')
    # read file and split text into channels
    channels = []
    headers = {}
    data = io.StringIO()
    for line in fh.readlines()[skip:]:
        if line.isspace():
            # blank line
            # check if any data fetched yet
            if len(headers) == 0 and data.tell() == 0:
                continue
            # append current channel
            data.seek(0)
            channels.append((headers, data))
            # create new channel
            headers = {}
            data = io.StringIO()
            if skip:
                # if skip is set only one trace is read, everything else makes
                # no sense.
                break
            continue
        elif line[0].isalpha():
            # header entry
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            headers[key] = value
        elif not headonly:
            # data entry - may be written in multiple columns
            data.write(line.strip() + ' ')
    fh.close()
    # create ObsPy stream object
    stream = Stream()
    # custom header
    custom_header = {}
    if delta:
        custom_header["delta"] = delta
    if length:
        custom_header["npts"] = length

    for headers, data in channels:
        # create Stats
        header = Stats(custom_header)
        header['sh'] = {}
        channel = [' ', ' ', ' ']
        # generate headers
        for key, value in headers.items():
            if key == 'DELTA':
                header['delta'] = float(value)
            elif key == 'LENGTH':
                header['npts'] = int(value)
            elif key == 'CALIB':
                header['calib'] = float(value)
            elif key == 'STATION':
                header['station'] = value
            elif key == 'COMP':
                channel[2] = value[0]
            elif key == 'CHAN1':
                channel[0] = value[0]
            elif key == 'CHAN2':
                channel[1] = value[0]
            elif key == 'START':
                # 01-JAN-2009_01:01:01.0
                # 1-OCT-2009_12:46:01.000
                header['starttime'] = to_utcdatetime(value)
            else:
                # everything else gets stored into sh entry
                if key in SH_KEYS_INT:
                    header['sh'][key] = int(value)
                elif key in SH_KEYS_FLOAT:
                    header['sh'][key] = float(value)
                else:
                    header['sh'][key] = value
        # set channel code
        header['channel'] = ''.join(channel)
        if headonly:
            # skip data
            stream.append(Trace(header=header))
        else:
            # read data
            data = np.loadtxt(data, dtype=np.float32, ndmin=1)

            # cut data if requested
            if skip and length:
                data = data[:length]

            # use correct value in any case
            header["npts"] = len(data)

            stream.append(Trace(data=data, header=header))
    return stream


def _write_asc(stream, filename, included_headers=None, npl=4,
               custom_format="%-.6e", append=False,
               **kwargs):  # @UnusedVariable
    """
    Writes a Seismic Handler ASCII file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of the ASCII file to write.
    :type npl: int, optional
    :param npl: Number of data columns in file, default to four.
    :type included_headers: list or None, optional
    :param included_headers: If set to a list, only these header entries will
        be written to file. DELTA and LENGTH are written in any case. If it's
        set to None, a basic set will be included.
    :type custom_format: str, optional
    :param custom_format: Parameter for number formatting of samples, defaults
        to "%-.6e".
    :type append: bool, optional
    :param append: If filename exists append all data to file, default False.
    """
    if included_headers is None:
        included_headers = STANDARD_ASC_HEADERS

    sio = io.StringIO()
    for trace in stream:
        # write headers
        sio.write("DELTA: %-.6e\n" % (trace.stats.delta))
        sio.write("LENGTH: %d\n" % trace.stats.npts)
        # additional headers
        for key, value in trace.stats.get('sh', {}).items():
            if included_headers and key not in included_headers:
                continue
            sio.write("%s: %s\n" % (key, value))
        # special format for start time
        if "START" in included_headers:
            dt = trace.stats.starttime
            sio.write("START: %s\n" % from_utcdatetime(dt))
        # component must be split
        if len(trace.stats.channel) > 2 and "COMP" in included_headers:
            sio.write("COMP: %c\n" % trace.stats.channel[2])
        if len(trace.stats.channel) > 0 and "CHAN1" in included_headers:
            sio.write("CHAN1: %c\n" % trace.stats.channel[0])
        if len(trace.stats.channel) > 1 and "CHAN2" in included_headers:
            sio.write("CHAN2: %c\n" % trace.stats.channel[1])
        if "STATION" in included_headers:
            sio.write("STATION: %s\n" % trace.stats.station)
        if "CALIB" in included_headers:
            sio.write("CALIB: %-.6e\n" % (trace.stats.calib))
        # write data in npl columns
        mask = ([''] * (npl - 1)) + ['\n']
        delimiter = mask * ((trace.stats.npts // npl) + 1)
        delimiter = delimiter[:trace.stats.npts - 1]
        delimiter.append('\n')
        for (sample, delim) in zip(trace.data, delimiter):
            value = custom_format % (sample)
            sio.write("%s %s" % (value, delim))
        sio.write("\n")
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    with open(filename, mode=mode) as fh:
        sio.seek(0)
        fh.write(sio.read().encode('ascii', 'strict'))


def _is_q(filename):
    """
    Checks whether a file is a Seismic Handler Q file or not.

    :type filename: str
    :param filename: Name of the Q file to be checked.
    :rtype: bool
    :return: ``True`` if a Seismic Handler Q file.

    .. rubric:: Example

    >>> _is_q("/path/to/QFILE-TEST.QHD")  #doctest: +SKIP
    True
    """
    # file must start with magic number 43981
    try:
        with open(filename, 'rb') as f:
            temp = f.read(5)
    except Exception:
        return False
    if temp != b'43981':
        return False
    return True


def _read_q(filename, headonly=False, data_directory=None, byteorder='=',
            **kwargs):  # @UnusedVariable
    """
    Reads a Seismic Handler Q file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Q header file to be read. Must have a `QHD` file
        extension.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type data_directory: str, optional
    :param data_directory: Data directory where the corresponding QBN file can
        be found.
    :type byteorder: str, optional
    :param byteorder: Enforce byte order for data file. This is important for
        Q files written in older versions of Seismic Handler, which don't
        explicit state the `BYTEORDER` flag within the header file. Can be
        little endian (``'<'``), big endian (``'>'``), or native byte order
        (``'='``). Defaults to ``'='``.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    Q files consists of two files per data set:

    * a ASCII header file with file extension `QHD` and the
    * binary data file with file extension `QBN`.

    The read method only accepts header files for the ``filename`` parameter.
    ObsPy assumes that the corresponding data file is within the same directory
    if the ``data_directory`` parameter is not set. Otherwise it will search
    in the given ``data_directory`` for a file with the `QBN` file extension.
    This function should NOT be called directly, it registers via the
    ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/QFILE-TEST.QHD")
    >>> st    #doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    3 Trace(s) in Stream:
    .TEST..BHN | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
    .TEST..BHE | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
    .WET..HHZ  | 2010-01-01T01:01:05.999000Z - ... | 100.0 Hz, 4001 samples
    """
    if not headonly:
        if not data_directory:
            path = Path(filename)
            data_file = Path(path.parent) / (path.stem+".QBN")

        else:
            path = Path(filename)
            data_file = Path(data_directory) / Path(filename).stem+".QBN"
        if not Path(data_file).is_file():
            msg = "Can't find corresponding QBN file at %s."
            raise IOError(msg % data_file)
        fh_data = open(data_file, 'rb')
    # loop through read header file
    with open(filename, 'rt') as fh:
        lines = fh.read().splitlines()
    # number of comment lines
    cmtlines = int(lines[0][5:7])
    # trace lines
    traces = {}
    i = -1
    id = ''
    for line in lines[cmtlines:]:
        cid = int(line[0:2])
        if cid != id:
            id = cid
            i += 1
        traces.setdefault(i, '')
        traces[i] += line[3:]
    # create stream object
    stream = Stream()
    for id in sorted(traces.keys()):
        # fetch headers
        header = {}
        header['sh'] = {
            "FROMQ": True,
            "FILE": Path(filename).stem,
        }
        channel = ['', '', '']
        npts = 0
        for item in traces[id].split('~'):
            key = item.lstrip()[0:4]
            value = item.lstrip()[5:]
            if key == 'L001':
                npts = header['npts'] = int(value)
            elif key == 'L000':
                continue
            elif key == 'R000':
                header['delta'] = float(value)
            elif key == 'R026':
                header['calib'] = float(value)
            elif key == 'S001':
                header['station'] = value
            elif key == 'C000' and value:
                channel[2] = value[0]
            elif key == 'C001' and value:
                channel[0] = value[0]
            elif key == 'C002' and value:
                channel[1] = value[0]
            elif key == 'C003':
                if value == '<' or value == '>':
                    byteorder = header['sh']['BYTEORDER'] = value
            elif key == 'S021':
                # 01-JAN-2009_01:01:01.0
                # 1-OCT-2009_12:46:01.000
                header['starttime'] = to_utcdatetime(value)
            elif key == 'S022':
                header['sh']['P-ONSET'] = to_utcdatetime(value)
            elif key == 'S023':
                header['sh']['S-ONSET'] = to_utcdatetime(value)
            elif key == 'S024':
                header['sh']['ORIGIN'] = to_utcdatetime(value)
            elif key:
                key = INVERTED_SH_IDX.get(key, key)
                if key in SH_KEYS_INT:
                    header['sh'][key] = int(value)
                elif key in SH_KEYS_FLOAT:
                    header['sh'][key] = float(value)
                else:
                    header['sh'][key] = value
        # set channel code
        header['channel'] = ''.join(channel)
        # remember record number
        header['sh']['RECNO'] = len(stream) + 1
        if headonly:
            # skip data
            stream.append(Trace(header=header))
        else:
            if not npts:
                stream.append(Trace(header=header))
                continue
            # read data
            data = fh_data.read(npts * 4)
            dtype = byteorder + 'f4'
            data = from_buffer(data, dtype=dtype)
            # convert to system byte order
            data = np.require(data, '=f4')
            stream.append(Trace(data=data, header=header))
    if not headonly:
        fh_data.close()
    return stream


def _write_q(stream, filename, data_directory=None, byteorder='=',
             append=False, **kwargs):  # @UnusedVariable
    """
    Writes a Seismic Handler Q file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of the Q file to write.
    :type data_directory: str, optional
    :param data_directory: Data directory where the corresponding QBN will be
        written.
    :type byteorder: str, optional
    :param byteorder: Enforce byte order for data file. Can be little endian
        (``'<'``), big endian (``'>'``), or native byte order (``'='``).
        Defaults to ``'='``.
    :type append: bool, optional
    :param append: If filename exists append all data to file, default False.
    """
    if filename.endswith('.QHD') or filename.endswith('.QBN'):
        path = Path(filename)
        filename = str(Path(path.parent)/path.stem)
    if data_directory:
        filename_data = Path(data_directory) / Path(filename).name
    else:
        filename_data = filename
    filename_header = filename + '.QHD'

    # if the header file exists its assumed that the data is also there
    if Path(filename_header).exists() and append:
        try:
            trcs = _read_q(filename_header, headonly=True)
        except Exception:
            raise Exception("Target filename '%s' not readable!" % filename)
        mode = 'ab'
        count_offset = len(trcs)
        cur_npts_offset = sum([trcs[i].stats.npts for i in range(len(trcs))])
    else:
        append = False
        mode = 'wb'
        count_offset = 0
        cur_npts_offset = 0

    fh = open(filename_header, mode)
    fh_data = open(filename_data + '.QBN', mode)

    # build up header strings
    headers = []
    minnol = 4
    cur_npts = 0 + cur_npts_offset
    for trace in stream:
        temp = "L000:%d~ " % cur_npts
        cur_npts += trace.stats.npts
        temp += "L001:%d~ R000:%f~ R026:%f~ " % (trace.stats.npts,
                                                 trace.stats.delta,
                                                 trace.stats.calib)
        if trace.stats.station:
            temp += "S001:%s~ " % trace.stats.station
        # component must be split
        if len(trace.stats.channel) > 2:
            temp += "C000:%c~ " % trace.stats.channel[2]
        if len(trace.stats.channel) > 0:
            temp += "C001:%c~ " % trace.stats.channel[0]
        if len(trace.stats.channel) > 1:
            temp += "C002:%c~ " % trace.stats.channel[1]
        # special format for start time
        dt = trace.stats.starttime
        temp += "S021:%s~ " % from_utcdatetime(dt)
        for key, value in trace.stats.get('sh', {}).items():
            # skip unknown keys
            if not key or key not in SH_IDX.keys():
                continue
            # convert UTCDateTimes into strings
            if isinstance(value, UTCDateTime):
                value = from_utcdatetime(value)
            temp += "%s:%s~ " % (SH_IDX[key], value)
        headers.append(temp)
        # get maximal number of trclines
        nol = len(temp) // 74 + 1
        if nol > minnol:
            minnol = nol
    # first line: magic number, cmtlines, trclines
    # XXX: comment lines are ignored
    if not append:
        line = "43981 1 %d\n" % minnol
        fh.write(line.encode('ascii', 'strict'))

    for i, trace in enumerate(stream):
        # write headers
        temp = [headers[i][j:j + 74] for j in range(0, len(headers[i]), 74)]
        for j in range(0, minnol):
            try:
                line = "%02d|%s\n" % ((i + 1 + count_offset) % 100, temp[j])
                fh.write(line.encode('ascii', 'strict'))
            except Exception:
                line = "%02d|\n" % ((i + 1 + count_offset) % 100)
                fh.write(line.encode('ascii', 'strict'))
        # write data in given byte order
        dtype = byteorder + 'f4'
        data = np.require(trace.data, dtype=dtype)
        fh_data.write(data.data)
    fh.close()
    fh_data.close()


def to_utcdatetime(value):
    """
    Converts time string used within Seismic Handler into a UTCDateTime.

    :type value: str
    :param value: A Date time string.
    :return: Converted :class:`~obspy.core.utcdatetime.UTCDateTime` object.

    .. rubric:: Example

    >>> to_utcdatetime(' 2-JAN-2008_03:04:05.123')
    UTCDateTime(2008, 1, 2, 3, 4, 5, 123000)
    >>> to_utcdatetime('2-JAN-2008')
    UTCDateTime(2008, 1, 2, 0, 0)
    >>> to_utcdatetime('2-JAN-08')
    UTCDateTime(2008, 1, 2, 0, 0)
    >>> to_utcdatetime('2-JAN-99')
    UTCDateTime(1999, 1, 2, 0, 0)
    >>> to_utcdatetime('2-JAN-2008_1')
    UTCDateTime(2008, 1, 2, 1, 0)
    >>> to_utcdatetime('2-JAN-2008_1:1')
    UTCDateTime(2008, 1, 2, 1, 1)
    """
    try:
        date, time = value.split('_')
    except ValueError:
        date = value
        time = "0:0:0"
    day, month, year = date.split('-')
    time = time.split(':')
    try:
        hour, mins, secs = time
    except ValueError:
        hour = time[0]
        mins = "0"
        secs = "0"
        if len(time) == 2:
            mins = time[1]
    day = int(day)
    try:
        month = MONTHS.index(month.upper()) + 1
    except ValueError:
        month = MONTHS_DE.index(month.upper()) + 1
    if len(year) == 2:
        if int(year) < 70:
            year = "20" + year
        else:
            year = "19" + year
    year = int(year)
    hour = int(hour)
    mins = int(mins)
    secs = float(secs)
    return UTCDateTime(year, month, day, hour, mins) + secs


def from_utcdatetime(dt):
    """
    Converts UTCDateTime object into a time string used within Seismic Handler.

    :type dt: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param dt: A UTCDateTime object.
    :return: Converted date time string usable by Seismic Handler.

    .. rubric:: Example

    >>> from obspy import UTCDateTime
    >>> dt = UTCDateTime(2008, 1, 2, 3, 4, 5, 123456)
    >>> print(from_utcdatetime(dt))
     2-JAN-2008_03:04:05.123
    """
    pattern = "%2d-%3s-%4d_%02d:%02d:%02d.%03d"

    return pattern % (dt.day, MONTHS[dt.month - 1], dt.year, dt.hour,
                      dt.minute, dt.second, dt.microsecond / 1000)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
