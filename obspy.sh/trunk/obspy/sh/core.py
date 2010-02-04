# -*- coding: utf-8 -*-
"""
SH bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from StringIO import StringIO
from obspy.core import Stream, Trace, UTCDateTime, Stats
from obspy.core.util import formatScientific
import numpy as np
import os


MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
          'OCT', 'NOV', 'DEC' ]

SH_IDX = {
    'LENGTH':'L001',
    'SIGN':'I011',
    'EVENTNO':'I012',
    'MARK':'I014',
    'DELTA':'R000',
    'CALIB':'R026',
    'DISTANCE':'R011',
    'AZIMUTH':'R012',
    'SLOWNESS':'R018',
    'INCI':'R013',
    'DEPTH':'R014',
    'MAGNITUDE':'R015',
    'LAT':'R016',
    'LON':'R017',
    'SIGNOISE':'R022',
    'PWDW':'R023',
    'DCVREG':'R024',
    'DCVINCI':'R025',
    'COMMENT':'S000',
    'STATION':'S001',
    'OPINFO':'S002',
    'FILTER':'S011',
    'QUALITY':'S012',
    'COMP':'C000',
    'CHAN1':'C001',
    'CHAN2':'C002',
    'BYTEORDER':'C003',
    'START':'S021',
    'P-ONSET':'S022',
    'S-ONSET':'S023',
    'ORIGIN':'S024'
}

INVERTED_SH_IDX = dict([(v, k) for (k, v) in SH_IDX.iteritems()])


def isASC(filename):
    """
    Checks whether a file is ASC or not. Returns True or False.

    Parameters
    ----------

    filename : string
        Name of the ASC file to be read.
    """
    # first six chars should contain 'DELTA:'
    try:
        temp = open(filename, 'rb').read(6)
    except:
        return False
    if temp != 'DELTA:':
        return False
    return True


def readASC(filename, headonly=False):
    """
    Reads a ASC file and returns an ObsPy Stream object.

    This function should NOT be called directly, it registers via the
    obspy :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        ASC file to be read.
    headonly : bool, optional
        If set to True, read only the head. This is most useful for
        scanning available data in huge (temporary) data sets.

    Returns
    -------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.

    Example
    -------
    >>> from obspy.core import read # doctest: +SKIP
    >>> st = read("seisan_file") # doctest: +SKIP
    """
    fh = open(filename, 'rt')
    # read file and split text into channels
    channels = []
    headers = {}
    data = StringIO()
    for line in fh.xreadlines():
        if line.isspace():
            # blank line
            # check if any data fetched yet
            if len(headers) == 0:
                continue
            # append current channel
            data.seek(0)
            channels.append((headers, data))
            # create new channel
            headers = {}
            data = StringIO()
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
    for headers, data in channels:
        # create Stats
        header = Stats()
        header['sh'] = {}
        channel = [' ', ' ', ' ']
        # generate headers
        for key, value in headers.iteritems():
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
                header['starttime'] = toUTCDateTime(value)
            else:
                # everything else gets stored into sh entry
                header['sh'][key] = value
        # set channel code
        header['channel'] = ''.join(channel)
        if headonly:
            # skip data
            stream.append(Trace(header=header))
        else:
            # read data
            data = np.loadtxt(data, dtype='float32')
            stream.append(Trace(data=data, header=header))
    return stream


def writeASC(stream, filename):
    """
    Writes a ASC file from given ObsPy Stream object.

    This function should NOT be called directly, it registers via the
    obspy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
    Stream object, call this instead.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    filename : string
        Name of ASC file to be written.
    """
    fh = open(filename, 'wb')
    for trace in stream:
        # write headers
        fh.write("DELTA: %s\n" % formatScientific("%-.6e" % trace.stats.delta))
        fh.write("LENGTH: %d\n" % trace.stats.npts)
        # additional headers
        for key, value in trace.stats.get('sh', []).iteritems():
            fh.write("%s: %s\n" % (key, value))
        # special format for start time
        dt = trace.stats.starttime
        fh.write("START: %s\n" % fromUTCDateTime(dt))
        # component must be split
        if len(trace.stats.channel) >= 2:
            fh.write("COMP: %c\n" % trace.stats.channel[2])
        if len(trace.stats.channel) >= 0:
            fh.write("CHAN1: %c\n" % trace.stats.channel[0])
        if len(trace.stats.channel) >= 1:
            fh.write("CHAN2: %c\n" % trace.stats.channel[1])
        fh.write("STATION: %s\n" % trace.stats.station)
        fh.write("CALIB: %s\n" % formatScientific("%-.6e" % trace.stats.calib))
        # write data in four columns
        delimiter = ['', '', '', '\n'] * ((trace.stats.npts / 4) + 1)
        delimiter = delimiter[0:trace.stats.npts - 1]
        delimiter.append('\n')
        for (sample, delim) in zip(trace.data, delimiter):
            fh.write("%s %s" % (formatScientific("%-.6e" % sample), delim))
        fh.write("\n")
    fh.close()


def isQ(filename):
    """
    Checks whether a file is Q or not. Returns True or False.

    Parameters
    ----------

    filename : string
        Name of the Q file to be read.
    """
    # file must start with magic number 43981
    try:
        temp = open(filename, 'rb').read(5)
    except:
        return False
    if temp != '43981':
        return False
    return True


def readQ(filename, headonly=False, data_directory=None, byteorder='='):
    """
    Reads a Q file and returns an ObsPy Stream object.

    Q files consists of two files per data set:

     * a ASCII header file with file extension `QHD` and the
     * binary data file with file extension `QBN`.

    The read method only accepts header files for the ``filename`` parameter.
    ObsPy assumes that the corresponding data file is within the same directory
    if the ``data_directory`` parameter is not set. Otherwise it will search
    in the given ``data_directory`` for a file with the `QBN` file extension.
    This function should NOT be called directly, it registers via the
    obspy :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        Q header file to be read. Must have a `QHD` file extension.
    headonly : bool, optional
        If set to True, read only the head. This is most useful for
        scanning available data in huge (temporary) data sets.
    data_directory : string, optional
        Data directory where the corresponding QBN file can be found.
    byteorder : [ '<' | '>' | '=' ], optional
        Enforce byte order for data file. This is important for Q files written
        in older versions of Seismic Handler, which don't explicit state the
        BYTEORDER flag within the header file. Defaults to '=' (local byte
        order).

    Returns
    -------
    :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.

    Example
    -------
    >>> from obspy.core import read # doctest: +SKIP
    >>> st = read("Q_file") # doctest: +SKIP
    """
    if not headonly:
        if not data_directory:
            data_file = os.path.splitext(filename)[0] + '.QBN'
        else:
            data_file = os.path.basename(os.path.splitext(filename)[0])
            data_file = os.path.join(data_directory, data_file + '.QBN')
        if not os.path.isfile(data_file):
            msg = "Can't find corresponding QBN file at %s."
            raise IOError(msg % data_file)
        fh_data = open(data_file, 'rb')
    # loop through read header file
    fh = open(filename, 'rt')
    line = fh.readline()
    cmtlines = int(line[5:7]) - 1
    # comment lines
    comments = []
    for _i in xrange(0, cmtlines):
        comments += [fh.readline()]
    # trace lines
    traces = {}
    for line in fh.xreadlines():
        id = int(line[0:2])
        traces.setdefault(id, '')
        traces[id] += line[3:].strip()
    # create stream object
    stream = Stream()
    for id in sorted(traces.keys()):
        # fetch headers
        header = {}
        header['sh'] = {}
        channel = [' ', ' ', ' ']
        npts = 0
        for item in traces[id].split('~'):
            key = item.strip()[0:4]
            value = item.strip()[5:].strip()
            if key == 'L001':
                npts = header['npts'] = int(value)
            elif key == 'R000':
                header['delta'] = float(value)
            elif key == 'R026':
                header['calib'] = float(value)
            elif key == 'S001':
                header['station'] = value
            elif key == 'C000':
                channel[2] = value[0]
            elif key == 'C001':
                channel[0] = value[0]
            elif key == 'C002':
                channel[1] = value[0]
            elif key == 'C003':
                if value == '<' or value == '>':
                    byteorder = header['sh']['BYTEORDER'] = value
            elif key == 'S021':
                # 01-JAN-2009_01:01:01.0
                # 1-OCT-2009_12:46:01.000
                header['starttime'] = toUTCDateTime(value)
            elif key:
                key = INVERTED_SH_IDX.get(key, key)
                header['sh'][key] = value
        # set channel code
        header['channel'] = ''.join(channel)
        if headonly:
            # skip data
            stream.append(Trace(header=header))
        else:
            if not npts:
                continue
            # read data
            data = fh_data.read(npts * 4)
            dtype = byteorder + 'f4'
            data = np.fromstring(data, dtype=dtype)
            # convert to system byte order
            data = np.require(data, '=f4')
            stream.append(Trace(data=data, header=header))
    if not headonly:
        fh_data.close()
    return stream


def writeQ(stream, filename, data_directory=None, byteorder='='):
    """
    Writes a Q file from given ObsPy Stream object.

    This function should NOT be called directly, it registers via the
    obspy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
    Stream object, call this instead.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    filename : string
        Name of Q file to be written.
    data_directory : string, optional
        Data directory where the corresponding QBN will be written.
    byteorder : [ '<' | '>' | '=' ], optional
        Enforce byte order for data file. Defaults to '=' (local byte order).
    """
    if filename.endswith('.QHD'):
        filename = os.path.splitext(filename)[0]
    if data_directory:
        temp = os.path.basename(filename)
        filename_data = os.path.join(data_directory, temp + '.QBN')
    else:
        filename_data = filename
    fh = open(filename + '.QHD', 'wb')
    fh_data = open(filename_data + '.QBN', 'wb')

    # build up header strings
    headers = []
    maxnol = 0
    for trace in stream:
        temp = "L001:%d~ R000:%f~ R026:%f~ " % (trace.stats.npts,
                                                trace.stats.delta,
                                                trace.stats.calib)
        if trace.stats.station:
            temp += "S001:%s~ " % trace.stats.station
        # component must be split
        if len(trace.stats.channel) >= 2:
            temp += "C000:%c~ " % trace.stats.channel[2]
        if len(trace.stats.channel) >= 0:
            temp += "C001:%c~ " % trace.stats.channel[0]
        if len(trace.stats.channel) >= 1:
            temp += "C002:%c~ " % trace.stats.channel[1]
        # special format for start time
        dt = trace.stats.starttime
        temp += "S021: %s~ " % fromUTCDateTime(dt)
        for key, value in trace.stats.get('sh', []).iteritems():
            # skip unknown keys
            if not key or key not in SH_IDX.keys():
                continue
            temp += "%s:%s~ " % (SH_IDX[key], value)
        headers.append(temp)
        # get maximal number of trclines
        nol = len(temp) / 74 + 1
        if nol > maxnol:
            maxnol = nol
    # first line: magic number, cmtlines, trclines
    # XXX: comment lines are ignored
    fh.write("43981 1 %d\n" % maxnol)

    for i, trace in enumerate(stream):
        # write headers
        temp = [headers[i][j:j + 74] for j in range(0, len(headers[i]), 74)]
        for j in xrange(0, maxnol):
            try:
                fh.write("%02d|%s\n" % (i + 1, temp[j]))
            except:
                fh.write("%02d|\n" % (i + 1))
        # write data in given byte order
        dtype = byteorder + 'f4'
        data = np.require(trace.data, dtype=dtype)
        data.tofile(fh_data)
    fh.close()
    fh_data.close()

def toUTCDateTime(value):
    date, time = value.split('_')
    day, month, year = date.split('-')
    hour, mins, secs = time.split(':')
    day = int(day)
    month = MONTHS.index(month.upper()) + 1
    year = int(year)
    hour = int(hour)
    mins = int(mins)
    secs = float(secs)

    return UTCDateTime(year, month, day, hour, mins) + secs

def fromUTCDateTime(dt):
    pattern = "%2d-%3s-%4d_%02d:%02d:%02d.%03d"

    return pattern % (dt.day, MONTHS[dt.month - 1], dt.year, dt.hour,
                        dt.minute, dt.second, dt.microsecond / 1000)

