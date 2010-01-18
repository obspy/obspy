# -*- coding: utf-8 -*-

from StringIO import StringIO
from obspy.core import Stream, Trace, UTCDateTime, Stats
import numpy as np
from obspy.core.util import formatScientific


MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
          'OCT', 'NOV', 'DEC' ]


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
                date, time = value.split('_')
                day, month, year = date.split('-')
                hour, mins, secs = time.split(':')
                day = int(day)
                month = MONTHS.index(month) + 1
                year = int(year)
                hour = int(hour)
                mins = int(mins)
                secs = float(secs)
                header['starttime'] = UTCDateTime(year, month, day, hour,
                                                  mins) + secs
            else:
                # everything else gets stored into sh entry 
                header['sh'][key] = value
        # set channel code
        header['channel'] = ''.join(channel)
        if headonly:
            # skip data
            stream.append(Trace(header=header))
        else:
            # write data
            data = np.loadtxt(data, dtype='float64')
            stream.append(Trace(data=data, header=header))
    return stream


def writeASC(stream, filename):
    """
    Writes a ASC file from given ObsPy Stream object.

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
        for key, value in trace.stats.sh.iteritems():
            fh.write("%s: %s\n" % (key, value))
        # special format for start time
        dt = trace.stats.starttime
        pattern = "START: %2d-%3s-%4d_%02d:%02d:%02d.%03d\n"
        fh.write(pattern % (dt.day, MONTHS[dt.month - 1], dt.year, dt.hour,
                            dt.minute, dt.second, dt.microsecond / 1000))
        # component must be splitted
        if len(trace.stats.channel) >= 2:
            fh.write("COMP: %c\n" % trace.stats.channel[2])
        if len(trace.stats.channel) >= 0:
            fh.write("CHAN1: %c\n" % trace.stats.channel[0])
        if len(trace.stats.channel) >= 1:
            fh.write("CHAN2: %c\n" % trace.stats.channel[1])
        fh.write("STATION: %s\n" % trace.stats.station)
        fh.write("CALIB: %s\n" % formatScientific("%-.6e" % trace.stats.calib))
        # write data in four coloums
        delimiter = ['', '', '', '\n'] * ((trace.stats.npts / 4) + 1)
        delimiter = delimiter[0:trace.stats.npts - 1]
        delimiter.append('\n')
        for (sample, delim) in zip(trace.data, delimiter):
            fh.write("%s %s" % (formatScientific("%-.6e" % sample), delim))
        fh.write("\n")
    fh.close()
