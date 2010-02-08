# -*- coding: utf-8 -*-
"""
GSE2 bindings to ObsPy core module.
"""

from obspy.core import Trace, UTCDateTime, Stream
from obspy.gse2 import libgse2
import numpy as np


def isGSE2(filename):
    """
    Checks whether a file is GSE2 or not. Returns True or False.

    Parameters
    ----------
    filename : string
        GSE2 file to be checked.
    """
    # Open file.
    try:
        f = open(filename)
        data = f.read(4)
    except:
        return False
    f.close()
    if data == 'WID2':
        return True
    return False


convert_dict = {
    'station': 'station',
    'samp_rate':'sampling_rate',
    'n_samps': 'npts',
    'channel': 'channel',
    'calib': 'calib',
}

gse2_extra = [
    'instype',
    'datatype',
    'vang',
    'hang',
    'auxid',
    'calper',
]


def readGSE2(filename, headonly=False, verify_chksum=True, **kwargs):
    """
    Reads a GSE2 file and returns a Stream object.

    GSE2 files containing multiple WID2 entries/traces are supported.
    This function should NOT be called directly, it registers via the
    ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        GSE2 file to be read.
    headonly : boolean, optional
        If True read only head of GSE2 file.
    verify_chksum : boolean, optional
        If True verify Checksum and raise Exception if it is not correct.

    Returns
    -------
    :class:`~obspy.core.stream.Stream`
        Stream object containing header and data.

    Example
    -------
    >>> from obspy.core import read # doctest: +SKIP
    >>> st = read("loc_RJOB20050831023349.z") # doctest: +SKIP
    """
    traces = []
    # read GSE2 file
    f = open(filename, 'rb')
    for _k in xrange(10000): # avoid endless loop
        pos = f.tell()
        widi = f.readline()[0:4]
        if widi == '': # end of file
            break
        elif widi != 'WID2':
            continue
        else: # valid gse2 part
            f.seek(pos)
            if headonly:
                header = libgse2.readHead(f)
            else:
                header, data = libgse2.read(f, verify_chksum=verify_chksum)
            # assign all header entries to a new dictionary compatible with an 
            # Obspy Trace object.
            new_header = {}
            for i, j in convert_dict.iteritems():
                value = header[i]
                if isinstance(value, str):
                    value = value.strip()
                new_header[j] = value
            # assign gse specific header entries
            new_header['gse2'] = {}
            for i in gse2_extra:
                new_header['gse2'][i] = header[i]
            # Calculate start time.
            new_header['starttime'] = UTCDateTime(
                header['d_year'], header['d_mon'], header['d_day'],
                header['t_hour'], header['t_min'], 0) + header['t_sec']
            if headonly:
                traces.append(Trace(header=new_header))
            else:
                traces.append(Trace(header=new_header, data=data))
    f.close()
    return Stream(traces=traces)


def writeGSE2(stream, filename, inplace=False, **kwargs):
    """
    Write GSE2 file from a Stream object.

    This function should NOT be called directly, it registers via the
    obspy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
    Stream object, call this instead.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        The ObsPy Stream object to write.
    filename : string
        Name of file to write.
    inplace : boolean, optional
        If True, do compression not on a copy of the data but on the data
        itself - note this will change the data values and make them therefore
        unusable!
    """
    #
    # Translate the common (renamed) entries
    f = open(filename, 'wb')
    for trace in stream:
        header = {}
        for _j, _k in convert_dict.iteritems():
            header[_j] = trace.stats[_k]
        for _j in gse2_extra:
            try:
                header[_j] = trace.stats.gse2[_j]
            except:
                pass
        # year, month, day, hour, min, sec
        try:
            (header['d_year'],
             header['d_mon'],
             header['d_day'],
             header['t_hour'],
             header['t_min'],
             header['t_sec']) = trace.stats.starttime.timetuple()[0:6]
            header['t_sec'] += trace.stats.starttime.microsecond / 1.0e6
        except:
            raise
        dtype = np.dtype('int')
        if trace.data.dtype.name == dtype.name:
            trace.data = np.require(trace.data, dtype, ['C_CONTIGUOUS'])
        else:
            msg = "GSE2 data must be of type %s, but are of type %s" % \
                (dtype.name, trace.data.dtype)
            raise Exception(msg)
        libgse2.write(header, trace.data, f, inplace)
    f.close()
