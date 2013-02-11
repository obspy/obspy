# -*- coding: utf-8 -*-
"""
GSE2/GSE1 bindings to ObsPy core module.
"""

from obspy import Trace, UTCDateTime, Stream
from obspy.gse2 import libgse2, libgse1
import numpy as np


def isGSE2(filename):
    """
    Checks whether a file is GSE2 or not.

    :type filename: string
    :param filename: GSE2 file to be checked.
    :rtype: bool
    :return: ``True`` if a GSE2 file.
    """
    # Open file.
    try:
        f = open(filename)
        libgse2.isGse2(f)
        f.close()
    except:
        return False
    return True


convert_dict = {
    'station': 'station',
    'samp_rate': 'sampling_rate',
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


def readGSE2(filename, headonly=False, verify_chksum=True,
             **kwargs):  # @UnusedVariable
    """
    Reads a GSE2 file and returns a Stream object.

    GSE2 files containing multiple WID2 entries/traces are supported.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: GSE2 file to be read.
    :type headonly: boolean, optional
    :param headonly: If True read only head of GSE2 file.
    :type verify_chksum: boolean, optional
    :param verify_chksum: If True verify Checksum and raise Exception if
        it is not correct.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/loc_RJOB20050831023349.z")
    """
    traces = []
    # read GSE2 file
    f = open(filename, 'rb')
    for _k in xrange(10000):  # avoid endless loop
        pos = f.tell()
        widi = f.readline()[0:4]
        if widi == '':  # end of file
            break
        elif widi != 'WID2':
            continue
        else:  # valid gse2 part
            f.seek(pos)
            if headonly:
                header = libgse2.readHead(f)
            else:
                header, data = libgse2.read(f, verify_chksum=verify_chksum)
            # assign all header entries to a new dictionary compatible with an
            # ObsPy Trace object.
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


def writeGSE2(stream, filename, inplace=False, **kwargs):  # @UnusedVariable
    """
    Write GSE2 file from a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: string
    :param filename: Name of file to write.
    :type inplace: boolean, optional
    :param inplace: If True, do compression not on a copy of the data but
        on the data itself - note this will change the data values and make
        them therefore unusable!

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write('filename.gse', format='GSE2') #doctest: +SKIP
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
        (header['d_year'],
            header['d_mon'],
            header['d_day'],
            header['t_hour'],
            header['t_min'],
            header['t_sec']) = trace.stats.starttime.timetuple()[0:6]
        header['t_sec'] += trace.stats.starttime.microsecond / 1.0e6
        dtype = np.dtype('int32')
        if trace.data.dtype.name == dtype.name:
            trace.data = np.require(trace.data, dtype, ['C_CONTIGUOUS'])
        else:
            msg = "GSE2 data must be of type %s, but are of type %s" % \
                (dtype.name, trace.data.dtype)
            raise Exception(msg)
        libgse2.write(header, trace.data, f, inplace)
    f.close()


def isGSE1(filename):
    """
    Checks whether a file is GSE1 or not.

    :type filename: string
    :param filename: GSE1 file to be checked.
    :rtype: bool
    :return: ``True`` if a GSE1 file.
    """
    # Open file.
    try:
        f = open(filename)
        data = f.readline()
    except:
        return False
    f.close()
    if data.startswith('WID1') or data.startswith('XW01'):
        return True
    return False


def readGSE1(filename, headonly=False, verify_chksum=True,
             **kwargs):  # @UnusedVariable
    """
    Reads a GSE1 file and returns a Stream object.

    GSE1 files containing multiple WID1 entries/traces are supported.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :type param: GSE2 file to be read.
    :type headonly: boolean, optional
    :param headonly: If True read only header of GSE1 file.
    :type verify_chksum: boolean, optional
    :param verify_chksum: If True verify Checksum and raise Exception if
        it is not correct.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/y2000.gse")
    """
    traces = []
    # read GSE1 file
    fh = open(filename, 'rb')
    while True:
        try:
            if headonly:
                header = libgse1.readHeader(fh)
                traces.append(Trace(header=header))
            else:
                header, data = libgse1.read(fh, verify_chksum=verify_chksum)
                traces.append(Trace(header=header, data=data))
        except EOFError:
            break
    fh.close()
    return Stream(traces=traces)
