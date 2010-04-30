# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from obspy.core import Trace, UTCDateTime, Stream, AttribDict
from obspy.sac.sacio import ReadSac
import struct
import numpy as np
import os


# we put here everything but the time, they are going to stats.starttime
# left SAC attributes, right trace attributes, see also
# http://www.iris.edu/KB/questions/13/SAC+file+format 
convert_dict = {
    'npts': 'npts',
    'delta': 'delta',
    'kcmpnm': 'channel',
    'kstnm': 'station',
    'scale': 'calib',
    'knetwk': 'network',
    'khole': 'location'
}

#XXX NOTE not all values from the read in dictionary are converted
# this is definetly a problem when reading an writing a read SAC file.
sac_extra = [
    'depmin', 'depmax', 'odelta', 'b', 'e', 'o', 'a', 't0', 't1',
    't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'f', 'stla', 'stlo',
    'stel', 'stdp', 'evla', 'evlo', 'evdp', 'mag', 'user0', 'user1', 'user2',
    'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'dist',
    'az', 'baz', 'gcarc', 'depmen', 'cmpaz', 'cmpinc', 'nzyear', 'nzjday',
    'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'nvhdr', 'norid', 'nevid', 'nwfid',
    'iftype', 'idep', 'iztype', 'iinst', 'istreg', 'ievreg', 'ievtype',
    'iqual', 'isynth', 'imagtyp', 'imagsrc', 'leven', 'lpspol', 'lovrok',
    'lcalda', 'kevnm', 'ko', 'ka', 'kt0', 'kt1', 'kt2', 'kt3', 'kt4',
    'kt5', 'kt6', 'kt7', 'kt8', 'kt9', 'kf', 'kuser0', 'kuser1', 'kuser2',
    'kdatrd', 'kinst',
]


def isSAC(filename):
    """
    Checks whether a file is SAC or not. Returns True or False.

    Parameters
    ----------
    filename : string
        SAC file to be checked.
    """
    try:
        f = open(filename, 'rb')
        # 70 header floats, 9 position in header integers
        f.seek(4 * 70 + 4 * 9)
        data = f.read(4)
        f.close()
        npts = struct.unpack('<i', data)[0]
    except:
        return False
    # check file size
    st = os.stat(filename)
    sizecheck = st.st_size - (632 + 4 * npts)
    if sizecheck != 0:
        ## check if if file is big-endian
        npts = struct.unpack('>i', data)[0]
        sizecheck = st.st_size - (632 + 4 * npts)
        if sizecheck != 0:
            # File-size and theoretical size inconsistent!
            return False
    return True


def readSAC(filename, headonly=False, **kwargs):
    """
    Reads a SAC file and returns an ObsPy Stream object.
    
    This function should NOT be called directly, it registers via the
    ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        SAC file to be read.

    Returns
    -------
    :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.

    Example
    -------
    >>> from obspy.core import read # doctest: +SKIP
    >>> st = read("sac_file") # doctest: +SKIP
    """
    # read SAC file
    t = ReadSac()
    if headonly:
        t.ReadSacHeader(filename)
    else:
        t.ReadSacFile(filename)
    # assign all header entries to a new dictionary compatible with an ObsPy
    header = {}
    for i, j in convert_dict.iteritems():
        value = t.GetHvalue(i)
        if isinstance(value, str):
            value = value.strip()
        header[j] = value
    header['sac'] = {}
    for i in sac_extra:
        header['sac'][i] = t.GetHvalue(i)
    # convert time to UTCDateTime
    header['starttime'] = UTCDateTime(year=t.GetHvalue('nzyear'),
                                      julday=t.GetHvalue('nzjday'),
                                      hour=t.GetHvalue('nzhour'),
                                      minute=t.GetHvalue('nzmin'),
                                      second=t.GetHvalue('nzsec'),
                                      microsecond=t.GetHvalue('nzmsec') * 1000)
    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])


def writeSAC(stream, filename, **kwargs):
    """
    Writes SAC file.
    
    This function should NOT be called directly, it registers via the
    obspy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
    Stream object, call this instead.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    filename : string
        SAC file to be written.
    """
    # Translate the common (renamed) entries
    i = 0
    base, ext = os.path.splitext(filename)
    for trace in stream:
        t = ReadSac()
        t.InitArrays()
        # Check for necessary values, set a default if they are missing
        trace.stats.setdefault('npts', len(trace.data))
        trace.stats.setdefault('sampling_rate', 1.0)
        trace.stats.setdefault('starttime', UTCDateTime(0.0))
        # SAC version needed 0<version<20
        trace.stats.setdefault('sac', AttribDict())
        trace.stats['sac'].setdefault('nvhdr', 6)
        # building array of floats, require correct type
        # filling ObsPy defaults
        t.fromarray(np.require(trace.data, '<f4'))
        for _j, _k in convert_dict.iteritems():
            try:
                t.SetHvalue(_j, trace.stats[_k])
            except:
                pass
        # filling up SAC specific values
        for _i in sac_extra:
            try:
                t.SetHvalue(_i, trace.stats.sac[_i])
            except:
                pass
        # setting start time
        start = trace.stats.starttime
        t.SetHvalue('nzyear', start.year)
        t.SetHvalue('nzjday', start.strftime("%j"))
        t.SetHvalue('nzhour', start.hour)
        t.SetHvalue('nzmin', start.minute)
        t.SetHvalue('nzsec', start.second)
        t.SetHvalue('nzmsec', start.microsecond / 1e3)
        if i != 0:
            filename = "%s%02d%s" % (base, i, ext)
        t.WriteSacBinary(filename)
        i += 1
