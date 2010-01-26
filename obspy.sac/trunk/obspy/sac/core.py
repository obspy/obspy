# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.
"""

from obspy.core import Trace, UTCDateTime, Stream
from obspy.sac.sacio import ReadSac
import struct
import numpy as np
import os


# we put here everything but the time, they are going to stats.starttime
# left SAC attributes, right trace attributes
convert_dict = {
    'npts': 'npts',
    'delta':'sampling_rate',
    'kcmpnm': 'channel',
    'kstnm': 'station'
}

#XXX NOTE not all values from the read in dictionary are converted
# this is definetly a problem when reading an writing a read SAC file.
sac_extra = [
    'depmin', 'depmax', 'scale', 'odelta', 'b', 'e', 'o', 'a', 't0', 't1',
    't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'f', 'stla', 'stlo',
    'stel', 'stdp', 'evla', 'evlo', 'evdp', 'mag', 'user0', 'user1', 'user2',
    'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'dist',
    'az', 'baz', 'gcarc', 'depmen', 'cmpaz', 'cmpinc', 'nzyear', 'nzjday',
    'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'nvhdr', 'norid', 'nevid', 'nwfid',
    'iftype', 'idep', 'iztype', 'iinst', 'istreg', 'ievreg', 'ievtype',
    'iqual', 'isynth', 'imagtyp', 'imagsrc', 'leven', 'lpspol', 'lovrok',
    'lcalda', 'kevnm', 'khole', 'ko', 'ka', 'kt0', 'kt1', 'kt2', 'kt3', 'kt4',
    'kt5', 'kt6', 'kt7', 'kt8', 'kt9', 'kf', 'kuser0', 'kuser1', 'kuser2',
    'knetwk', 'kdatrd', 'kinst',
]


def isSAC(filename):
    """
    Checks whether a file is SAC or not. Returns True or False.
    
    :param filename: SAC file to be read.
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
        # File-size and theoretical size inconsistent!
        return False
    return True


def readSAC(filename, headonly=False, **kwargs):
    """
    Reads a SAC file and returns an L{obspy.Stream} object.
    
    :param filename: SAC file to be read.
    :rtype: L{obspy.Stream}.
    :return: A ObsPy Stream object.
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
        header[j] = t.GetHvalue(i)
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


def writeSAC(stream_object, filename, **kwargs):
    """
    Writes SAC file.
    
    :type stream_object: L{obspy.Stream}.
    :param stream_object: A ObsPy Stream object.
    :param filename: SAC file to be written.
    """
    # Translate the common (renamed) entries
    i = 0
    base, ext = os.path.splitext(filename)
    for trace in stream_object:
        t = ReadSac()
        t.InitArrays()
        # Check for necessary values, set a default if they are missing
        trace.stats.setdefault('npts', len(trace.data))
        trace.stats.setdefault('sampling_rate', 1.0)
        trace.stats.setdefault('starttime', UTCDateTime(0.0))
        # SAC version needed 0<version<20
        trace.stats.sac.setdefault('nvhdr', 1)
        # filling ObsPy defaults
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
        # building array of floats, require correct type
        t.seis = np.require(trace.data, '<f4')
        if i != 0:
            filename = "%s%02d%s" % (base, i, ext)
        t.WriteSacBinary(filename)
        i += 1
