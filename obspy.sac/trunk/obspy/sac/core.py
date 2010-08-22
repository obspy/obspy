# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from obspy.core import Trace, Stream
from obspy.sac.sacio import SacIO
import struct
import os
import string


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

# all the sac specific extras, the SAC reference time specific headers are
# handled separately and are directly controlled by trace.stats.starttime.
sac_extra = [
    'depmin', 'depmax', 'odelta', 'o', 'a', 't0', 't1',
    't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 'f', 'stla', 'stlo',
    'stel', 'stdp', 'evla', 'evlo', 'evdp', 'mag', 'user0', 'user1', 'user2',
    'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9', 'dist',
    'az', 'baz', 'gcarc', 'depmen', 'cmpaz', 'cmpinc',
    'nvhdr', 'norid', 'nevid', 'nwfid',
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
        # check if file is big-endian
        npts = struct.unpack('>i', data)[0]
        sizecheck = st.st_size - (632 + 4 * npts)
        if sizecheck != 0:
            # File-size and theoretical size inconsistent!
            return False
    return True

def istext(filename, blocksize = 512):
    ### Find out if it is a text or a binary file. This should
    ### always be true if a file is a text-file and only true for a
    ### binary file in rare occasions (Recipe 173220 found on
    ### http://code.activestate.com/
    text_characters = "".join(map(chr, range(32, 127)) + list("\n\r\t\b"))
    _null_trans = string.maketrans("", "")
    s = open(filename).read(blocksize)
    if "\0" in s:
        return 0

    if not s:  # Empty files are considered text
        return 1

    # Get the non-text characters (maps a character to itself then
    # use the 'remove' option to get rid of the text characters.)
    t = s.translate(_null_trans, text_characters)

    # If more than 30% non-text characters, then
    # this is considered a binary file
    if len(t)/len(s) > 0.30:
        return 0
    return 1



def isSACXY(filename):
    """
    Checks whether a file is alphanumeric SAC or not. Returns True or False.

    Parameters
    ----------
    filename : string
        SAC file to be checked.
    """
    ### First find out if it is a text or a binary file. This should
    ### always be true if a file is a text-file and only true for a
    ### binary file in rare occasions (Recipe 173220 found on
    ### http://code.activestate.com/
    if not istext(filename, blocksize = 512):
        return False
    try:
        SacIO.ReadSacXY(filename)
    except:
        return False
    return True

def readSACXY(filename, headonly=False, **kwargs):
    """
    Reads an alphanumeric SAC file and returns an ObsPy Stream object.
    
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
    t = SacIO()
    if headonly:
        t.ReadSacXYHeader(filename)
    else:
        t.ReadSacXY(filename)
    # assign all header entries to a new dictionary compatible with an ObsPy
    header = {}

    # convert common header types of the obspy trace object
    for i, j in convert_dict.iteritems():
        value = t.GetHvalue(i)
        if isinstance(value, str):
            value = value.strip()
            if value == '-12345':
                value = ''
        header[j] = value
    if header['calib'] == -12345.0:
        header['calib'] = 1.0
    # assign extra header types of sac
    header['sac'] = {}
    for i in sac_extra:
        header['sac'][i] = t.GetHvalue(i)
    # convert time to UTCDateTime
    header['starttime'] = t.starttime
    # always add the begin time (if it's defined) to get the given
    # SAC reference time, no matter which iztype is given
    # note that the B and E times should not be in the sac_extra
    # dictionary, as they would overwrite the t.fromarray which sets them
    # according to the starttime, npts and delta.
    header['sac']['b'] = float(t.GetHvalue('b'))
    header['sac']['e'] = float(t.GetHvalue('e'))
    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])

def writeSACXY(stream, filename, **kwargs):
    """
    Writes SAC file.
    
    This function should NOT be called directly, it registers via the
    ObsPy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
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
        t = SacIO()
        # extracting relative SAC time as specified with b
        try:
            b = float(trace.stats['sac']['b'])
        except KeyError:
            b = 0.0
        # filling in SAC/sacio specific defaults
        t.fromarray(trace.data, begin=b, delta=trace.stats.delta,
                    starttime=trace.stats.starttime)
        # overwriting with ObsPy defaults
        for _j, _k in convert_dict.iteritems():
            t.SetHvalue(_j, trace.stats[_k])
        # overwriting up SAC specific values
        # note that the SAC reference time values (including B and E) are
        # not used in here any more, they are already set by t.fromarray
        # and directly deduce from tr.starttime
        for _i in sac_extra:
            try:
                t.SetHvalue(_i, trace.stats.sac[_i])
            except KeyError:
                pass
        if len(stream) != 1:
            filename = "%s%02d%s" % (base, i+1, ext)
        t.WriteSacXY(filename)
        i += 1

                                                        

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
    t = SacIO()
    if headonly:
        t.ReadSacHeader(filename)
    else:
        t.ReadSacFile(filename)
    # assign all header entries to a new dictionary compatible with an ObsPy
    header = {}

    # convert common header types of the ObsPy trace object
    for i, j in convert_dict.iteritems():
        value = t.GetHvalue(i)
        if isinstance(value, str):
            value = value.strip()
            if value == '-12345':
                value = ''
        header[j] = value
    if header['calib'] == -12345.0:
        header['calib'] = 1.0
    # assign extra header types of SAC
    header['sac'] = {}
    for i in sac_extra:
        header['sac'][i] = t.GetHvalue(i)
    # convert time to UTCDateTime
    header['starttime'] = t.starttime
    # always add the begin time (if it's defined) to get the given
    # SAC reference time, no matter which iztype is given
    # note that the B and E times should not be in the sac_extra
    # dictionary, as they would overwrite the t.fromarray which sets them
    # according to the starttime, npts and delta.
    header['sac']['b'] = float(t.GetHvalue('b'))
    header['sac']['e'] = float(t.GetHvalue('e'))
    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])


def writeSAC(stream, filename, **kwargs):
    """
    Writes SAC file.
    
    This function should NOT be called directly, it registers via the
    ObsPy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
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
        t = SacIO()
        # extracting relative SAC time as specified with b
        try:
            b = float(trace.stats['sac']['b'])
        except KeyError:
            b = 0.0
        # filling in SAC/sacio specific defaults
        t.fromarray(trace.data, begin=b, delta=trace.stats.delta,
                    starttime=trace.stats.starttime)
        # overwriting with ObsPy defaults
        for _j, _k in convert_dict.iteritems():
            t.SetHvalue(_j, trace.stats[_k])
        # overwriting up SAC specific values
        # note that the SAC reference time values (including B and E) are
        # not used in here any more, they are already set by t.fromarray
        # and directly deduce from tr.starttime
        for _i in sac_extra:
            try:
                t.SetHvalue(_i, trace.stats.sac[_i])
            except KeyError:
                pass
        if len(stream) != 1:
            filename = "%s%02d%s" % (base, i + 1, ext)
        t.WriteSacBinary(filename)
        i += 1
