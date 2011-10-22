# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from obspy.core import Trace, Stream
from obspy.sac.sacio import SacIO
import os
import string
import struct



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


def istext(filename, blocksize=512):
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
    if len(t) / len(s) > 0.30:
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
    if not istext(filename, blocksize=512):
        return False
    try:
        f = open(filename)
        hdcards = []
        # read in the header cards
        for _i in xrange(30):
            hdcards.append(f.readline())
        npts = int(hdcards[15].split()[-1])
        # read in the seismogram
        seis = f.read(-1).split()
    except:
        return False
    # check that npts header value and seismogram length are consistent
    if npts != len(seis):
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
        Alphanumeric SAC file to be read.

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
    # assign all header entries to a new dictionary compatible with ObsPy
    header = t.get_obspy_header()

    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])


def writeSACXY(stream, filename, **kwargs):
    """
    Writes an alphanumeric SAC file

    This function should NOT be called directly, it registers via the
    ObsPy :meth:`~obspy.core.stream.Stream.write` method of an ObsPy
    Stream object, call this instead.

    Parameters
    ----------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    filename : string
        Alphanumeric SAC file to be written.
    """
    # Translate the common (renamed) entries
    base, ext = os.path.splitext(filename)
    for i, trace in enumerate(stream):
        t = SacIO(trace)
        if len(stream) != 1:
            filename = "%s%02d%s" % (base, i + 1, ext)
        t.WriteSacXY(filename)
    return


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
    header = t.get_obspy_header()

    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])


def writeSAC(stream, filename, **kwargs):
    """
    Writes a SAC file

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
    base, ext = os.path.splitext(filename)
    for i, trace in enumerate(stream):
        t = SacIO(trace)
        if len(stream) != 1:
            filename = "%s%02d%s" % (base, i + 1, ext)
        t.WriteSacBinary(filename)
    return
