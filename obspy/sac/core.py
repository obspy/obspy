# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Ammon
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy import Trace, Stream
from obspy.sac.sacio import SacIO, _isText
import os
import struct


def isSAC(filename):
    """
    Checks whether a file is a SAC file or not.

    :type filename: string
    :param filename: SAC file to be checked.
    :rtype: bool
    :return: ``True`` if a SAC file.

    .. rubric:: Example

    >>> isSAC('/path/to/test.sac')  #doctest: +SKIP
    """
    try:
        with open(filename, 'rb') as f:
            # read delta (first header float)
            delta_bin = f.read(4)
            delta = struct.unpack('<f', delta_bin)[0]
            # read nvhdr (70 header floats, 6 position in header integers)
            f.seek(4 * 70 + 4 * 6)
            nvhdr_bin = f.read(4)
            nvhdr = struct.unpack('<i', nvhdr_bin)[0]
            # read leven (70 header floats, 35 header integers, 0 position in
            # header bool)
            f.seek(4 * 70 + 4 * 35)
            leven_bin = f.read(4)
            leven = struct.unpack('<i', leven_bin)[0]
            # read lpspol (70 header floats, 35 header integers, 1 position in
            # header bool)
            f.seek(4 * 70 + 4 * 35 + 4 * 1)
            lpspol_bin = f.read(4)
            lpspol = struct.unpack('<i', lpspol_bin)[0]
            # read lovrok (70 header floats, 35 header integers, 2 position in
            # header bool)
            f.seek(4 * 70 + 4 * 35 + 4 * 2)
            lovrok_bin = f.read(4)
            lovrok = struct.unpack('<i', lovrok_bin)[0]
            # read lcalda (70 header floats, 35 header integers, 3 position in
            # header bool)
            f.seek(4 * 70 + 4 * 35 + 4 * 3)
            lcalda_bin = f.read(4)
            lcalda = struct.unpack('<i', lcalda_bin)[0]
            # check if file is big-endian
            if nvhdr < 0 or nvhdr > 20:
                nvhdr = struct.unpack('>i', nvhdr_bin)[0]
                delta = struct.unpack('>f', delta_bin)[0]
                leven = struct.unpack('>i', leven_bin)[0]
                lpspol = struct.unpack('>i', lpspol_bin)[0]
                lovrok = struct.unpack('>i', lovrok_bin)[0]
                lcalda = struct.unpack('>i', lcalda_bin)[0]
            # check again nvhdr
            if nvhdr < 1 or nvhdr > 20:
                return False
            if delta <= 0:
                return False
            if leven != 0 and leven != 1:
                return False
            if lpspol != 0 and lpspol != 1 and lpspol != -12345:
                return False
            if lovrok != 0 and lovrok != 1 and lovrok != -12345:
                return False
            if lcalda != 0 and lcalda != 1 and lcalda != -12345:
                return False
    except:
        return False
    return True


def isSACXY(filename):
    """
    Checks whether a file is alphanumeric SAC file or not.

    :type filename: string
    :param filename: Alphanumeric SAC file to be checked.
    :rtype: bool
    :return: ``True`` if a alphanumeric SAC file.

    .. rubric:: Example

    >>> isSACXY('/path/to/testxy.sac')  #doctest: +SKIP
    """
    ### First find out if it is a text or a binary file. This should
    ### always be true if a file is a text-file and only true for a
    ### binary file in rare occasions (Recipe 173220 found on
    ### http://code.activestate.com/
    if not _isText(filename, blocksize=512):
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


def readSACXY(filename, headonly=False, debug_headers=False,
              **kwargs):  # @UnusedVariable
    """
    Reads an alphanumeric SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Alphanumeric SAC file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type debug_headers: bool, optional
    :param debug_headers: Extracts also the SAC headers ``'nzyear', 'nzjday',
        'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'delta', 'scale', 'npts',
        'knetwk', 'kstnm', 'kcmpnm'`` which are usually directly mapped to the
        :class:`~obspy.core.stream.Stream` object if set to ``True``. Those
        values are not synchronized with the Stream object itself and won't
        be used during writing of a SAC file! Defaults to ``False``.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read # doctest: +SKIP
    >>> st = read("/path/to/testxy.sac") # doctest: +SKIP
    """
    t = SacIO(debug_headers=debug_headers)
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


def writeSACXY(stream, filename, **kwargs):  # @UnusedVariable
    """
    Writes a alphanumeric SAC file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("testxy.sac", format="SACXY")  #doctest: +SKIP
    """
    # Translate the common (renamed) entries
    base, ext = os.path.splitext(filename)
    for i, trace in enumerate(stream):
        t = SacIO(trace)
        if len(stream) != 1:
            filename = "%s%02d%s" % (base, i + 1, ext)
        t.WriteSacXY(filename)
    return


def readSAC(filename, headonly=False, debug_headers=False, fsize=True,
            **kwargs):  # @UnusedVariable
    """
    Reads an SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SAC file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type debug_headers: bool, optional
    :param debug_headers: Extracts also the SAC headers ``'nzyear', 'nzjday',
        'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'delta', 'scale', 'npts',
        'knetwk', 'kstnm', 'kcmpnm'`` which are usually directly mapped to the
        :class:`~obspy.core.stream.Stream` object if set to ``True``. Those
        values are not synchronized with the Stream object itself and won't
        be used during writing of a SAC file! Defaults to ``False``.
    :type fsize: bool, optional
    :param fsize: Check if file size is consistent with theoretical size
        from header. Defaults to ``True``.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read # doctest: +SKIP
    >>> st = read("/path/to/test.sac") # doctest: +SKIP
    """
    # read SAC file
    t = SacIO(debug_headers=debug_headers)
    if headonly:
        t.ReadSacHeader(filename)
    else:
        t.ReadSacFile(filename, fsize)
    # assign all header entries to a new dictionary compatible with an ObsPy
    header = t.get_obspy_header()

    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])


def writeSAC(stream, filename, **kwargs):  # @UnusedVariable
    """
    Writes a SAC file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of file to write.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("test.sac", format="SAC")  #doctest: +SKIP
    """
    # Translate the common (renamed) entries
    base, ext = os.path.splitext(filename)
    for i, trace in enumerate(stream):
        t = SacIO(trace)
        if len(stream) != 1:
            filename = "%s%02d%s" % (base, i + 1, ext)
        t.WriteSacBinary(filename)
    return
