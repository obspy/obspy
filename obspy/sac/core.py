# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Ammon
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from obspy import Trace, Stream
from obspy.core.compatibility import is_bytes_buffer, is_text_buffer
from obspy.sac.sacio import SacIO, _isText
import os
import struct


def isSAC(filename):
    """
    Checks whether a file is a SAC file or not.

    :type filename: str or file-like object
    :param filename: SAC file to be checked.
    :rtype: bool
    :return: ``True`` if a SAC file.

    .. rubric:: Example

    >>> isSAC('/path/to/test.sac')  #doctest: +SKIP
    """
    if is_bytes_buffer(filename):
        return _isSAC(filename)
    elif isinstance(filename, (str, bytes)):
        with open(filename, "rb") as fh:
            return _isSAC(fh)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _isSAC(buf):
    """
    Checks whether a file-like obejcts contains a SAC file or not.

    :type buf: file-like object or open file.
    :param buf: SAC file to be checked.
    :rtype: bool
    :return: ``True`` if a SAC file.
    """
    starting_pos = buf.tell()
    try:
        # read delta (first header float)
        delta_bin = buf.read(4)
        delta = struct.unpack(native_str('<f'), delta_bin)[0]
        # read nvhdr (70 header floats, 6 position in header integers)
        buf.seek(4 * 70 + 4 * 6)
        nvhdr_bin = buf.read(4)
        nvhdr = struct.unpack(native_str('<i'), nvhdr_bin)[0]
        # read leven (70 header floats, 35 header integers, 0 position in
        # header bool)
        buf.seek(4 * 70 + 4 * 35)
        leven_bin = buf.read(4)
        leven = struct.unpack(native_str('<i'), leven_bin)[0]
        # read lpspol (70 header floats, 35 header integers, 1 position in
        # header bool)
        buf.seek(4 * 70 + 4 * 35 + 4 * 1)
        lpspol_bin = buf.read(4)
        lpspol = struct.unpack(native_str('<i'), lpspol_bin)[0]
        # read lovrok (70 header floats, 35 header integers, 2 position in
        # header bool)
        buf.seek(4 * 70 + 4 * 35 + 4 * 2)
        lovrok_bin = buf.read(4)
        lovrok = struct.unpack(native_str('<i'), lovrok_bin)[0]
        # read lcalda (70 header floats, 35 header integers, 3 position in
        # header bool)
        buf.seek(4 * 70 + 4 * 35 + 4 * 3)
        lcalda_bin = buf.read(4)
        lcalda = struct.unpack(native_str('<i'), lcalda_bin)[0]
        # check if file is big-endian
        if nvhdr < 0 or nvhdr > 20:
            nvhdr = struct.unpack(native_str('>i'), nvhdr_bin)[0]
            delta = struct.unpack(native_str('>f'), delta_bin)[0]
            leven = struct.unpack(native_str('>i'), leven_bin)[0]
            lpspol = struct.unpack(native_str('>i'), lpspol_bin)[0]
            lovrok = struct.unpack(native_str('>i'), lovrok_bin)[0]
            lcalda = struct.unpack(native_str('>i'), lcalda_bin)[0]
        # check again nvhdr
        if nvhdr < 1 or nvhdr > 20:
            return False
        if delta <= 0:
            return False
        if leven != 0 and leven != 1 and leven != -12345:
            return False
        if lpspol != 0 and lpspol != 1 and lpspol != -12345:
            return False
        if lovrok != 0 and lovrok != 1 and lovrok != -12345:
            return False
        if lcalda != 0 and lcalda != 1 and lcalda != -12345:
            return False
    except:
        return False
    finally:
        # Reset buffer head position after reading.
        buf.seek(starting_pos, 0)
    return True


def isSACXY(filename):
    """
    Checks whether a file is alphanumeric SAC file or not.

    :type filename: str
    :param filename: Alphanumeric SAC file to be checked.
    :rtype: bool
    :return: ``True`` if a alphanumeric SAC file.

    .. rubric:: Example

    >>> isSACXY('/path/to/testxy.sac')  #doctest: +SKIP
    """
    # First find out if it is a text or a binary file. This should
    # always be true if a file is a text-file and only true for a
    # binary file in rare occasions (Recipe 173220 found on
    # http://code.activestate.com/
    if not _isText(filename, blocksize=512):
        return False
    try:
        with open(filename) as f:
            hdcards = []
            # read in the header cards
            for _i in range(30):
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

    :type filename: filename of file-like object.
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
    # Alphanumeric so bytes and text should both work.
    if is_bytes_buffer(filename) or is_text_buffer(filename):
        return _readSACXY(buf=filename, headonly=headonly,
                          debug_headers=debug_headers, **kwargs)
    else:
        with open(filename, "rb") as fh:
            return _readSACXY(buf=fh, headonly=headonly,
                              debug_headers=debug_headers, **kwargs)


def _readSACXY(buf, headonly=False, debug_headers=False,
               **kwargs):  # @UnusedVariable
    """
    Reads an alphanumeric SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type buf: file or file-like object
    :param buf: Alphanumeric SAC file to be read.
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
        t.ReadSacXYHeader(buf)
    else:
        t.ReadSacXY(buf)
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

    :type filename: filename, open file, or file-like object.
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
    # Only byte buffers for binary SAC.
    if is_bytes_buffer(filename):
        return _readSAC(buf=filename, headonly=headonly,
                        debug_headers=debug_headers, fsize=fsize, **kwargs)
    elif isinstance(filename, (str, bytes)):
        with open(filename, "rb") as fh:
            return _readSAC(buf=fh, headonly=headonly,
                            debug_headers=debug_headers, fsize=fsize, **kwargs)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _readSAC(buf, headonly=False, debug_headers=False, fsize=True,
             **kwargs):  # @UnusedVariable
    """
    Reads an SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type buf: file or file-like object.
    :param buf: SAC file to be read.
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
    """
    # read SAC file
    t = SacIO(debug_headers=debug_headers)
    if headonly:
        t.ReadSacHeader(buf)
    else:
        t.ReadSacFile(buf, fsize)
    # assign all header entries to a new dictionary compatible with an ObsPy
    header = t.get_obspy_header()

    if headonly:
        tr = Trace(header=header)
    else:
        tr = Trace(header=header, data=t.seis)
    return Stream([tr])


def writeSAC(stream, filename, byteorder="<", **kwargs):  # @UnusedVariable
    """
    Writes a SAC file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str or file-like object
    :param filename: Name of file to write. If it is not a filename, it only
        supports writing single Trace streams.
    :type byteorder: int or str, optional
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MSBF or big-endian.
        Defaults to little endian.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("test.sac", format="SAC")  #doctest: +SKIP
    """
    # Bytes buffer are ok, but only if the Stream object contains only one
    # Trace. SAC can only store one Trace per file.
    if is_bytes_buffer(filename):
        if len(stream) > 1:
            raise ValueError("If writing to a file-like object in the SAC "
                             "format, the Stream object must only contain "
                             "one Trace")
        _writeSAC(stream[0], filename, byteorder=byteorder, **kwargs)
        return
    elif isinstance(filename, (str, bytes)):
        # Otherwise treat it as a filename
        # Translate the common (renamed) entries
        base, ext = os.path.splitext(filename)
        for i, trace in enumerate(stream):
            if len(stream) != 1:
                filename = "%s%02d%s" % (base, i + 1, ext)
            with open(filename, "wb") as fh:
                _writeSAC(trace, fh, byteorder=byteorder, **kwargs)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _writeSAC(trace, buf, byteorder="<", **kwargs):  # @UnusedVariable
    """
    Writes a single trace to an open file or file-like object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type trace: :class:`~obspy.core.trace.Trace`
    :param trace: The ObsPy Trace object to write.
    :type buf: open file or file-like object
    :param buf: Object to write to.
    :type byteorder: int or str, optional
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MSBF or big-endian.
        Defaults to little endian.
    """
    if byteorder in ("<", 0, "0"):
        byteorder = 0
    elif byteorder in (">", 1, "1"):
        byteorder = 1
    else:
        msg = "Invalid byte order. It must be either '<', '>', 0 or 1"
        raise ValueError(msg)
    t = SacIO(trace)
    if (byteorder == 1 and t.byteorder == 'little') or \
            (byteorder == 0 and t.byteorder == 'big'):
        t.swap_byte_order()
    t.WriteSacBinary(buf)
