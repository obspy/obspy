# -*- coding: utf-8 -*-
"""
SAC bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Ammon & J. MacCarthy
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
import os
import struct

from obspy import Stream

from .sactrace import SACTrace


def _is_sac(filename):
    """
    Checks whether a file is a SAC file or not.

    :param filename: SAC file to be checked.
    :type filename: str, open file, or file-like object
    :rtype: bool
    :return: ``True`` if a SAC file.

    .. rubric:: Example

    >>> from obspy.core.util import get_example_file
    >>> _is_sac(get_example_file('test.sac'))
    True
    >>> _is_sac(get_example_file('test.mseed'))
    False
    """
    if isinstance(filename, io.BufferedIOBase):
        return _internal_is_sac(filename)
    elif isinstance(filename, (str, bytes)):
        with open(filename, "rb") as fh:
            return _internal_is_sac(fh)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_is_sac(buf):
    """
    Checks whether a file-like object contains a SAC file or not.

    :param buf: SAC file to be checked.
    :type buf: file-like object or open file
    :rtype: bool
    :return: ``True`` if a SAC file.
    """
    starting_pos = buf.tell()
    try:
        # read delta (first header float)
        delta_bin = buf.read(4)
        delta = struct.unpack('<f', delta_bin)[0]
        # read nvhdr (70 header floats, 6 position in header integers)
        buf.seek(starting_pos + 4 * 70 + 4 * 6, 0)
        nvhdr_bin = buf.read(4)
        nvhdr = struct.unpack('<i', nvhdr_bin)[0]
        # read leven (70 header floats, 35 header integers, 0 position in
        # header bool)
        buf.seek(starting_pos + 4 * 70 + 4 * 35, 0)
        leven_bin = buf.read(4)
        leven = struct.unpack('<i', leven_bin)[0]
        # read lpspol (70 header floats, 35 header integers, 1 position in
        # header bool)
        buf.seek(starting_pos + 4 * 70 + 4 * 35 + 4 * 1, 0)
        lpspol_bin = buf.read(4)
        lpspol = struct.unpack('<i', lpspol_bin)[0]
        # read lovrok (70 header floats, 35 header integers, 2 position in
        # header bool)
        buf.seek(starting_pos + 4 * 70 + 4 * 35 + 4 * 2, 0)
        lovrok_bin = buf.read(4)
        lovrok = struct.unpack('<i', lovrok_bin)[0]
        # read lcalda (70 header floats, 35 header integers, 3 position in
        # header bool)
        buf.seek(starting_pos + 4 * 70 + 4 * 35 + 4 * 3, 0)
        lcalda_bin = buf.read(4)
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
        if leven != 0 and leven != 1 and leven != -12345:
            return False
        if lpspol != 0 and lpspol != 1 and lpspol != -12345:
            return False
        if lovrok != 0 and lovrok != 1 and lovrok != -12345:
            return False
        if lcalda != 0 and lcalda != 1 and lcalda != -12345:
            return False
    except Exception:
        return False
    finally:
        # Reset buffer head position after reading.
        buf.seek(starting_pos, 0)
    return True


def _is_sac_xy(filename):
    """
    Checks whether a file is alphanumeric SAC file or not.

    :param filename: Alphanumeric SAC file to be checked.
    :type filename: str, open file, or file-like object
    :rtype: bool
    :return: ``True`` if a alphanumeric SAC file.

    .. rubric:: Example

    >>> from obspy.core.util import get_example_file
    >>> _is_sac_xy(get_example_file('testxy.sac'))
    True
    >>> _is_sac_xy(get_example_file('test.sac'))
    False
    """
    if isinstance(filename, io.BufferedIOBase):
        return _internal_is_sac_xy(filename)
    elif isinstance(filename, (str, bytes)):
        with open(filename, "rb") as fh:
            return _internal_is_sac_xy(fh)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_is_sac_xy(buf):
    """
    Checks whether a file is alphanumeric SAC file or not.

    :param buf: Alphanumeric SAC file to be checked.
    :type buf: file-like object or open file
    :rtype: bool
    :return: ``True`` if a alphanumeric SAC file.
    """
    cur_pos = buf.tell()
    try:
        try:
            hdcards = []
            # read in the header cards
            for _i in range(30):
                hdcards.append(buf.readline())
            npts = int(hdcards[15].split()[-1])
            # read in the seismogram
            seis = buf.read(-1).split()
        except Exception:
            return False
        # check that npts header value and seismogram length are consistent
        if npts != len(seis):
            return False
        return True
    finally:
        buf.seek(cur_pos, 0)


def _read_sac_xy(filename, headonly=False, debug_headers=False,
                 **kwargs):  # @UnusedVariable
    """
    Reads an alphanumeric SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: Alphanumeric SAC file to be read.
    :type filename: str, open file, or file-like object
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type headonly: bool
    :param debug_headers: Extracts also the SAC headers ``'nzyear', 'nzjday',
        'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'delta', 'scale', 'npts',
        'knetwk', 'kstnm', 'kcmpnm'`` which are usually directly mapped to the
        :class:`~obspy.core.stream.Stream` object if set to ``True``. Those
        values are not synchronized with the Stream object itself and won't
        be used during writing of a SAC file! Defaults to ``False``.
    :type debug_headers: bool
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/testxy.sac")
    """
    if isinstance(filename, io.BufferedIOBase):
        return _internal_read_sac_xy(buf=filename, headonly=headonly,
                                     debug_headers=debug_headers, **kwargs)
    else:
        with open(filename, "rb") as fh:
            return _internal_read_sac_xy(buf=fh, headonly=headonly,
                                         debug_headers=debug_headers, **kwargs)


def _internal_read_sac_xy(buf, headonly=False, debug_headers=False,
                          **kwargs):  # @UnusedVariable
    """
    Reads an alphanumeric SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param buf: Alphanumeric SAC file to be read.
    :type buf: file or file-like object
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type headonly: bool
    :param debug_headers: Extracts also the SAC headers ``'nzyear', 'nzjday',
        'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'delta', 'scale', 'npts',
        'knetwk', 'kstnm', 'kcmpnm'`` which are usually directly mapped to the
        :class:`~obspy.core.stream.Stream` object if set to ``True``. Those
        values are not synchronized with the Stream object itself and won't
        be used during writing of a SAC file! Defaults to ``False``.
    :type debug_headers: bool
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/testxy.sac")
    """
    sac = SACTrace.read(buf, headonly=headonly, ascii=True)
    # assign all header entries to a new dictionary compatible with ObsPy
    tr = sac.to_obspy_trace(debug_headers=debug_headers)

    return Stream([tr])


def _write_sac_xy(stream, filename, **kwargs):  # @UnusedVariable
    """
    Writes a alphanumeric SAC file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :param stream: The ObsPy Stream object to write.
    :type stream: :class:`~obspy.core.stream.Stream`
    :param filename: Name of file to write. In case an open file or
        file-like object is passed, this function only supports writing
        Stream objects containing a single Trace. This is a limitation of
        the SAC file format. An exception will be raised in case it's
        necessary.
    :type filename: str, open file, or file-like object

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("testxy.sac", format="SACXY")  #doctest: +SKIP
    """
    # SAC can only store one Trace per file.
    if isinstance(filename, io.BufferedIOBase):
        if len(stream) > 1:
            raise ValueError("If writing to a file-like object in the SAC "
                             "format, the Stream object can only contain "
                             "one Trace.")
        _internal_write_sac_xy(stream[0], filename, **kwargs)
        return
    elif isinstance(filename, (str, bytes)):
        # Otherwise treat it as a filename
        # Translate the common (renamed) entries
        base, ext = os.path.splitext(filename)
        for i, trace in enumerate(stream):
            if len(stream) != 1:
                filename = "%s%02d%s" % (base, i + 1, ext)
            with open(filename, "wb") as fh:
                _internal_write_sac_xy(trace, fh, **kwargs)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_write_sac_xy(trace, buf, **kwargs):  # @UnusedVariable
    """
    Writes a single trace to alphanumeric SAC file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :param trace: The ObsPy Trace object to write.
    :type trace: :class:`~obspy.core.trace.Trace`
    :param buf: Object to write to.
    :type buf: file-like object
    """
    sac = SACTrace.from_obspy_trace(trace, keep_sac_header=True)
    sac.write(buf, ascii=True, flush_headers=False)


def _read_sac(filename, headonly=False, debug_headers=False, fsize=True,
              **kwargs):  # @UnusedVariable
    """
    Reads an SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: SAC file to be read.
    :type filename: str, open file, or file-like object
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type headonly: bool
    :param debug_headers: Extracts also the SAC headers ``'nzyear', 'nzjday',
        'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'delta', 'scale', 'npts',
        'knetwk', 'kstnm', 'kcmpnm'`` which are usually directly mapped to the
        :class:`~obspy.core.stream.Stream` object if set to ``True``. Those
        values are not synchronized with the Stream object itself and won't
        be used during writing of a SAC file! Defaults to ``False``.
    :type debug_headers: bool
    :param fsize: Check if file size is consistent with theoretical size
        from header. Defaults to ``True``.
    :type fsize: bool
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/test.sac")
    """
    # Only byte buffers for binary SAC.
    if isinstance(filename, io.BufferedIOBase):
        return _internal_read_sac(buf=filename, headonly=headonly,
                                  debug_headers=debug_headers, fsize=fsize,
                                  **kwargs)
    elif isinstance(filename, (str, bytes)):
        with open(filename, "rb") as fh:
            return _internal_read_sac(buf=fh, headonly=headonly,
                                      debug_headers=debug_headers, fsize=fsize,
                                      **kwargs)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_read_sac(buf, headonly=False, debug_headers=False, fsize=True,
                       byteorder=None, **kwargs):  # @UnusedVariable
    """
    Reads an SAC file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param buf: SAC file to be read.
    :type buf: file or file-like object
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type headonly: bool
    :param debug_headers: Extracts also the SAC headers ``'nzyear', 'nzjday',
        'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'delta', 'scale', 'npts',
        'knetwk', 'kstnm', 'kcmpnm'`` which are usually directly mapped to the
        :class:`~obspy.core.stream.Stream` object if set to ``True``. Those
        values are not synchronized with the Stream object itself and won't
        be used during writing of a SAC file! Defaults to ``False``.
    :type debug_headers: bool
    :param fsize: Check if file size is consistent with theoretical size
        from header. Defaults to ``True``.
    :type fsize: bool
    :param byteorder: If omitted or None, automatic byte-order checking is
        done, starting with native order. If byteorder is specified,
        {'little', 'big'}, and incorrect, a
        :class:`obspy.io.sac.util.SacIOError` is raised. Only valid for binary
        files.
    :type byteorder: str, optional
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.
    """

    # Extract encoding flag (default to ascii)
    encoding_str = kwargs.get('encoding', 'ascii')

    # read SAC file
    sac = SACTrace.read(buf, headonly=headonly, ascii=False,
                        byteorder=byteorder, checksize=fsize,
                        encoding=encoding_str)
    # assign all header entries to a new dictionary compatible with an ObsPy
    tr = sac.to_obspy_trace(debug_headers=debug_headers, encoding=encoding_str)

    return Stream([tr])


def _write_sac(stream, filename, byteorder="<", **kwargs):  # @UnusedVariable
    """
    Writes a SAC file.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :param stream: The ObsPy Stream object to write.
    :type stream: :class:`~obspy.core.stream.Stream`
    :param filename: Name of file to write. In case an open file or
        file-like object is passed, this function only supports writing
        Stream objects containing a single Trace. This is a limitation of
        the SAC file format. An exception will be raised in case it's
        necessary.
    :type filename: str, open file, or file-like object
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MSBF or big-endian.
        Defaults to little endian.
    :type byteorder: int or str

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read()
    >>> st.write("test.sac", format="SAC")  #doctest: +SKIP
    """
    # Bytes buffer are ok, but only if the Stream object contains only one
    # Trace. SAC can only store one Trace per file.
    if isinstance(filename, io.BufferedIOBase):
        if len(stream) > 1:
            raise ValueError("If writing to a file-like object in the SAC "
                             "format, the Stream object can only contain "
                             "one Trace.")
        _internal_write_sac(stream[0], filename, byteorder=byteorder, **kwargs)
        return
    elif isinstance(filename, (str, bytes)):
        # Otherwise treat it as a filename
        # Translate the common (renamed) entries
        base, ext = os.path.splitext(filename)
        for i, trace in enumerate(stream):
            if len(stream) != 1:
                filename = "%s%02d%s" % (base, i + 1, ext)
            with open(filename, "wb") as fh:
                _internal_write_sac(trace, fh, byteorder=byteorder, **kwargs)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_write_sac(trace, buf, byteorder="<", keep_sac_header=True,
                        **kwargs):
    """
    Writes a single trace to an open file or file-like object

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :param trace: The ObsPy Trace object to write.
    :type trace: :class:`~obspy.core.trace.Trace`
    :param buf: Object to write to.
    :type buf: open file or file-like object
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MSBF or big-endian.
        Defaults to little endian.
    :type byteorder: int or str
    :param keep_sac_header: Whether to merge the ``Stats`` header with an
        existing ``Stats.sac`` SAC header, if one exists. Defaults to ``True``.
        See :func:`~obspy.io.sac.util.obspy_to_sac_header` for more.
    :type keep_sac_header: bool
    """
    if byteorder in ("<", 0, "0"):
        byteorder = 'little'
    elif byteorder in (">", 1, "1"):
        byteorder = 'big'
    else:
        msg = "Invalid byte order. It must be either '<', '>', 0 or 1"
        raise ValueError(msg)

    sac = SACTrace.from_obspy_trace(trace, keep_sac_header)
    sac.write(buf, ascii=False, byteorder=byteorder)
