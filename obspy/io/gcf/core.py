# -*- coding: utf-8 -*-
"""
GCF bindings to ObsPy core module.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import Stream, Trace, UTCDateTime

from . import libgcf


def merge_gcf_stream(st):
    """
    Merges GCF stream (replacing Stream.merge(-1) for headonly=True)

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: GCF Stream object with no data
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """
    traces = []
    for tr in st:
        delta = tr.stats.delta
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
        for trace in traces:
            if tr.id == trace.id and delta == trace.stats.delta \
               and not starttime == trace.stats.starttime:
                if 0 < starttime - trace.stats.endtime <= delta:
                    trace.stats.npts += tr.stats.npts
                    break
                elif 0 < trace.stats.starttime - endtime <= delta:
                    trace.stats.starttime = UTCDateTime(starttime)
                    trace.stats.npts += tr.stats.npts
                    break
        else:
            traces.append(tr)
    return Stream(traces=traces)


def _is_gcf(filename):
    """
    Checks whether a file is GCF or not.

    :type filename: str
    :param filename: GCF file to be checked.
    :rtype: bool
    :return: ``True`` if a GCF file.
    """
    try:
        with open(filename, 'rb') as f:
            libgcf.is_gcf(f)
    except Exception:
        return False
    return True


def _read_gcf(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a GCF file and returns a Stream object.

    only GCF files containing data records are supported.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: GCF file to be read.
    :type headonly: bool, optional
    :param headonly: If True read only head of GCF file.
    :type channel_prefix: str, optional
    :param channel_prefix: Channel band and instrument codes.
        Defaults to ``HH``.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.

    .. rubric:: Example
    >>> from obspy import read
    >>> st = read("/path/to/20160603_1955n.gcf", format="GCF")
    """
    traces = []
    with open(filename, 'rb') as f:
        while True:
            try:
                if headonly:
                    header = libgcf.read_header(f, **kwargs)
                    if header:
                        traces.append(Trace(header=header))
                else:
                    hd = libgcf.read(f, **kwargs)
                    if hd:
                        traces.append(Trace(header=hd[0], data=hd[1]))
            except EOFError:
                break
    st = Stream(traces=traces)
    if headonly:
        st = merge_gcf_stream(st)
    else:
        st.merge(-1)
    return st
