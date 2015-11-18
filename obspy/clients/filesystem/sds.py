# -*- coding: utf-8 -*-
"""
obspy.clients.filesystem.sds - read support for SeisComP Data Structure
=======================================================================
This module provides read support for data stored locally in a SeisComP Data
Structure (SDS) directory structure.

The directory and file layout of SDS is defined as:

    <SDSdir>/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY

These fields are defined by SDS as follows:

    SDSdir :  arbitrary base directory
    YEAR   :  4 digit year
    NET    :  Network code/identifier, up to 8 characters, no spaces
    STA    :  Station code/identifier, up to 8 characters, no spaces
    CHAN   :  Channel code/identifier, up to 8 characters, no spaces
    TYPE   :  1 characters indicating the data type, recommended types are:
               'D' - Waveform data
               'E' - Detection data
               'L' - Log data
               'T' - Timing data
               'C' - Calibration data
               'R' - Response data
               'O' - Opaque data
    LOC    :  Location identifier, up to 8 characters, no spaces
    DAY    :  3 digit day of year, padded with zeros

See https://www.seiscomp3.org/wiki/doc/applications/slarchive/SDS.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import glob
import os
from datetime import timedelta

from obspy import Stream, read
from obspy.core.util.misc import BAND_CODE


SDS_FMTSTR = os.path.join(
    "{year}", "{network}", "{station}", "{channel}.{type}",
    "{network}.{station}.{location}.{channel}.{type}.{year}.{doy:03d}")


class Client(object):
    """
    Request client for SeisComP Data Structure archive on local filesystem.

    For details see the :meth:`~obspy.clients.filesystem.sds.Client.__init__()`
    method.
    """
    def __init__(self, sds_root, sds_type="D", format="MSEED",
                 fileborder_seconds=30, fileborder_samples=5000):
        """
        Initialize a SDS local filesystem client.

        >>> from obspy.clients.filesystem.sds import Client
        >>> client = Client("/my/SDS/archive/root")  # doctest: +SKIP

        :type sds_root: str
        :param sds_root: Root directory of SDS archive.
        :type sds_type: str
        :param sds_type: SDS data type identifier, one single character. Types
            recommended by the SDS definition are: 'D' - Waveform data,
            'E' - Detection data, 'L' - Log data, 'T' - Timing data,
            'C' - Calibration data, 'R' - Response data, 'O' - Opaque data. Can
            also be wildcarded by setting to ``?`` or ``*``.
        :type format: str
        :param format: File format the data is stored in, see
            :func:`~obspy.core.stream.read()` for a list of file formats
            supported by ObsPy. Usually, SDS archives are stored in "MSEED"
            format. Can be set to ``None`` for file format autodetection
            (slowing down the reading).
        :type fileborder_seconds: float
        :param fileborder_seconds: Defines in which cases the client checks the
            previous/next daily file for the requested data (data in SDS
            archives usually spill over on the day break, at least for a few
            seconds). For example setting ``fileborder_seconds=30`` means that
            for a request with ``starttime`` at ``00:00:29`` UTC (or an
            ``endtime`` at ``23:59:31``), the previous daily file is also
            checked, if it contains matching data. The maximum of both
            ``fileborder_seconds`` and ``fileborder_samples`` is used when
            determining if previous/next day should be checked for data.
        :type fileborder_samples: int
        :param fileborder_samples: Similar to ``fileborder_seconds``. The given
            number of samples is converted to seconds by mapping the band
            code of the requested channel to sampling frequency. The maximum of
            both ``fileborder_seconds`` and ``fileborder_samples`` is used when
            determining if previous/next day should be checked for data.
        """
        if not os.path.isdir(sds_root):
            msg = ("SDS root is not a local directory: " + sds_root)
            raise IOError(msg)
        self.sds_root = sds_root
        self.sds_type = sds_type
        self.format = format
        self.fileborder_seconds = fileborder_seconds
        self.fileborder_samples = fileborder_samples

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, merge=-1, sds_type=None):
        """
        Read data from a local SeisComP Data Structure (SDS) directory tree.

        >>> from obspy import UTCDateTime
        >>> t = UTCDateTime("2015-10-12T12")
        >>> st = client.get_waveforms("IU", "ANMO", "*", "HH?", t, t+30)
        ... # doctest: +SKIP

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :type merge: int or None
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge(...) <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: :class:`~obspy.core.stream.Stream`
        """
        if starttime >= endtime:
            msg = ("'endtime' must be after 'starttime'.")
            raise ValueError(msg)
        sds_type = sds_type or self.sds_type
        # SDS has data sometimes in adjacent days, so also try to read the
        # requested data from those files. Usually this is only a few seconds
        # of data after midnight, but for now we play safe here to catch all
        # requested data (and with MiniSEED - the usual SDS file format - we
        # can use starttime/endtime kwargs anyway to read only desired parts).
        year_doy = set()
        # determine how far before starttime/after endtime we should check
        # other dayfiles for the data
        t_buffer = self.fileborder_samples / BAND_CODE.get(channel[:1], 20.0)
        t_buffer = max(t_buffer, self.fileborder_seconds)
        t = starttime - t_buffer
        t_max = endtime + t_buffer
        # make a list of year/doy combinations that covers the whole requested
        # time window (plus day before and day after)
        while t < t_max:
            year_doy.add((t.year, t.julday))
            t += timedelta(days=1)
        year_doy.add((t_max.year, t_max.julday))

        st = Stream()
        full_paths = set()
        for year, doy in year_doy:
            filename = SDS_FMTSTR.format(
                network=network, station=station, location=location,
                channel=channel, year=year, doy=doy, type=sds_type)
            full_path = os.path.join(self.sds_root, filename)
            full_paths = full_paths.union(glob.glob(full_path))
        for full_path in full_paths:
            st += read(full_path, format=self.format, starttime=starttime,
                       endtime=endtime)

        st.trim(starttime, endtime)
        if merge is None or merge is False:
            pass
        else:
            st.merge(merge)
        return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
