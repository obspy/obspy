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
import warnings
from datetime import timedelta

import numpy as np

from obspy import Stream, read, UTCDateTime
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
    FMTSTR = SDS_FMTSTR

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
                      endtime, merge=-1, sds_type=None, **kwargs):
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

        st = Stream()
        full_paths = self._get_filenames(
            network=network, station=station, location=location,
            channel=channel, starttime=starttime, endtime=endtime,
            sds_type=sds_type)
        for full_path in full_paths:
            st += read(full_path, format=self.format, starttime=starttime,
                       endtime=endtime, **kwargs)

        # make sure we only have the desired data, just in case the file
        # contents do not match the expected SEED id
        st = st.select(network=network, station=station, location=location,
                       channel=channel)

        # avoid trim/merge operations when we do a headonly read for
        # `_get_availability_percentage()`
        if kwargs.get("_no_trim_or_merge", False):
            return st

        st.trim(starttime, endtime)
        if merge is None or merge is False:
            pass
        else:
            st.merge(merge)
        return st

    def _get_filenames(self, network, station, location, channel, starttime,
                       endtime, sds_type=None):
        """
        Get list of filenames for certain waveform and time span.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of time span.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of time span.
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: list of str
        """
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

        full_paths = set()
        for year, doy in year_doy:
            filename = self.FMTSTR.format(
                network=network, station=station, location=location,
                channel=channel, year=year, doy=doy, type=sds_type)
            full_path = os.path.join(self.sds_root, filename)
            full_paths = full_paths.union(glob.glob(full_path))

        return full_paths

    def _get_filename(self, network, station, location, channel, time,
                      sds_type=None):
        """
        Get filename for certain waveform.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Time of interest.
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: str
        """
        sds_type = sds_type or self.sds_type
        filename = self.FMTSTR.format(
            network=network, station=station, location=location,
            channel=channel, year=time.year, doy=time.julday, type=sds_type)
        return os.path.join(self.sds_root, filename)

    def get_availability_percentage(self, network, station, location, channel,
                                    starttime, endtime, sds_type=None):
        """
        Get percentage of available data.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: 2-tuple (float, int)
        :returns: 2-tuple of percentage of available data (``0.0`` to ``1.0``)
            and number of gaps/overlaps.
        """
        if starttime >= endtime:
            msg = ("'endtime' must be after 'starttime'.")
            raise ValueError(msg)
        sds_type = sds_type or self.sds_type
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Keyword headonly cannot be combined with "
                "starttime, endtime or dtype.", UserWarning)
            st = self.get_waveforms(network, station, location, channel,
                                    starttime, endtime, sds_type=sds_type,
                                    headonly=True, _no_trim_or_merge=True)
        st.sort(keys=['starttime', 'endtime'])
        st.traces = [tr for tr in st
                     if not (tr.stats.endtime < starttime or
                             tr.stats.starttime > endtime)]

        if not st:
            return (0, 1)

        total_duration = endtime - starttime
        # sum up gaps in the middle
        gaps = [gap[6] for gap in st.getGaps()]
        gap_sum = np.sum(gaps)
        gap_count = len(gaps)
        # check if we have a gap at start or end
        earliest = min([tr.stats.starttime for tr in st])
        latest = max([tr.stats.endtime for tr in st])
        if earliest > starttime:
            gap_sum += earliest - starttime
            gap_count += 1
        if latest < endtime:
            gap_sum += endtime - latest
            gap_count += 1

        return (1 - (gap_sum / total_duration), gap_count)

    def _get_current_endtime(self, network, station, location, channel,
                             sds_type=None, stop_time=None):
        """
        Get time of last sample for given stream.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :type stop_time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param stop_time: Time at which the search for data is stopped and
            `None` is returned. If not specified, stops at ``1950-01-01T00``.
        :rtype: :class:`~obspy.core.utcdatetime.UTCDateTime` or `None`
        """
        sds_type = sds_type or self.sds_type

        if not self.has_data(
                network=network, station=station, location=location,
                channel=channel, sds_type=sds_type):
            return None

        stop_time = stop_time or UTCDateTime(1950, 1, 1)
        st = None
        time = UTCDateTime()

        while not st:
            if time < stop_time:
                return None
            filename = self._get_filename(
                network=network, station=station, location=location,
                channel=channel, time=time, sds_type=sds_type)
            if os.path.isfile(filename):
                st = read(filename, format=self.format, headonly=True)
                if st:
                    break
            time -= 24 * 3600

        return max([tr.stats.endtime for tr in st])

    def get_latency(self, network, station, location, channel,
                    sds_type=None, stop_time=None):
        """
        Get time of last sample for given stream.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :type stop_time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param stop_time: Time at which the search for data is stopped and
            `None` is returned. If not specified, stops at ``1950-01-01T00``.
        :rtype: float or ``None``
        :returns: Latency in seconds or ``None`` if no data was encountered
            from current time backwards until ``stop_time``.
        """
        endtime = self._get_current_endtime(
            network=network, station=station, location=location,
            channel=channel, sds_type=sds_type, stop_time=stop_time)

        if endtime is None:
            return endtime

        return UTCDateTime() - endtime

    def has_data(self, network, station, location, channel, sds_type=None):
        """
        Check if specified stream has any data.

        Actually just checks whether a file is encountered in a folder that is
        expected to contain data.

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
        :type location: str
        :param location: Location code of requested data (e.g. "").
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: bool
        """
        sds_type = sds_type or self.sds_type
        pattern = self.FMTSTR.format(
            year="*", network=network, station=station, location=location,
            channel=channel, doy=0, type=sds_type)
        # can not insert wildcard for day-of-year above, so replace it now
        pattern = pattern.rsplit(".", 1)[0] + "*"
        pattern = os.path.join(self.sds_root, pattern)
        if glob.glob(pattern):
            return True
        else:
            return False

    def get_all_nslc(self, sds_type=None):
        """
        Return information on what streams are included in archive.

        Note that this can be very slow on network file systems because every
        single file has to be touched (because available location codes can not
        be discovered from folder structure alone).

        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: list
        :returns: List of (network, station, location, channel) 4-tuples of all
            available streams in archive.
        """
        sds_type = sds_type or self.sds_type
        pattern = self.FMTSTR.format(
            year="*", network="*", station="*", location="*",
            channel="*", doy=0, type=sds_type)
        # can not insert wildcard for day-of-year above, so replace it now
        pattern = pattern.rsplit(".", 1)[0] + "*"
        pattern = os.path.join(self.sds_root, pattern)
        all_files = glob.glob(pattern)
        result = set()
        for file_ in all_files:
            network, station, channel, filename = file_.split("/")[-4:]
            channel = channel.split(".")[0]
            location = filename.split(".")[2]
            result.add((network, station, location, channel))
        return result

    def get_all_stations(self, sds_type=None, strict=True):
        """
        Return information on what stations are included in archive.

        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :type strict: bool
        :param strict: Whether to only regard folders with valid
            network/station code lengths with respect to SEED (i.e. at most 2
            characters in network code, at most 5 characters in station code).
        :rtype: list
        :returns: List of (network, station) 2-tuples of all available stations
            in archive.
        """
        sds_type = sds_type or self.sds_type
        pattern = self.FMTSTR.format(
            year="*", network="*", station="*", location="*",
            channel="*", doy=0, type=sds_type)
        # can not insert wildcard for day-of-year above, so replace it now
        pattern = os.path.join(self.sds_root, pattern)
        pattern = os.path.dirname(os.path.dirname(pattern))
        all_folders = glob.glob(pattern)
        result = set()
        for file_ in all_folders:
            network, station = file_.split("/")[-2:]
            if strict:
                if len(network) > 2 or len(network) > 5:
                    continue
            result.add((network, station))
        return sorted(result)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
