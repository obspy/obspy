# -*- coding: utf-8 -*-
"""
obspy.clients.filesystem.sds - read support for SeisComP Data Structure
=======================================================================
This module provides read support for data stored locally in a SeisComP Data
Structure (SDS) directory structure.

The directory and file layout of SDS is defined as::

    <SDSdir>/YEAR/NET/STA/CHAN.TYPE/NET.STA.LOC.CHAN.TYPE.YEAR.DAY

These fields are defined by SDS as follows::

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
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import glob
import io
import os
import re
import warnings
from datetime import timedelta

import numpy as np

from obspy import Stream, read, UTCDateTime
from obspy.core.stream import _headonly_warning_msg
from obspy.core.util.misc import BAND_CODE
from obspy.imaging.scripts.scan import scan
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError


SDS_FMTSTR = os.path.join(
    "{year}", "{network}", "{station}", "{channel}.{sds_type}",
    "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}")
FORMAT_STR_PLACEHOLDER_REGEX = r"{(\w+?)?([!:].*?)?}"


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
        self.format = format and format.upper()
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
        :param kwargs: Additional kwargs that get passed on to
            :func:`~obspy.core.stream.read` internally, mostly for internal
            low-level purposes used by other methods.
        :rtype: :class:`~obspy.core.stream.Stream`
        """
        if starttime >= endtime:
            msg = ("'endtime' must be after 'starttime'.")
            raise ValueError(msg)
        sds_type = sds_type or self.sds_type

        seed_pattern = ".".join((network, station, location, channel))

        st = Stream()
        full_paths = self._get_filenames(
            network=network, station=station, location=location,
            channel=channel, starttime=starttime, endtime=endtime,
            sds_type=sds_type)
        for full_path in full_paths:
            try:
                st += read(full_path, format=self.format, starttime=starttime,
                           endtime=endtime, sourcename=seed_pattern, **kwargs)
            except ObsPyMSEEDFilesizeTooSmallError:
                # just ignore small MSEED files, in use cases working with
                # near-realtime data these are usually just being created right
                # at request time, e.g. when fetching current data right after
                # midnight
                continue

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
                       endtime, sds_type=None, only_existing_files=True):
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
        :type only_existing_files: bool
        :param only_existing_files: Whether to only return filenames of
            existing files or not. If True, globbing is performed and
            wildcards can be used in ``network`` and other fields.
        :rtype: list of str
        """
        if not only_existing_files:
            for field in (network, station, location, channel):
                if glob.has_magic(field):
                    msg = (
                        "No wildcards allowed with 'only_existing_files=False'"
                        " (input was: '{}')").format(field)
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

        full_paths = set()
        for year, doy in year_doy:
            filename = self.FMTSTR.format(
                network=network, station=station, location=location,
                channel=channel, year=year, doy=doy, sds_type=sds_type)
            full_path = os.path.join(self.sds_root, filename)
            if only_existing_files:
                full_paths = full_paths.union(glob.glob(full_path))
            else:
                full_paths.add(full_path)

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
            channel=channel, year=time.year, doy=time.julday,
            sds_type=sds_type)
        return os.path.join(self.sds_root, filename)

    def _split_stream_by_filenames(self, stream, sds_type=None):
        """
        Split stream into dictionary mapping by filenames in SDS archive.

        :type stream: str
        :param stream: Input stream to split up.
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: dict
        """
        sds_type = sds_type or self.sds_type
        ids = set([tr.id for tr in stream])
        dict_ = {}
        for id_ in ids:
            network, station, location, channel = id_.split(".")
            st_ = stream.select(id=id_)
            start = min([tr.stats.starttime for tr in st_])
            end = max([tr.stats.endtime for tr in st_])
            filenames = self._get_filenames(
                network=network, station=station, location=location,
                channel=channel, starttime=start, endtime=end,
                only_existing_files=False)
            for filename in filenames:
                start, end = self._filename_to_time_range(filename)
                st__ = st_.slice(start, end, nearest_sample=False)
                # leave out last sample if it is exactly on the boundary.
                # it belongs to the next file in that case.
                for tr in st__:
                    if tr.stats.endtime == end:
                        tr.data = tr.data[:-1]
                # the above can lead to empty traces.. remove those
                st__.traces = [tr for tr in st__.traces if len(tr.data)]
                for tr in st__:
                    if tr.stats.endtime == end:
                        tr.data = tr.data[:-1]
                if st__:
                    dict_[filename] = st__
        return dict_

    def _filename_to_time_range(self, filename):
        """
        Get expected start and end time of data stored in given filename (full
        path).

        >>> client = Client(sds_root="/tmp")
        >>> t = UTCDateTime("2016-05-18T14:12:43.682261Z")
        >>> filename = client._get_filename("NE", "STA", "LO", "CHA", t)
        >>> print(filename)
        /tmp/2016/NE/STA/CHA.D/NE.STA.LO.CHA.D.2016.139
        >>> client._filename_to_time_range(filename)
        (UTCDateTime(2016, 5, 18, 0, 0), UTCDateTime(2016, 5, 19, 0, 0))
        >>> filename = "/tmp/2016/NE/STA/CHA.D/NE.STA.LO.CHA.D.2016.366"
        >>> client._filename_to_time_range(filename)
        (UTCDateTime(2016, 12, 31, 0, 0), UTCDateTime(2017, 1, 1, 0, 0))

        :type filename: str
        :param filename: Filename to get expected start and end time for.
        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: dict
        """
        pattern = os.path.join(self.sds_root, self.FMTSTR)
        group_map = {i: groups[0] for i, groups in
                     enumerate(re.findall(FORMAT_STR_PLACEHOLDER_REGEX,
                                          pattern))}
        dict_ = _parse_path_to_dict(filename, pattern, group_map)
        starttime = UTCDateTime(year=dict_["year"], julday=dict_["doy"])
        endtime = starttime + 24 * 3600
        return (starttime, endtime)

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
                "ignore", _headonly_warning_msg, UserWarning,
                "obspy.core.stream")
            st = self.get_waveforms(network, station, location, channel,
                                    starttime, endtime, sds_type=sds_type,
                                    headonly=True, _no_trim_or_merge=True)
        # even if the warning was silently caught and not shown it gets
        # registered in the __warningregistry__ and will not be shown
        # subsequently in a place were it's not caught
        # see https://bugs.python.org/issue4180
        # see e.g. http://blog.ionelmc.ro/2013/06/26/testing-python-warnings/
        try:
            from obspy.core.stream import __warningregistry__ as \
                stream_warningregistry
        except ImportError:
            # import error means no warning has been issued from
            # obspy.core.stream before, so nothing to do.
            pass
        else:
            for key in list(stream_warningregistry.keys()):
                if key[0] == _headonly_warning_msg:
                    stream_warningregistry.pop(key)
        st.sort(keys=['starttime', 'endtime'])
        st.traces = [tr for tr in st
                     if not (tr.stats.endtime < starttime or
                             tr.stats.starttime > endtime)]

        if not st:
            return (0, 1)

        total_duration = endtime - starttime
        # sum up gaps in the middle
        gaps = [gap[6] for gap in st.get_gaps()]
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

        ``None`` is returned if no data at all is encountered when going
        backwards until `stop_time` (defaults to Jan 1st 1950).

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
            ``None`` is returned. If not specified, stops at ``1950-01-01T00``.
        :rtype: :class:`~obspy.core.utcdatetime.UTCDateTime` or ``None``
        """
        sds_type = sds_type or self.sds_type

        seed_pattern = ".".join((network, station, location, channel))

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
                try:
                    st = read(filename, format=self.format, headonly=True,
                              sourcename=seed_pattern)
                except ObsPyMSEEDFilesizeTooSmallError:
                    # just ignore small MSEED files, in use cases working with
                    # near-realtime data these are usually just being created
                    # right at request time, e.g. when fetching current data
                    # right after midnight
                    st = None
                else:
                    st = st.select(network=network, station=station,
                                   location=location, channel=channel)
                if st:
                    break
            time -= 24 * 3600

        return max([tr.stats.endtime for tr in st])

    def get_latency(self, network, station, location, channel,
                    sds_type=None, stop_time=None):
        """
        Get latency for given stream, i.e. difference of current time and
        latest available data for stream in SDS archive. ``None`` is returned
        if no data at all is encountered when going backwards until
        `stop_time` (defaults to Jan 1st 1950).

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
            ``None`` is returned. If not specified, stops at ``1950-01-01T00``.
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

        pattern = re.sub(
            FORMAT_STR_PLACEHOLDER_REGEX,
            _wildcarded_except(["network", "station", "location", "channel",
                                "sds_type"]),
            self.FMTSTR)
        pattern = pattern.format(
            network=network, station=station, location=location,
            channel=channel, sds_type=sds_type)
        pattern = os.path.join(self.sds_root, pattern)
        if glob.glob(pattern):
            return True
        else:
            return False

    def get_all_nslc(self, sds_type=None, datetime=None):
        """
        Return information on what streams are included in archive.

        Note that this can be very slow on network file systems because every
        single file has to be touched (because available location codes can not
        be discovered from folder structure alone).

        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Only return all streams that have data at specified
            time (checks if file exists that should have the data, i.e. streams
            might be returned that have data on the same day but not at exactly
            this point in time).
        :rtype: list
        :returns: List of (network, station, location, channel) 4-tuples of all
            available streams in archive.
        """
        sds_type = sds_type or self.sds_type
        result = set()
        # wildcarded pattern to match all files of interest
        if datetime is None:
            pattern = re.sub(
                FORMAT_STR_PLACEHOLDER_REGEX,
                _wildcarded_except(["sds_type"]),
                self.FMTSTR).format(sds_type=sds_type)
            pattern = os.path.join(self.sds_root, pattern)
        else:
            pattern = self._get_filename("*", "*", "*", "*", datetime)
        all_files = glob.glob(pattern)
        # set up inverse regex to extract kwargs/values from full paths
        pattern_ = os.path.join(self.sds_root, self.FMTSTR)
        group_map = {i: groups[0] for i, groups in
                     enumerate(re.findall(FORMAT_STR_PLACEHOLDER_REGEX,
                                          pattern_))}
        for file_ in all_files:
            dict_ = _parse_path_to_dict(file_, pattern_, group_map)
            try:
                network = dict_["network"]
                station = dict_["station"]
                location = dict_["location"]
                channel = dict_["channel"]
            except KeyError as e:
                msg = (
                    "Failed to extract key from pattern '{}' in path "
                    "'{}': {}").format(pattern, file_, e)
                warnings.warn(msg)
                continue
            result.add((network, station, location, channel))
        return sorted(result)

    def get_all_stations(self, sds_type=None):
        """
        Return information on what stations are included in archive.

        This method assumes that network/station combinations can be discovered
        from the folder structure alone (as opposed to the filenames).

        :type sds_type: str
        :param sds_type: Override SDS data type identifier that was specified
            during client initialization.
        :rtype: list
        :returns: List of (network, station) 2-tuples of all available stations
            in archive.
        """
        sds_type = sds_type or self.sds_type
        result = set()
        # wildcarded pattern to match all files of interest
        fmtstr = os.path.dirname(self.FMTSTR)
        pattern = re.sub(
            FORMAT_STR_PLACEHOLDER_REGEX,
            _wildcarded_except(["sds_type"]),
            fmtstr).format(sds_type=sds_type)
        pattern = os.path.join(self.sds_root, pattern)
        all_files = glob.glob(pattern)
        # set up inverse regex to extract kwargs/values from full paths
        pattern_ = os.path.join(self.sds_root, fmtstr)
        group_map = {i: groups[0] for i, groups in
                     enumerate(re.findall(FORMAT_STR_PLACEHOLDER_REGEX,
                                          pattern_))}
        for file_ in all_files:
            dict_ = _parse_path_to_dict(file_, pattern_, group_map)
            try:
                network = dict_["network"]
                station = dict_["station"]
            except KeyError as e:
                msg = (
                    "Failed to extract key from pattern '{}' in path "
                    "'{}': {}").format(pattern_, file_, e)
                warnings.warn(msg)
                continue
            result.add((network, station))
        return sorted(result)

    def extract_missing_data(self, filenames, output_folder, plot=True):
        """
        Extracts data that is missing in SDS archive from given local files.

        :type filenames: list
        :param filenames: Files to extract data from.
        :type output_folder: str
        :param output_folder: Output folder to save extracted data to.
        :type plot: bool
        :param plot: Whether to save plots and obspy-scan npz files with
            comparison of data present in SDS archive versus new data.
        """
        # assemble information on the files with new data for the SDS archive
        data = {}
        times_min = {}
        times_max = {}
        filenames_to_check_SDS = {}
        for filename in filenames:
            # read file and store info about traces
            for tr in read(filename, headonly=True):
                # remember overall earliest/latest data time per SEED ID
                times_min[tr.id] = min(
                    tr.stats.starttime,
                    times_min.get(tr.id, tr.stats.starttime))
                times_max[tr.id] = max(
                    tr.stats.endtime + tr.stats.delta,
                    times_max.get(tr.id, tr.stats.endtime))
                # remember that file contains data for respective SEED ID
                data.setdefault(tr.id, set()).add(filename)
                # remember which files we need to check for gaps in SDS archive
                filenames_to_check_SDS.setdefault(tr.id, set()).update(
                    self._get_filenames(
                        network=tr.stats.network, station=tr.stats.station,
                        location=tr.stats.location, channel=tr.stats.channel,
                        starttime=tr.stats.starttime, endtime=tr.stats.endtime,
                        only_existing_files=False))

        # assemble information on gaps in the SDS archive that might be covered
        # by new data
        for seed_id, filenames_ in filenames_to_check_SDS.items():
            _, gaps, _ = scan(
                paths=filenames_, format=self.format, recursive=False,
                ignore_links=False, starttime=times_min[seed_id], quiet=True,
                endtime=times_max[seed_id], seed_ids=[seed_id], plot=False)

            st = Stream()
            for _seed_id, start, end in gaps:
                if _seed_id != seed_id:
                    continue
                for filename in data[seed_id]:
                    st_ = read(filename, starttime=start, endtime=end,
                               sourcename=seed_id)
                    st += st_.select(id=seed_id)
            st.merge(-1)
            outfile = os.path.join(output_folder, seed_id + ".mseed")
            st.write(outfile, "MSEED")

            if plot:
                for tr in st:
                    tr.stats.location = "XX"

                with io.BytesIO() as tmp:
                    st.write(tmp, format="MSEED")
                    tmp.seek(0)
                    filenames_.add(tmp)
                    outfile_plot = os.path.join(output_folder,
                                                seed_id + ".png")
                    outfile_npz = os.path.join(output_folder, seed_id + ".npz")
                    scan(paths=filenames_, format=None, recursive=False,
                         quiet=True, ignore_links=False,
                         starttime=times_min[seed_id],
                         endtime=times_max[seed_id], seed_ids=[seed_id, tr.id],
                         plot=outfile_plot, npz_output=outfile_npz)


def _wildcarded_except(exclude=[]):
    """
    Function factory for :mod:`re` ``repl`` functions used in :func:`re.sub``,
    replacing all format string place holders with ``*`` wildcards, except
    named fields as specified in ``exclude``.
    """
    def _wildcarded(match):
        if match.group(1) in exclude:
            return match.group(0)
        return "*"
    return _wildcarded


def _parse_path_to_dict(path, pattern, group_map):
    # escape special regex characters "." and "\"
    # in principle we should escape all special characters in Python regex:
    #    . ^ $ * + ? { } [ ] \ | ( )
    # and also only replace them if outside of format string placeholders..
    regex = re.sub(r'([\.\\])', r'\\\1', pattern)
    # replace each format string placeholder with a regex group, matching
    # alphanumerics. append end-of-line otherwise the last non-greedy match
    # doesn't catch anything if it's at the end of the regex
    regex = re.sub(FORMAT_STR_PLACEHOLDER_REGEX, r'(\\w*?)', regex) + "$"
    match = re.match(regex, path)
    if match is None:
        return None
    result = {}
    for i, value in enumerate(match.groups()):
        key = group_map[i]
        if key in result:
            if result[key] != value:
                msg = (
                    "Failed to parse path '{}': Mismatching information "
                    "for key '{}' from pattern '{}'.").format(
                        path, key, pattern)
                warnings.warn(msg)
                return None
        result[key] = value
    for key in ("year", "doy"):
        if key in result:
            result[key] = int(result[key])
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
