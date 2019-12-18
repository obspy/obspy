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
import os
import re
import shutil
import tempfile
import traceback
import warnings
from datetime import timedelta

import numpy as np
from matplotlib.dates import num2date

from obspy import Stream, read, UTCDateTime
from obspy.core.stream import _headonly_warning_msg
from obspy.core.util.misc import BAND_CODE
from obspy.imaging.scripts.scan import scan, Scanner
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
        sds_root = os.path.abspath(sds_root)
        if not os.path.isdir(sds_root):
            msg = ("SDS root is not a local directory: " + sds_root)
            raise IOError(msg)
        self.sds_root = sds_root
        self.sds_type = sds_type
        self.format = format and format.upper() or format
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

    def _extract_missing_data(self, filenames):
        """
        Extracts information on data that is missing in SDS archive and could
        be retrieved from given local files.

        :type filenames: list of str
        :param filenames: Files to extract data from.
        """
        # assemble information on the files with new data for the SDS archive
        data = {}
        times_min = {}
        times_max = {}
        filenames_to_check_sds = {}

        for filename in filenames:
            for tr in read(filename, headonly=True):
                # remember overall earliest/latest data time per SEED ID
                times_min[tr.id] = min(
                    tr.stats.starttime,
                    times_min.get(tr.id, tr.stats.starttime))
                times_max[tr.id] = max(
                    tr.stats.endtime + tr.stats.delta,
                    times_max.get(tr.id, tr.stats.endtime))
                # remember that file contains data for respective SEED ID
                data.setdefault(tr.id, {}).setdefault(filename, []).append(tr)
                # remember which files we need to check for gaps in SDS archive
                filenames_to_check_sds.setdefault(tr.id, set()).update(
                    self._get_filenames(
                        network=tr.stats.network, station=tr.stats.station,
                        location=tr.stats.location, channel=tr.stats.channel,
                        starttime=tr.stats.starttime, endtime=tr.stats.endtime,
                        only_existing_files=False))

        # assemble information on gaps in the SDS archive that might be covered
        # by new data
        gaps = {}
        earliest = {}
        latest = {}
        for seed_id, filenames_ in filenames_to_check_sds.items():
            gaps.setdefault(seed_id, [])
            earliest.setdefault(seed_id, np.inf)
            latest.setdefault(seed_id, -np.inf)
            scanner = scan(
                paths=filenames_, format=self.format, recursive=False,
                ignore_links=False, starttime=times_min[seed_id],
                verbose=False, endtime=times_max[seed_id], seed_ids=[seed_id],
                plot=False, print_gaps=False)
            if scanner is None:
                continue
            for id_, info in scanner._info.items():
                gaps[id_].extend([(UTCDateTime(num2date(start_)),
                                   UTCDateTime(num2date(end_)))
                                  for start_, end_ in info["gaps"]])
                if len(info["data_startends_compressed"]):
                    earliest_ = min([start_ for start_, _ in
                                    info["data_startends_compressed"]])
                    earliest[id_] = min(earliest[id_], earliest_)
                    latest_ = max([end_ for _, end_ in
                                   info["data_startends_compressed"]])
                    latest[id_] = max(latest[id_], latest_)

        timeranges = {}
        for id_ in gaps:
            start = earliest[id_]
            end = latest[id_]
            if start == np.inf:
                start = None
                end = None
            else:
                start = UTCDateTime(num2date(start))
                end = UTCDateTime(num2date(end))
            timeranges[id_] = (start, end)

        # set a full-extent gap for all IDs that we did not encounter existing
        # data in the time range of new data
        for id_ in gaps:
            if timeranges[id_] == (None, None):
                gaps[id_] = [(times_min[seed_id], times_max[seed_id])]

        return data, gaps

    def add_data_to_archive(self, filenames, only_missing=True, backup=True,
                            verbose=True, plot=True):
        """
        Adds data from given files to SDS Archive.

        Currently only implemented for SDS archive in MSEED format.

        :type filenames: list of str
        :param filenames: Files to check for new data to add to archive.
        :type only_missing: bool
        :param only_missing: Whether to only add data missing in archive
            (slicing the input data to gaps present in archive) or just add all
            input data to archive without any checks of archive contents (might
            lead to duplicate data in archive).
        :type backup: bool
        :param backup: Whether to backup original version of changed files in
            SDS archive to a temporary directory.
        :type verbose: bool
        :param verbose: Whether to print info messages on performed operations.
        :type plot: bool or str
        :param plot: Whether to save a before/after comparison plot to an image
            file. ``False`` for no image output, ``True`` for output to a
            temporary file or a filename.
        :rtype: str
        :returns: Textual information of new data added to SDS archive.
        """
        format = self.format
        now = UTCDateTime()
        now_str = now.strftime("%Y%m%d%H%M%S")

        if format != "MSEED":
            msg = ("Currently only implemented for SDS Archive with format "
                   "'MSEED'.")
            raise NotImplementedError(msg)

        self._backupdir = None
        tmp_prefix = "obspy-sds-backup-{}-".format(now_str)
        # maps original file paths (that were appended to) to backup file paths
        # (or `None` if backup option is not selected)
        changed_files = {}
        new_files = set()
        if plot:
            scanner = Scanner(verbose=False, recursive=False,
                              ignore_links=False)

        new_data_string = []

        def _handle_gap(id, start, end):
            """
            """
            backupdir = self._backupdir

            data_files = set()
            for filename, traces in data.get(id, {}).items():
                for tr in traces:
                    if start and tr.stats.endtime < start:
                        continue
                    if end and tr.stats.starttime > end:
                        continue
                    data_files.add(filename)
            for filename in data_files:
                st = read(filename, starttime=start, endtime=end,
                          sourcename=id, details=True)
                st = st.select(id=id).trim(start, end,
                                           nearest_sample=False)
                for tr in st:
                    if tr.stats.endtime == end:
                        tr.data = tr.data[:-1]
                st_dict = self._split_stream_by_filenames(st)
                if not st_dict:
                    continue
                for filename_, st_ in st_dict.items():
                    if not st_:
                        continue
                    # backup original file
                    backupfile = None
                    if (backup and filename_ not in changed_files and
                            filename_ not in new_files and
                            os.path.exists(filename_)):
                        if backupdir is None:
                            self._backupdir = tempfile.mkdtemp(
                                prefix=tmp_prefix)
                            backupdir = self._backupdir
                        backupfile = os.path.join(
                            backupdir,
                            self._filename_strip_sds_root(filename_))
                        target_dir = os.path.dirname(backupfile)
                        try:
                            if not os.path.isdir(target_dir):
                                os.makedirs(target_dir)
                            shutil.copy2(filename_, backupfile)
                        except Exception:
                            err_msg = \
                                traceback.format_exc(0).\
                                splitlines()[-1]
                            info = ""
                            if changed_files:
                                info = (
                                    " The following files have so far "
                                    "been modified: {}").format(
                                        changed_files.keys())
                            msg = (
                                "Backup option chosen and backup of "
                                "file '{}' to '{}' failed ({}). "
                                "Aborting appending to SDS archive.{}")
                            msg = msg.format(filename_, backupfile,
                                             err_msg, info)
                            raise Exception(msg)
                    # scan original file for before/after comparison
                    if plot and filename_ not in changed_files:
                        scanner.parse(filename_)
                    # check if we can convert data to int32 without
                    # changing it
                    for tr in st_:
                        try:
                            data_int32 = tr.data.astype(np.int32)
                            np.testing.assert_array_equal(
                                tr.data, data_int32)
                        except:
                            pass
                        else:
                            tr.data = data_int32
                    # now append the missing segment to the file
                    target_dir = os.path.dirname(filename_)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    with open(filename_, "ab") as fh:
                        st_.write(fh, format=format)

                    new_data_string.extend(str(st_).splitlines()[1:])
                    if filename_ in new_files:
                        pass
                    elif filename_ in changed_files:
                        pass
                    elif backupfile:
                        changed_files[filename_] = backupfile
                    else:
                        new_files.add(filename_)

        if only_missing:
            data, gaps = self._extract_missing_data(filenames)
            # handle gaps inside data
            for id, gaps_ in gaps.items():
                for start, end in gaps_:
                    _handle_gap(id, start, end)
        else:
            # XXX TODO
            raise NotImplementedError()

        if verbose:
            if changed_files:
                print("The following files have been appended to:")
                print("\n".join("\t" + filename
                                for filename in sorted(changed_files)))
            if new_files:
                print("The following new files have been created:")
                print("\n".join("\t" + filename
                                for filename in sorted(new_files)))
            if backup and changed_files:
                print("Backups of original files have been stored "
                      "in: {}".format(self._backupdir))

        plot_output_file = None
        if plot and (changed_files or new_files):
            if plot is True:
                fd, plot_output_file = tempfile.mkstemp(
                    prefix=tmp_prefix, suffix=".png")
                os.close(fd)
            else:
                plot_output_file = plot
            # change seed id's of "before" data so that they don't clash with
            # the "after" data
            for id_ in scanner.data.keys():
                scanner.data[id_ + " (before)"] = scanner.data.pop(id_)
                scanner.samp_int[id_ + " (before)"] = scanner.samp_int.pop(id_)
            # now scan modified files
            for path in changed_files.keys():
                scanner.parse(path)
            for id_ in scanner.data.keys():
                if id_.endswith(" (before)"):
                    continue
                scanner.data[id_ + "  (after)"] = scanner.data.pop(id_)
                scanner.samp_int[id_ + "  (after)"] = scanner.samp_int.pop(id_)
            # also scan new files
            for path in new_files:
                scanner.parse(path)
            for id_ in scanner.data.keys():
                if id_.endswith(" (before)") or id_.endswith("  (after)"):
                    continue
                scanner.data[id_ + "    (new)"] = scanner.data.pop(id_)
                scanner.samp_int[id_ + "    (new)"] = scanner.samp_int.pop(id_)
            # plot everything together
            scanner.plot(show=False, outfile=plot_output_file)
            if verbose:
                print(("Before/after comparison plot saved as: {}").format(
                    plot_output_file))

        new_data_string = "\n".join(sorted(new_data_string))
        if verbose and new_data_string:
            print("New data added to archive:")
            print(new_data_string)

        return (new_data_string, changed_files, self._backupdir,
                plot_output_file)

    def _filename_strip_sds_root(self, filename):
        """
        Strip SDS root from a full-path filename in the SDS archive.
        """
        return re.sub(r'^{}{}*'.format(self.sds_root, os.sep), '', filename)


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
