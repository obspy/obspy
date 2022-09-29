# -*- coding: utf-8 -*-
"""
Apollo Lunar Surface Experiments Package (ALSEP) seismometer reader for ObsPy
"""
from collections import deque

import numpy as np

from obspy import Stream, Trace
from obspy.core import Stats

from .assign import assign_alsep_words
from .define import channels, package_id_to_apollo_station, FORMAT_ALSEP_WTN
from .pse.tape import PseTape
from .util import get_utc, check_date, check_sync_code
from .wt.tape import WtnTape, WthTape


def _is_pse(filename):
    """
    Checks whether a file is ALSEP PSE tape or not.

    :type filename: str
    :param filename: ALSEP PSE tape file to be checked.
    :rtype: bool
    :return: ``True`` if an ALSEP PSE tape file.
    """
    header = np.fromfile(filename, dtype='u1', count=16)
    # File has less than 16 characters
    if len(header) != 16:
        return False
    # Tape type: 1 for PSE; 2 for Event; 3 for WTN; 4 for WTH
    tape_type = (header[0] << 8) + header[1]
    if tape_type in [1, 2]:
        apollo_station = (header[2] << 8) + header[3]
        if apollo_station in [11, 12, 14, 15, 16, 17]:
            return True
    return False


def _is_wtn(filename):
    """
    Checks whether a file is ALSEP WTN tape or not.

    :type filename: str
    :param filename: ALSEP WTN tape file to be checked.
    :rtype: bool
    :return: ``True`` if an ALSEP WTN tape file.
    """
    header = np.fromfile(filename, dtype='u1', count=16)
    # File has less than 16 characters
    if len(header) != 16:
        return False
    # Tape type: 1 for PSE; 2 for Event; 3 for WTN; 4 for WTH
    tape_type = (header[0] << 8) + header[1]
    if tape_type != 3:
        return False
    active_stations = [(header[2] & 0x70) >> 4,
                       (header[2] & 0x0e) >> 1,
                       ((header[2] & 0x01) << 2) + (header[3] >> 6),
                       (header[3] & 0x38) >> 3,
                       header[3] & 0x07]
    if all(active_station <= 5 for active_station in active_stations):
        return True
    else:
        return False


def _is_wth(filename):
    """
    Checks whether a file is ALSEP WTH tape or not.

    :type filename: str
    :param filename: ALSEP WTH tape file to be checked.
    :rtype: bool
    :return: ``True`` if an ALSEP WTH tape file.
    """
    header = np.fromfile(filename, dtype='u1', count=16)
    # File has less than 16 characters
    if len(header) != 16:
        return False
    # Tape type: 1 for PSE; 2 for Event; 3 for WTN; 4 for WTH
    tape_type = (header[0] << 8) + header[1]
    if tape_type != 4:
        return False
    active_stations = [(header[2] & 0x70) >> 4,
                       (header[2] & 0x0e) >> 1,
                       ((header[2] & 0x01) << 2) + (header[3] >> 6),
                       (header[3] & 0x38) >> 3,
                       header[3] & 0x07]
    if all(active_station <= 5 for active_station in active_stations):
        return True
    else:
        return False


def _read_pse(filename, headonly=False, year=None, ignore_error=False,
              **kwargs):
    """
    Reads a PSE file and returns ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: PSE file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type year: int, optional
    :param year: Overwrite year if set. The PSE files: pse.a12.10.X requires
        this option due to invalid year. X is as follows:
        91,92,93,94,95,97,98,102,103,104,106,107,108,109, and 111.
    :type ignore_error: bool, optional
    :param ignore_error: Include error frames as much as possible if True.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/pse.a15.1.2.mini")
    """
    stream = Stream()
    pse_tape = PseTape()
    with pse_tape.open(filename) as tape:
        prev_msec_of_year = 0
        prev_spz = None
        trace_data = {}
        for record in tape:
            for frame in record:
                if year is None:
                    year = record.year
                start_time = get_utc(year, frame.msec_of_year)

                if ignore_error is False:
                    if check_date(record.apollo_station, start_time) is False:
                        continue
                    if check_sync_code(frame.barker_code) is False:
                        continue

                seismic_data = \
                    {'start_time': start_time,
                     'apollo_station': record.apollo_station,
                     'channels': assign_alsep_words(frame.alsep_words,
                                                    record.apollo_station,
                                                    record.format,
                                                    frame.frame_count,
                                                    prev_spz)}

                # Check connectivity from previous frame
                prev_spz = None
                count = _get_frame_diff_count(frame.msec_of_year,
                                              prev_msec_of_year)
                # Store values for interpolation to the next frame
                if count == 1:
                    if 'spz' in seismic_data['channels'] and \
                            seismic_data['channels']['spz'][30] is not None:
                        prev_spz = (seismic_data['channels']['spz'][30],
                                    seismic_data['channels']['spz'][31])
                else:
                    _append_trace(stream, trace_data, headonly)

                # Append new data to existing data
                _append_data(trace_data, seismic_data)

                prev_msec_of_year = frame.msec_of_year

        # Append final trace
        _append_trace(stream, trace_data, headonly)
    return stream


def _read_wtn(filename, headonly=False, ignore_error=False, **kwargs):
    """
    Reads a WTN file and returns ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: WTN file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type ignore_error: bool, optional
    :param ignore_error: Include error frames as much as possible if True.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/wtn.1.2.mini")
    """

    stream = Stream()
    wtn_tape = WtnTape()
    with wtn_tape.open(filename) as tape:
        prev_values = None
        prev_msec_of_year = {12: 0, 14: 0, 15: 0, 16: 0, 17: 0}
        trace_data = {}
        for record in tape:
            for frame in record:
                if frame.is_valid() is False:
                    continue

                start_time = get_utc(record.year, frame.msec_of_year)
                apollo_station = \
                    package_id_to_apollo_station[frame.alsep_package_id]

                if ignore_error is False:
                    if check_date(apollo_station, start_time) is False:
                        continue
                    if check_sync_code(frame.barker_code) is False:
                        continue

                # Assign ALSEP words to each seismic data type
                seismic_data = \
                    {'start_time': start_time,
                     'apollo_station': apollo_station,
                     'channels': assign_alsep_words(frame.alsep_words,
                                                    apollo_station,
                                                    FORMAT_ALSEP_WTN,
                                                    frame.frame_count,
                                                    prev_values)}

                # Check connectivity from previous frame
                prev_values = None
                count = \
                    _get_frame_diff_count(frame.msec_of_year,
                                          prev_msec_of_year[apollo_station])

                # Store values for interpolation to the next frame
                if count == 1:
                    if apollo_station != 17:
                        if 'spz' in seismic_data['channels']:
                            if seismic_data['channels']['spz'][30] is not None:
                                prev_values = \
                                    (seismic_data['channels']['spz'][30],
                                     seismic_data['channels']['spz'][31])
                    else:
                        if 'lsg' in seismic_data['channels']:
                            if seismic_data['channels']['lsg'][30] is not None:
                                prev_values = \
                                    (seismic_data['channels']['lsg'][30],
                                     seismic_data['channels']['lsg'][31])
                else:
                    _append_trace(stream, trace_data, headonly)

                # Append new data to existing data
                _append_data(trace_data, seismic_data)

                prev_msec_of_year[apollo_station] = frame.msec_of_year

        # Append final trace
        for data_id in trace_data.keys():
            if len(trace_data[data_id]['data']) > 0:
                _append_trace(stream, trace_data, headonly)
    return stream


def _read_wth(filename, headonly=False, ignore_error=False, **kwargs):
    """
    Reads a WTH file and returns ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: WTH file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :type ignore_error: bool, optional
    :param ignore_error: Include error frames as much as possible if True.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/wth.1.5.mini")
    """

    stream = Stream()
    wth_tape = WthTape()
    with wth_tape.open(filename) as tape:
        prev_msec_of_year = {12: 0, 14: 0, 15: 0, 16: 0, 17: 0}
        trace_data = {}
        for record in tape:
            for frame in record:
                if frame.is_valid() is False:
                    continue

                start_time = get_utc(record.year, frame.msec_of_year)
                apollo_station = \
                    package_id_to_apollo_station[frame.alsep_package_id]

                if ignore_error is False:
                    if check_date(apollo_station, start_time) is False:
                        continue

                seismic_data = {'start_time': start_time,
                                'apollo_station': apollo_station,
                                'channels': {'geo1': frame.geophone[1],
                                             'geo2': frame.geophone[2],
                                             'geo3': frame.geophone[3],
                                             'geo4': frame.geophone[4]}}

                # Check connectivity from previous frame
                count = \
                    _get_frame_diff_count(frame.msec_of_year,
                                          prev_msec_of_year[apollo_station])

                if count != 1:
                    _append_trace(stream, trace_data, headonly)

                # Append new data to existing data
                _append_data(trace_data, seismic_data)

                prev_msec_of_year[apollo_station] = frame.msec_of_year

        # Append final trace
        for data_id in trace_data.keys():
            if len(trace_data[data_id]['data']) > 0:
                _append_trace(stream, trace_data, headonly)
    return stream


def _get_frame_diff_count(curr_msec_of_year, prev_msec_of_year):
    """
    Calculate the time difference in sampling period between two frames

    :type curr_msec_of_year: int
    :param curr_msec_of_year: current frame timestamp in millisecond of year
    :param prev_msec_of_year: previous frame timestamp in millisecond of year
    :rtype: int
    :return: the expected number of frames between two frames
    """
    frame_interval = 640 / 1060.
    time_diff = (curr_msec_of_year - prev_msec_of_year) / 1000.
    return int(np.around(time_diff / frame_interval))


def _append_data(trace_data, seismic_data):
    """
    Append new seismic data to existing data.

    If existing data does not have data type in the new data, the new data
    type is created and stored the new data.
    The start_time parameter is only used when the existing data is empty.

    :type trace_data: dict
    :param trace_data: existing data
    :type seismic_data: dict
    :param seismic_data: seismic data, apollo station number, and start time
    """
    for data_type in seismic_data['channels'].keys():
        data_id = '{network}_{station}_{location}_{channel}'.format(
            network='XA', station=seismic_data['apollo_station'],
            location='', channel=channels[data_type]['channel'])
        if data_id not in trace_data:
            trace_data[data_id] = \
                {'data': deque([]),
                 'apollo_station': seismic_data['apollo_station']}
        if len(trace_data[data_id]['data']) == 0:
            trace_data[data_id]['start_time'] = seismic_data['start_time']
        trace_data[data_id]['data_type'] = data_type
        trace_data[data_id]['data'].extend(
            seismic_data['channels'][data_type])


def _append_trace(stream, data, headonly=False):
    """
    Append data as trace to stream.

    :type stream: obspy.Stream
    :param stream: stream
    :type data: dict
    :param data: seismic data
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    """
    for data_id in data.keys():
        if len(data[data_id]['data']) > 0:
            stats = Stats()
            stats.network = 'XA'
            stats.station = 'S{}'.format(data[data_id]['apollo_station'])
            stats.location = ''
            stats.channel = channels[data[data_id]['data_type']]['channel']
            stats.sampling_rate = \
                channels[data[data_id]['data_type']]['sampling_rate']
            stats.starttime = data[data_id]['start_time']
            stats.npts = len(data[data_id]['data'])
            if headonly is True:
                stream.append(Trace(header=stats))
            else:
                stream.append(Trace(data=np.array(data[data_id]['data']),
                                    header=stats))
            data[data_id]['data'] = deque([])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
