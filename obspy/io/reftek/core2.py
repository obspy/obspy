# -*- coding: utf-8 -*-
"""
REFTEK130 read support.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import calendar
import codecs
import datetime
import io
import warnings

import numpy as np

from obspy import Trace, Stream, UTCDateTime
from obspy.io.mseed.util import _unpack_steim_1


NOW = UTCDateTime()


_timegm_cache = {}


def _get_timestamp_for_start_of_year(year):
    if year < 50:
        year += 2000
    else:
        year += 1900

    try:
        t = _timegm_cache[year]
    except KeyError:
        t = calendar.timegm(datetime.datetime(year, 1, 1).utctimetuple())
        _timegm_cache[year] = t
    return t


def _timestrings_to_seconds(timestrings):
    """
    Helper routine to convert timestrings of form "DDDHHMMSSsss" to array of
    floating point seconds.

    :param timestring: numpy.ndarray
    :rtype: numpy.ndarray
    """
    # split up the time string into tuple of
    # (day of year, hours, minutes, seconds, milliseconds), still as string
    seconds = [(string[:3], string[3:5], string[5:7],
                string[7:9], string[9:]) for string in timestrings]
    seconds = np.array(seconds, dtype="S3").astype(np.float64)
    # now scale the columns of the array, so that everything is in seconds
    seconds[:, 0] -= 1
    seconds[:, 0] *= 86400
    seconds[:, 1] *= 3600
    seconds[:, 2] *= 60
    seconds[:, 4] *= 1e-3
    # sum up days, hours, minutes etc. for every row of the array
    seconds = seconds.sum(axis=1)
    return seconds


def bcd(_i):
    return (_i >> 4 & 0xF).astype(np.uint32) * 10 + (_i & 0xF)


def bcd_16bit_int(_i):
    _i = bcd(_i)
    return _i[::, 0] * 100 + _i[::, 1]


def bcd_hex(_i):
    result = [codecs.encode(chars, "hex_codec").decode("ASCII").upper()
              for chars in _i]
    return np.array(result)


def bcd_8bit_hex(_i):
    return np.array(["{:X}".format(x) for x in _i], dtype="|S2")


def bcd_julian_day_string_to_seconds_of_year(_i):
    timestrings = bcd_hex(_i)
    return _timestrings_to_seconds(timestrings)


# The extended header which is the same for EH/ET/DT packets.
# tuples are:
#  - field name
#  - dtype during initial reading
#  - conversion routine (if any)
#  - dtype after conversion
PACKET = [
    ("packet_type", native_str("|S2"), None, native_str("S2")),
    ("experiment_number", np.uint8, bcd, np.uint8),
    ("year", np.uint8, bcd, np.uint8),
    ("unit_id", (np.uint8, 2), bcd_hex, native_str("S4")),
    ("time", (np.uint8, 6), bcd_julian_day_string_to_seconds_of_year,
     np.float64),
    ("byte_count", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("packet_sequence", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("event_number", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("data_stream_number", np.uint8, bcd, np.uint8),
    ("channel_number", np.uint8, bcd, np.uint8),
    ("number_of_samples", (np.uint8, 2), bcd_16bit_int, np.uint32),
    ("flags", np.uint8, None, np.uint8),
    ("data_format", np.uint8, bcd_8bit_hex, native_str("S2")),
    # Temporarily store the payload here.
    ("payload", (np.uint8, 1000), None, (np.uint8, 1000)),
]


packet_initial_unpack_dtype = np.dtype([
    (native_str(name), dtype_initial)
    for name, dtype_initial, converter, dtype_final in PACKET])

packet_final_dtype = np.dtype([
    (native_str(name), dtype_final)
    for name, dtype_initial, converter, dtype_final in PACKET])


def _initial_unpack_packets(bytestring):
    data = np.fromstring(
        bytestring, dtype=packet_initial_unpack_dtype)
    result = np.empty_like(data, dtype=packet_final_dtype)

    for name, dtype_initial, converter, dtype_final in PACKET:
        if converter is None:
            result[name][:] = data[name][:]
            continue
        result[name][:] = converter(data[name])
    # time unpacking is special and needs some additional work.
    # we need to add the POSIX timestamp of the start of respective year to the
    # already unpacked seconds into the respective year..
    result['time'][:] += [_get_timestamp_for_start_of_year(y)
                          for y in result['year']]

    return result


class Reftek130(object):
    def __init__(self, filename=None):
        pass

    @staticmethod
    def from_file(filename):
        with io.open(filename, "rb") as fh:
            string = fh.read()
        rt = Reftek130()
        rt._data = _initial_unpack_packets(string)
        return rt

    def check_packet_sequence_and_sort(self):
        """
        Checks if packet sequence is ordered. If not, shows a warning and sorts
        packets by packet sequence. This should ensure that data (DT) packets
        are properly enclosed by the appropriate event header/trailer (EH/ET)
        packets.
        """
        if np.any(np.diff(self._data['packet_sequence']) - 1):
            msg = ("Detected permuted packet sequence, sorting.")
            warnings.warn(msg)
            self._data.sort(order="packet_sequence")

    def check_packet_sequence_contiguous(self):
        """
        Checks if packet sequence is contiguous, i.e. without missing packets
        in between. Currently raises if that is the case because this case is
        not covered by test data yet.
        """
        if np.any(np.diff(self._data['packet_sequence']) - 1 != 0):
            msg = ("Detected a non-contiguous packet sequence, this is not "
                   "yet tested, please provide an example file for testing.")
            raise NotImplementedError(msg)

    def drop_leading_non_eh_packets(self):
        """
        Checks if first packet is an event header (EH) packet. Drop any other
        packets before the first EH packet.
        """
        if self._data['packet_type'][0] != "EH":
            is_eh = self._data['packet_type'] == "EH"
            if not np.any(is_eh):
                msg = ("No event header (EH) packets in packet sequence.")
                raise NotImplementedError(msg)
            first_eh = np.nonzero(is_eh)[0][0]
            msg = ("First packet in sequence is not an event header (EH) "
                   "packet. Dropped {:d} packet(s) at the start until first "
                   "EH packet in sequence.").format(first_eh)
            warnings.warn(msg)
            self._data = self._data[first_eh:]

    def drop_trailing_packets_after_et_packet(self):
        """
        Checks if last packet is an event trailer (ET) packet. Drop any other
        packets after the first ET packet. Warn if no ET packet is present.
        """
        is_et = self._data['packet_type'] == "ET"
        if not np.any(is_et):
            msg = ("No event trailer (ET) packets in packet sequence. "
                   "File might be truncated.")
            warnings.warn(msg)
            return
        if self._data['packet_type'][-1] != "ET":
            first_et = np.nonzero(is_et)[0][0]
            msg = ("Last packet in sequence is not an event trailer (ET) "
                   "packet. Dropped {:d} packet(s) at the end after "
                   "encountering the first ET packet in sequence.").format(
                        len(self._data) - first_et + 1)
            warnings.warn(msg)
            self._data = self._data[:first_et+1]
            return

    def to_stream(self, network="", location="", component_codes=None,
                  **kwargs):
        self.check_packet_sequence_and_sort()
        self.check_packet_sequence_contiguous()
        self.drop_leading_non_eh_packets()
        self.drop_trailing_packets_after_et_packet()
        eh = EHPacket(self._data[0])
        # only "C0" encoding supported right now
        if eh.data_format != "C0":
            msg = ("Reftek data encoding '{}' not implemented yet. Please "
                   "open an issue on GitHub and provide a small (< 50kb) "
                   "test file.").format(eh.type)
            raise NotImplementedError(msg)
        header = {
            "network": network,
            "station": (eh.station_name + eh.station_name_extension).strip(),
            "location": location, "sampling_rate": eh.sampling_rate,
            "reftek130": eh._to_dict()}
        delta = 1.0 / eh.sampling_rate
        st = Stream()
        for channel_number in np.unique(self._data['channel_number']):
            inds = self._data['channel_number'] == channel_number
            # channel number of EH/ET packets also equals zero (one of the
            # three unused bytes in the extended header of EH/ET packets)
            inds &= self._data['packet_type'] == "DT"
            packets = self._data[inds]

            # split into contiguous blocks, i.e. find gaps. packet sequence was
            # sorted already..
            endtimes = (packets[:-1]["time"] +
                        packets[:-1]["number_of_samples"] * delta)
            # check if next starttime matches seamless to last chunk
            # 1e-3 seconds == 1 millisecond is the smallest time difference
            # reftek130 format can represent, so anything larger or equal
            # means a gap/overlap.
            # for now be conservative and check even more rigorous against
            # 1e-4 to be on the safe side, but in the gapless data example
            # the differences are always 0 or -2e-7 (for POSIX timestamps
            # of order 1e9) which seems like a floating point accuracy
            # issue for np.float64.
            gaps = np.abs(packets[1:]["time"] - endtimes) > 1e-4
            if np.any(gaps):
                gap_split_indices = np.nonzero(gaps) + 1
                contiguous = np.split_array(gap_split_indices)
            else:
                contiguous = [packets]

            for packets_ in contiguous:
                starttime = packets_[0]['time']
                data = []
                npts = 0
                for p in packets_:
                    piece = _unpack_steim_1(
                        data_string=p['payload'][40:].tobytes(),
                        npts=p['number_of_samples'], swapflag=1)
                    data.append(piece)
                    npts += p['number_of_samples']
                data = np.hstack(data)

                tr = Trace(data=data, header=header.copy())
                tr.stats.starttime = UTCDateTime(starttime)
                # if component codes were explicitly provided, use them
                # together with the stream label
                if component_codes is not None:
                    tr.stats.channel = (eh.stream_name.strip() +
                                        component_codes[channel_number])
                # otherwise check if channel code is set for the given channel
                # (seems to be not the case usually)
                elif eh.channel_code[channel_number] is not None:
                    tr.stats.channel = eh.channel_code[channel_number]
                # otherwise fall back to using the stream label together with
                # the number of the channel in the file (starting with 0, as
                # Z-1-2 is common use for data streams not oriented against
                # North)
                else:
                    msg = ("No channel code specified in the data file and no "
                           "component codes specified. Using stream label and "
                           "number of channel in file as channel codes.")
                    warnings.warn(msg)
                    tr.stats.channel = (
                        eh.stream_name.strip() + str(channel_number))
                # check if endtime of trace is consistent
                t_last = packets[-1]['time']
                npts_last = packets[-1]['number_of_samples']
                try:
                    assert npts == len(data)
                    assert tr.stats.endtime == UTCDateTime(
                        t_last + (npts_last - 1) * delta)
                    assert tr.stats.endtime == UTCDateTime(
                        tr.stats.starttime + (npts - 1) * delta)
                except AssertionError:
                    msg = ("Reftek file has a trace with an inconsistent "
                           "endtime or number of samples. Please open an "
                           "issue on GitHub and provide your file for"
                           "testing.")
                    raise Exception(msg)
                st += tr

        return st


def _decode_ascii(chars):
    return chars.decode("ASCII")


def _parse_long_time(time_bytestring, decode=True):
    if decode:
        time_string = time_bytestring.decode()
    else:
        time_string = time_bytestring
    if not time_string.strip():
        return None
    time_string, milliseconds = time_string[:-3], int(time_string[-3:])
    return (UTCDateTime.strptime(time_string, '%Y%j%H%M%S') +
            1e-3 * milliseconds)


def _16_tuple_ascii(bytestring):
    item_count = 16
    chars = bytestring.decode("ASCII")
    if len(chars) % item_count != 0:
        raise NotImplementedError("Should not happen, contact developers.")
    item_size = int(len(chars) / item_count)
    result = []
    for i in range(item_count):
        chars_ = chars[i*item_size:(i+1)*item_size]
        result.append(chars_.strip() or None)
    return tuple(result)


def _16_tuple_int(bytestring):
    ascii_tuple = _16_tuple_ascii(bytestring)
    result = []
    for chars in ascii_tuple:
        if chars is None or not chars.strip():
            result.append(None)
            continue
        result.append(int(chars))
    return tuple(result)


EH_PAYLOAD = {
    "trigger_time_message": (0, 33, _decode_ascii),
    "time_source": (33, 1, _decode_ascii),
    "time_quality": (34, 1, _decode_ascii),
    "station_name_extension": (35, 1, _decode_ascii),
    "station_name": (36, 4, _decode_ascii),
    "stream_name": (40, 16, _decode_ascii),
    "_reserved_2": (56, 8, _decode_ascii),
    "sampling_rate": (64, 4, int),
    "trigger_type": (68, 4, _decode_ascii),
    "trigger_time": (72, 16, _parse_long_time),
    "first_sample_time": (88, 16, _parse_long_time),
    "detrigger_time": (104, 16, _parse_long_time),
    "last_sample_time": (120, 16, _parse_long_time),
    "channel_adjusted_nominal_bit_weights": (136, 128, _16_tuple_ascii),
    "channel_true_bit_weights": (264, 128, _16_tuple_ascii),
    "channel_gain_code": (392, 16, _16_tuple_ascii),
    "channel_ad_resolution_code": (408, 16, _16_tuple_ascii),
    "channel_fsa_code": (424, 16, _16_tuple_ascii),
    "channel_code": (440, 64, _16_tuple_ascii),
    "channel_sensor_fsa_code": (504, 16, _16_tuple_ascii),
    "channel_sensor_vpu": (520, 96, _16_tuple_int),
    "channel_sensor_units_code": (616, 16, _16_tuple_ascii),
    "station_channel_number": (632, 48, _16_tuple_int),
    "_reserved_3": (680, 156, _decode_ascii),
    "total_installed_channels": (836, 2, int),
    "station_comment": (838, 40, _decode_ascii),
    "digital_filter_list": (878, 16, _decode_ascii),
    "position": (894, 26, _decode_ascii),
    "reftek_120": (920, 80, None)}


class EHPacket(object):
    __slots__ = ["_data"] + EH_PAYLOAD.keys()
    _headers = ('experiment_number', 'unit_id', 'byte_count',
                'packet_sequence', 'time', 'event_number',
                'data_stream_number', 'data_format', 'flags')

    def __init__(self, data):
        self._data = data
        payload = self._data["payload"].tobytes()
        for name, (start, length, converter) in EH_PAYLOAD.items():
            data = payload[start:start+length]
            if converter is not None:
                data = converter(data)
            setattr(self, name, data)

    @property
    def type(self):
        return self._data['packet_type'].item()

    def __getattr__(self, name):
        if name in self._headers:
            return self._data[name].item()

    @property
    def timestamp(self):
        return self._data['time'].item()

    @property
    def time(self):
        return UTCDateTime(self._data['time'].item())

    def _to_dict(self):
        """
        Convert to dictionary structure.
        """
        return {key: getattr(self, key) for key in EH_PAYLOAD.keys()}


def _read_reftek130(filename):
    if NOW.year > 2050:
        raise NotImplementedError()
    return Reftek130.from_file(filename).to_stream()


if __name__ == "__main__":
    filename = "_helpers/000000000_0036EE80.rt130"
    rt = Reftek130.from_file(filename)
    st = rt.to_stream()
    from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
