# -*- coding: utf-8 -*-
"""
Routines for handling of Reftek130 packets.

Currently only event header (EH), event trailer (ET) and data (DT) packets are
handled. These three packets have more or less the same meaning in the first 8
bytes of the payload which makes the first 24 bytes the so called extended
header.
"""
import sys
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core.compatibility import from_buffer
from obspy.io.mseed.headers import clibmseed

from .util import (
    _decode_ascii, _parse_long_time, _16_tuple_ascii, _16_tuple_int,
    _16_tuple_float, bcd, bcd_hex,
    bcd_julian_day_string_to_nanoseconds_of_year, bcd_16bit_int, bcd_8bit_hex,
    _get_nanoseconds_for_start_of_year)


class Reftek130UnpackPacketError(ValueError):
    pass


PACKET_TYPES_IMPLEMENTED = ("EH", "ET", "DT")
PACKET_TYPES_NOT_IMPLEMENTED = ("AD", "CD", "DS", "FD", "OM", "SC", "SH")
PACKET_TYPES = PACKET_TYPES_IMPLEMENTED + PACKET_TYPES_NOT_IMPLEMENTED


# The extended header which is the same for EH/ET/DT packets.
# tuples are:
#  - field name
#  - dtype during initial reading
#  - conversion routine (if any)
#  - dtype after conversion
PACKET = [
    ("packet_type", "|S2", None, "S2"),
    ("experiment_number", np.uint8, bcd, np.uint8),
    ("year", np.uint8, bcd, np.uint8),
    ("unit_id", (np.uint8, 2), bcd_hex, "S4"),
    ("time", (np.uint8, 6), bcd_julian_day_string_to_nanoseconds_of_year,
     np.int64),
    ("byte_count", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("packet_sequence", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("event_number", (np.uint8, 2), bcd_16bit_int, np.uint16),
    ("data_stream_number", np.uint8, bcd, np.uint8),
    ("channel_number", np.uint8, bcd, np.uint8),
    ("number_of_samples", (np.uint8, 2), bcd_16bit_int, np.uint32),
    ("flags", np.uint8, None, np.uint8),
    ("data_format", np.uint8, bcd_8bit_hex, "S2"),
    # Temporarily store the payload here.
    ("payload", (np.uint8, 1000), None, (np.uint8, 1000))]


# name, offset, length (bytes) and converter routine for EH/ET packet payload
EH_PAYLOAD = {
    "trigger_time_message": (0, 33, _decode_ascii),
    "time_source": (33, 1, _decode_ascii),
    "time_quality": (34, 1, _decode_ascii),
    "station_name_extension": (35, 1, _decode_ascii),
    "station_name": (36, 4, _decode_ascii),
    "stream_name": (40, 16, _decode_ascii),
    "_reserved_2": (56, 8, _decode_ascii),
    "sampling_rate": (64, 4, float),
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
    "channel_sensor_vpu": (520, 96, _16_tuple_float),
    "channel_sensor_units_code": (616, 16, _16_tuple_ascii),
    "station_channel_number": (632, 48, _16_tuple_int),
    "_reserved_3": (680, 156, _decode_ascii),
    "total_installed_channels": (836, 2, int),
    "station_comment": (838, 40, _decode_ascii),
    "digital_filter_list": (878, 16, _decode_ascii),
    "position": (894, 26, _decode_ascii),
    "reftek_120": (920, 80, None)}


# mseed steim compression is big endian
if sys.byteorder == 'little':
    SWAPFLAG = 1
else:
    SWAPFLAG = 0


class Packet(object):
    _headers = ('experiment_number', 'unit_id', 'byte_count',
                'packet_sequence', 'time')

    @staticmethod
    def from_data(data):
        packet_type = data['packet_type'].decode("ASCII", "ignore")
        if packet_type in ("EH", "ET"):
            return EHPacket(data)
        elif packet_type == "DT":
            return DTPacket(data)
        else:
            msg = "Can not create Reftek packet for packet type '{}'"
            raise NotImplementedError(msg.format(packet_type))

    def __init__(self, data):
        raise NotImplementedError()

    @property
    def type(self):
        return self._data['packet_type'].item()

    def __getattr__(self, name):
        if name in self._headers:
            return self._data[name].item()

    @property
    def nanoseconds(self):
        return self._data['time'].item()

    @property
    def time(self):
        return UTCDateTime(ns=self._data['time'].item())


class EHPacket(Packet):
    __slots__ = ["_data"] + list(EH_PAYLOAD.keys())
    _headers = ('packet_sequence', 'experiment_number', 'unit_id',
                'byte_count', 'time', 'event_number', 'data_stream_number',
                'data_format', 'flags')

    def __init__(self, data):
        self._data = data
        payload = self._data["payload"].tobytes()
        for name, (start, length, converter) in EH_PAYLOAD.items():
            data = payload[start:start + length]
            if converter is not None:
                data = converter(data)
            setattr(self, name, data)

    def _to_dict(self):
        """
        Convert to dictionary structure.
        """
        return {key: getattr(self, key) for key in EH_PAYLOAD.keys()}

    def __str__(self, compact=False):
        if compact:
            sta = (self.station_name.strip() +
                   self.station_name_extension.strip())
            info = ("{:04d} {:2s} {:4s} {:2d} {:4d} {:4d} {:2d} {:2s} "
                    "{:5s}  {:4s}        {!s}").format(
                        self.packet_sequence, self.type.decode(),
                        self.unit_id.decode(), self.experiment_number,
                        self.byte_count, self.event_number,
                        self.data_stream_number, self.data_format.decode(),
                        sta, str(self.sampling_rate)[:4], self.time)
        else:
            info = []
            for key in self._headers:
                value = getattr(self, key)
                if key in ("unit_id", "data_format"):
                    value = value.decode()
                info.append("{}: {}".format(key, value))
            info.append("-" * 20)
            for key in sorted(EH_PAYLOAD.keys()):
                value = getattr(self, key)
                if key in ("trigger_time", "detrigger_time",
                           "first_sample_time", "last_sample_time"):
                    if value is not None:
                        value = UTCDateTime(ns=value)
                info.append("{}: {}".format(key, value))
            info = "{} Packet\n\t{}".format(self.type.decode(),
                                            "\n\t".join(info))
        return info


class DTPacket(Packet):
    __slots__ = ["_data"]
    _headers = ('packet_sequence', 'experiment_number', 'unit_id',
                'byte_count', 'time', 'event_number', 'data_stream_number',
                'channel_number', 'number_of_samples', 'data_format', 'flags')

    def __init__(self, data):
        self._data = data

    def __str__(self, compact=False):
        if compact:
            info = ("{:04d} {:2s} {:4s} {:2d} {:4d} {:4d} {:2d} {:2s} "
                    "           {:2d} {:4d} {!s}").format(
                        self.packet_sequence, self.type.decode(),
                        self.unit_id.decode(), self.experiment_number,
                        self.byte_count, self.event_number,
                        self.data_stream_number, self.data_format.decode(),
                        self.channel_number, self.number_of_samples, self.time)
        else:
            info = []
            for key in self._headers:
                value = getattr(self, key)
                if key in ("unit_id", "data_format"):
                    value = value.decode()
                info.append("{}: {}".format(key, value))
            info = "{} Packet\n\t{}".format(self.type.decode(),
                                            "\n\t".join(info))
        return info


PACKET_INITIAL_UNPACK_DTYPE = np.dtype([
    (name, dtype_initial)
    for name, dtype_initial, converter, dtype_final in PACKET])

PACKET_FINAL_DTYPE = np.dtype([
    (name, dtype_final)
    for name, dtype_initial, converter, dtype_final in PACKET])


def _initial_unpack_packets(bytestring):
    """
    First unpack data with dtype matching itemsize of storage in the reftek
    file, than allocate result array with dtypes for storage of python
    objects/arrays and fill it with the unpacked data.
    """
    if not len(bytestring):
        return np.array([], dtype=PACKET_FINAL_DTYPE)

    if len(bytestring) % 1024 != 0:
        tail = len(bytestring) % 1024
        bytestring = bytestring[:-tail]
        msg = ("Length of data not a multiple of 1024. Data might be "
               "truncated. Dropping {:d} byte(s) at the end.").format(tail)
        warnings.warn(msg)
    data = from_buffer(
        bytestring, dtype=PACKET_INITIAL_UNPACK_DTYPE)
    result = np.empty_like(data, dtype=PACKET_FINAL_DTYPE)

    for name, dtype_initial, converter, dtype_final in PACKET:
        if converter is None:
            result[name][:] = data[name][:]
        else:
            try:
                result[name][:] = converter(data[name])
            except Exception as e:
                raise Reftek130UnpackPacketError(str(e))
    # time unpacking is special and needs some additional work.
    # we need to add the POSIX timestamp of the start of respective year to the
    # already unpacked seconds into the respective year..
    result['time'][:] += [_get_nanoseconds_for_start_of_year(y)
                          for y in result['year']]
    return result


def _unpack_C0_C2_data(packets, encoding):  # noqa
    """
    Unpacks sample data from a packet array that uses 'C0' or 'C2' data
    encoding.

    :type packets: :class:`numpy.ndarray` (dtype ``PACKET_FINAL_DTYPE``)
    :param packets: Array of data packets (``packet_type`` ``'DT'``) from which
        to unpack the sample data (with data encoding 'C0' or 'C2').
    :type encoding: str
    :param encoding: Reftek data encoding as specified in event header (EH)
        packet, either ``'C0'`` or ``'C2'``.
    """
    if encoding == 'C0':
        encoding_bytes = b'C0'
    elif encoding == 'C2':
        encoding_bytes = b'C2'
    else:
        msg = "Unregonized encoding: '{}'".format(encoding)
        raise ValueError(msg)
    if np.any(packets['data_format'] != encoding_bytes):
        differing_formats = np.unique(
            packets[packets['data_format'] !=
                    encoding_bytes]['data_format']).tolist()
        msg = ("Using '{}' data format unpacking routine but some packet(s) "
               "specify other data format(s): {}".format(encoding,
                                                         differing_formats))
        warnings.warn(msg)
    # if the packet array is contiguous in memory (which it generally should
    # be), we can work with the memory address of the first packed data byte
    # and advance it by a fixed offset when moving from one packet to the next
    if packets.flags['C_CONTIGUOUS'] and packets.flags['F_CONTIGUOUS']:
        return _unpack_C0_C2_data_fast(packets, encoding)
    # if the packet array is *not* contiguous in memory, fall back to slightly
    # slower unpacking with looking up the memory position of the first packed
    # byte in each packet individually.
    else:
        return _unpack_C0_C2_data_safe(packets, encoding)


def _unpack_C0_C2_data_fast(packets, encoding):  # noqa
    """
    Unpacks sample data from a packet array that uses 'C0' or 'C2' data
    encoding.

    Unfortunately the whole data cannot be unpacked with one call to
    libmseed as some payloads do not take the full 960 bytes. They are
    thus padded which would results in padded pieces directly in a large
    array and libmseed (understandably) does not support that.

    Thus we resort to *tada* pointer arithmetics in Python ;-) This is
    quite a bit faster then correctly casting to an integer pointer so
    it's worth it.

    Also avoid a data copy.

    Writing this directly in C would be about 3 times as fast so it might
    be worth it.

    :type packets: :class:`numpy.ndarray` (dtype ``PACKET_FINAL_DTYPE``)
    :param packets: Array of data packets (``packet_type`` ``'DT'``) from which
        to unpack the sample data (with data encoding 'C0' or 'C2').
    :type encoding: str
    :param encoding: Reftek data encoding as specified in event header (EH)
        packet, either ``'C0'`` or ``'C2'``.
    """
    if encoding == 'C0':
        decode_steim = clibmseed.msr_decode_steim1
    elif encoding == 'C2':
        decode_steim = clibmseed.msr_decode_steim2
    else:
        msg = "Unregonized encoding: '{}'".format(encoding)
        raise ValueError(msg)
    npts = packets["number_of_samples"].sum()
    unpacked_data = np.empty(npts, dtype=np.int32)
    pos = 0
    s = packets[0]["payload"][40:].ctypes.data
    if len(packets) > 1:
        offset = (
            packets[1]["payload"][40:].ctypes.data - s)
    else:
        offset = 0
    for _npts in packets["number_of_samples"]:
        decode_steim(
            s, 960, _npts, unpacked_data[pos:], _npts, None,
            SWAPFLAG)
        pos += _npts
        s += offset
    return unpacked_data


def _unpack_C0_C2_data_safe(packets, encoding):  # noqa
    """
    Unpacks sample data from a packet array that uses 'C0' or 'C2' data
    encoding.

    If the packet array is *not* contiguous in memory, fall back to slightly
    slower unpacking with looking up the memory position of the first packed
    byte in each packet individually.

    :type packets: :class:`numpy.ndarray` (dtype ``PACKET_FINAL_DTYPE``)
    :param packets: Array of data packets (``packet_type`` ``'DT'``) from which
        to unpack the sample data (with data encoding 'C0' or 'C2').
    :type encoding: str
    :param encoding: Reftek data encoding as specified in event header (EH)
        packet, either ``'C0'`` or ``'C2'``.
    """
    if encoding == 'C0':
        decode_steim = clibmseed.msr_decode_steim1
    elif encoding == 'C2':
        decode_steim = clibmseed.msr_decode_steim2
    else:
        msg = "Unregonized encoding: '{}'".format(encoding)
        raise ValueError(msg)
    npts = packets["number_of_samples"].sum()
    unpacked_data = np.empty(npts, dtype=np.int32)
    pos = 0
    # Sample data starts at byte 40 in the DT packet's
    # payload.
    for p in packets:
        _npts = p["number_of_samples"]
        decode_steim(
            p["payload"][40:].ctypes.data, 960, _npts,
            unpacked_data[pos:], _npts, None, SWAPFLAG)
        pos += _npts
    return unpacked_data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
