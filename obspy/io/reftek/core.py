# -*- coding: utf-8 -*-
"""
REFTEK130 read support.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import io
import os
import warnings

import numpy as np

from obspy import Trace, Stream, UTCDateTime
from obspy.io.mseed.headers import clibmseed

from .packet import (Packet, EHPacket, _initial_unpack_packets, PACKET_TYPES,
                     PACKET_TYPES_IMPLEMENTED, PACKET_FINAL_DTYPE)


NOW = UTCDateTime()


def _is_reftek130(filename):
    """
    Checks whether a file is REFTEK130 format or not.

    :type filename: str
    :param filename: REFTEK130 file to be checked.
    :rtype: bool
    :return: ``True`` if a REFTEK130 file.

    Checks if overall length of file is consistent (i.e. multiple of 1024
    bytes) and checks for valid packet type identifiers in the first 20
    expected packet positions.
    """
    if not os.path.isfile(filename):
        return False
    filesize = os.stat(filename).st_size
    # check if overall file size is a multiple of 1024
    if filesize < 1024 or filesize % 1024 != 0:
        return False

    with open(filename, 'rb') as fp:
        # check first 20 expected packets' type header field
        while True:
            packet_type = fp.read(2).decode("ASCII", "ignore")
            if not packet_type:
                break
            if packet_type not in PACKET_TYPES:
                return False
            fp.seek(1022, os.SEEK_CUR)
    return True


def _read_reftek130(filename, network="", location="", component_codes=None,
                    headonly=False, **kwargs):
    """
    Read a REFTEK130 file into an ObsPy Stream.

    :type filename: str
    :param filename: REFTEK130 file to be read.
    :type network: str
    :param network: Network code to fill in for all data (network code is not
        stored in EH/ET/DT packets).
    :type location: str
    :param location: Location code to fill in for all data (network code is not
        stored in EH/ET/DT packets).
    :type component_codes: list
    :param component_codes: Iterable of single-character component codes (e.g.
        ``['Z', 'N', 'E']``) to be appended to two-character stream name parsed
        from event header packet (e.g. ``'HH'``) for each of the channels in
        the data (e.g. to make the channel codes in a three channel data file
        to ``'HHZ'``, ``'HHN'``, ``'HHE'`` in the created stream object).
    :type headonly: bool
    :param headonly: Determines whether or not to unpack the data or just
        read the headers.
    :rtype: :class:`~obspy.core.stream.Stream`
    """
    # Reftek 130 data format stores only the last two digits of the year.  We
    # currently assume that 00-49 are years 2000-2049 and 50-99 are years
    # 2050-2099. We deliberately raise an exception if the current year will
    # become 2050 (just in case someone really still uses this code then.. ;-)
    # At that point the year would probably have to be explicitly specified
    # when reading data to be on the safe side.
    if NOW.year > 2050:
        raise NotImplementedError()
    return Reftek130.from_file(filename).to_stream(
        network=network, location=location, component_codes=component_codes,
        headonly=headonly)


class Reftek130(object):
    def __init__(self):
        self._data = np.array([], dtype=PACKET_FINAL_DTYPE)

    def __str__(self, compact=True):
        if compact:
            info = [
                "Reftek130 ({:d} packets)".format(len(self._data)),
                "Packet Sequence  Byte Count  Data Fmt  Sampling Rate      "
                "Time",
                "  | Packet Type   |  Event #  | Station | Channel #         "
                "|",
                "  |   |  Unit ID  |    | Data Stream #  |   |  # of samples "
                "|",
                "  |   |   |  Exper.#   |   |  |  |      |   |    |          "
                "|"]
            for data in self._data:
                info.append(Packet.from_data(data).__str__(compact=True))
            info.append("(detailed packet information with: "
                        "'print(Reftek130.__str__(compact=False))')")
        else:
            info = ["Reftek130 ({:d} packets)".format(len(self._data))]
            for data in self._data:
                info.append(str(Packet.from_data(data)))
        return "\n".join(info)

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
            self._data.sort(order=native_str("packet_sequence"))

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
        if self._data['packet_type'][0] != b"EH":
            is_eh = self._data['packet_type'] == b"EH"
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
        is_et = self._data['packet_type'] == b"ET"
        if not np.any(is_et):
            msg = ("No event trailer (ET) packets in packet sequence. "
                   "File might be truncated.")
            warnings.warn(msg)
            return
        if self._data['packet_type'][-1] != b"ET":
            first_et = np.nonzero(is_et)[0][0]
            msg = ("Last packet in sequence is not an event trailer (ET) "
                   "packet. Dropped {:d} packet(s) at the end after "
                   "encountering the first ET packet in sequence.").format(
                        len(self._data) - first_et + 1)
            warnings.warn(msg)
            self._data = self._data[:first_et+1]
            return

    def drop_not_implemented_packet_types(self):
        """
        Checks if there are packets of a type that is currently not implemented
        and drop them showing a warning message.
        """
        is_implemented = np.in1d(
            self._data['packet_type'],
            [x.encode() for x in PACKET_TYPES_IMPLEMENTED])
        if not np.all(is_implemented):
            not_implemented = np.invert(is_implemented)
            count_not_implemented = not_implemented.sum()
            types_not_implemented = np.unique(
                self._data['packet_type'][not_implemented])
            msg = ("Encountered some packets of types that are not "
                   "implemented yet (types: {}). Dropped {:d} packets "
                   "overall.")
            msg = msg.format(types_not_implemented.tolist(),
                             count_not_implemented)
            warnings.warn(msg)
            return
        self._data = self._data[is_implemented]

    def to_stream(self, network="", location="", component_codes=None,
                  headonly=False):
        """
        :type headonly: bool
        :param headonly: Determines whether or not to unpack the data or just
            read the headers.
        """
        self.check_packet_sequence_and_sort()
        self.check_packet_sequence_contiguous()
        self.drop_not_implemented_packet_types()
        self.drop_leading_non_eh_packets()
        self.drop_trailing_packets_after_et_packet()
        eh = EHPacket(self._data[0])
        # only "C0" encoding supported right now
        if eh.data_format != b"C0":
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
            inds &= self._data['packet_type'] == b"DT"
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
                gap_split_indices = np.nonzero(gaps)[0] + 1
                contiguous = np.array_split(packets, gap_split_indices)
            else:
                contiguous = [packets]

            for packets_ in contiguous:
                starttime = packets_[0]['time']

                # Unfortunately the whole data cannot be unpacked with one
                # call to libmseed as some payloads do not take the full 960
                # bytes. They are thus padded which would results in padded
                # pieces directly in a large array and libmseed
                # (understandably) does not support that.
                #
                # Thus we resort to *tada* pointer arithmetics in Python ;-)
                # This is quite a bit faster then correctly casting to an
                # integer pointer so its worth it.
                #
                # Also avoid a data copy.
                #
                # Writing this directly in C would be about 3 times as fast so
                # it might be worth it.
                npts = packets_["number_of_samples"].sum()
                if headonly:
                    data = np.array([], dtype=np.int32)
                else:
                    data = np.empty(npts, dtype=np.int32)
                    pos = 0
                    s = packets_[0]["payload"][40:].ctypes.data
                    if len(packets_) > 1:
                        offset = packets_[1]["payload"][40:].ctypes.data - s
                    else:
                        offset = 0
                    for p in packets_:
                        _npts = p["number_of_samples"]
                        clibmseed.msr_decode_steim1(
                            s, 960, _npts, data[pos:], _npts, None, 1)
                        pos += _npts
                        s += offset

                tr = Trace(data=data, header=header.copy())
                if headonly:
                    tr.stats.npts = npts
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
                t_last = packets_[-1]['time']
                npts_last = packets_[-1]['number_of_samples']
                try:
                    if not headonly:
                        assert npts == len(data)
                    if npts_last:
                        assert tr.stats.endtime == UTCDateTime(
                            t_last + (npts_last - 1) * delta)
                    if npts:
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
