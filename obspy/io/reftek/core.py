# -*- coding: utf-8 -*-
"""
REFTEK130 read support, core routines.
"""
import copy
import io
import os
from pathlib import Path
import warnings

import numpy as np

from obspy import Trace, Stream, UTCDateTime
from obspy.core.util.obspy_types import ObsPyException

from .packet import (Packet, EHPacket, _initial_unpack_packets, PACKET_TYPES,
                     PACKET_TYPES_IMPLEMENTED, PACKET_FINAL_DTYPE,
                     Reftek130UnpackPacketError, _unpack_C0_C2_data)


NOW = UTCDateTime()


class Reftek130Exception(ObsPyException):
    pass


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
    if not Path(filename).is_file():
        return False
    filesize = Path(filename).stat().st_size
    # check if overall file size is a multiple of 1024
    if filesize < 1024 or filesize % 1024 != 0:
        return False

    with open(filename, 'rb') as fp:
        # check first 20 expected packets' type header field
        for i in range(20):
            packet_type = fp.read(2).decode("ASCII", "replace")
            if not packet_type:
                # reached end of file..
                break
            if packet_type not in PACKET_TYPES:
                return False
            fp.seek(1022, os.SEEK_CUR)
    return True


def _read_reftek130(filename, network="", location="", component_codes=None,
                    headonly=False, verbose=False,
                    sort_permuted_package_sequence=False, **kwargs):
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
    :type sort_permuted_package_sequence: bool
    :param sort_permuted_package_sequence: Determines whether or not the
        package list is sorted when a permuted package sequence is encountered.
        This should only be used if problems occur with files that have a
        permuted package sequence (showing the related warning message).
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
    try:
        rt130 = Reftek130.from_file(filename)
        st = rt130.to_stream(
            network=network, location=location,
            component_codes=component_codes, headonly=headonly,
            verbose=verbose,
            sort_permuted_package_sequence=sort_permuted_package_sequence)
        st.merge(-1)
        st.sort()
        return st
    except Reftek130UnpackPacketError:
        msg = ("Unable to read file '{}' as a Reftek130 file. Please contact "
               "developers if you think this is a valid Reftek130 file.")
        raise Reftek130Exception(msg.format(filename))


class Reftek130(object):
    _info_header = "Reftek130 ({:d} packets{})"
    _info_compact_header = [
        "Packet Sequence  Byte Count  Data Fmt  Sampling Rate      Time",
        "  | Packet Type   |  Event #  | Station | Channel #         |",
        "  |   |  Unit ID  |    | Data Stream #  |   |  # of samples |",
        "  |   |   |  Exper.#   |   |  |  |      |   |    |          |"]
    _info_compact_footer = ("(detailed packet information with: "
                            "'print(Reftek130.__str__(compact=False))')")

    def __init__(self):
        self._data = np.array([], dtype=PACKET_FINAL_DTYPE)
        self._filename = None

    def __str__(self, compact=True):
        filename = self._filename and ', file: {}'.format(self._filename) or ''
        info = [self._info_header.format(len(self._data), filename)]
        if compact:
            info += copy.deepcopy(self._info_compact_header)
            info[0] = info[0].format(len(self._data))
            for data in self._data:
                info.append(Packet.from_data(data).__str__(compact=True))
            info.append(self._info_compact_footer)
        else:
            for data in self._data:
                info.append(str(Packet.from_data(data)))
        return "\n".join(info)

    @staticmethod
    def from_file(filename):
        with io.open(filename, "rb") as fh:
            string = fh.read()
        rt = Reftek130()
        rt._data = _initial_unpack_packets(string)
        rt._filename = filename
        return rt

    def check_packet_sequence_and_sort(self, sort_permuted_package_sequence):
        """
        Checks if packet sequence is ordered. If not, shows a warning and sorts
        packets by packet sequence if ``sort_permuted_package_sequence=True``.
        This should ensure that data (DT) packets are properly enclosed by the
        appropriate event header/trailer (EH/ET) packets.
        """
        diff = np.diff(self._data['packet_sequence'].astype(np.int16))
        # rollover from 9999 to 0 is not a packet sequence jump..
        jump = (diff < 1) & (diff != -9999)
        if np.any(jump):
            msg = ("Detected permuted packet sequence, sorting.")
            warnings.warn(msg)
            if sort_permuted_package_sequence:
                self._data.sort(order=[
                    key for key in ("packet_sequence", "time")])

    def check_packet_sequence_contiguous(self):
        """
        Checks if packet sequence is contiguous, i.e. without missing packets
        in between. Currently raises if that is the case because this case is
        not covered by test data yet.
        """
        diff = np.diff(self._data['packet_sequence'].astype(np.int16))
        if np.any(diff > 1):
            msg = ("Detected a non-contiguous packet sequence!")
            warnings.warn(msg)

    def drop_not_implemented_packet_types(self):
        """
        Checks if there are packets of a type that is currently not implemented
        and drop them showing a warning message.
        """
        is_implemented = np.in1d(
            self._data['packet_type'],
            [x.encode() for x in PACKET_TYPES_IMPLEMENTED])
        # if all packets are of a type that is implemented, the nothing to do..
        if np.all(is_implemented):
            return
        # otherwise reduce packet list to what is implemented and warn
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
        self._data = self._data[is_implemented]

    def to_stream(self, network="", location="", component_codes=None,
                  headonly=False, verbose=False,
                  sort_permuted_package_sequence=False):
        """
        :type headonly: bool
        :param headonly: Determines whether or not to unpack the data or just
            read the headers.
        """
        if verbose:
            print(self)
        if not len(self._data):
            msg = "No packet data in Reftek130 object (file: {})"
            raise Reftek130Exception(msg.format(self._filename))
        self.check_packet_sequence_and_sort(sort_permuted_package_sequence)
        self.check_packet_sequence_contiguous()
        self.drop_not_implemented_packet_types()
        if not len(self._data):
            msg = ("No packet data left in Reftek130 object after dropping "
                   "non-implemented packets (file: {})").format(self._filename)
            raise Reftek130Exception(msg)
        st = Stream()
        for event_number in np.unique(self._data['event_number']):
            data = self._data[self._data['event_number'] == event_number]
            # we should have exactly one EH and one ET packet, truncated data
            # sometimes misses the header or trailer packet.
            eh_packets = data[data['packet_type'] == b"EH"]
            et_packets = data[data['packet_type'] == b"ET"]
            if len(eh_packets) == 0 and len(et_packets) == 0:
                msg = ("Reftek data contains data packets without "
                       "corresponding header or trailer packet.")
                raise Reftek130Exception(msg)
            if len(eh_packets) > 1 or len(et_packets) > 1:
                msg = ("Reftek data contains data packets with multiple "
                       "corresponding header or trailer packets.")
                raise Reftek130Exception(msg)
            if len(eh_packets) != 1:
                msg = ("No event header (EH) packets in packet sequence. "
                       "File might be truncated.")
                warnings.warn(msg)
            if len(et_packets) != 1:
                msg = ("No event trailer (ET) packets in packet sequence. "
                       "File might be truncated.")
                warnings.warn(msg)
            # use either the EH or ET packet, they have the same content (only
            # trigger stop time is not in EH)
            if len(eh_packets):
                eh = EHPacket(eh_packets[0])
            else:
                eh = EHPacket(et_packets[0])
            # only C0, C2, 16, 32 encodings supported right now
            if eh.data_format == b"C0":
                encoding = 'C0'
            elif eh.data_format == b"C2":
                encoding = 'C2'
            elif eh.data_format == b"16":
                encoding = '16'
            elif eh.data_format == b"32":
                encoding = '32'
            else:
                msg = ("Reftek data encoding '{}' not implemented yet. Please "
                       "open an issue on GitHub and provide a small (< 50kb) "
                       "test file.").format(eh.data_format)
                raise NotImplementedError(msg)
            header = {
                "network": network,
                "station": (eh.station_name +
                            eh.station_name_extension).strip(),
                "location": location, "sampling_rate": eh.sampling_rate,
                "reftek130": eh._to_dict()}
            delta = 1.0 / eh.sampling_rate
            delta_nanoseconds = int(delta * 1e9)
            inds_dt = data['packet_type'] == b"DT"
            data_channels = np.unique(data[inds_dt]['channel_number'])
            for channel_number in data_channels:
                inds = data['channel_number'] == channel_number
                # channel number of EH/ET packets also equals zero (one of the
                # three unused bytes in the extended header of EH/ET packets)
                inds &= data['packet_type'] == b"DT"
                packets = data[inds]

                # split into contiguous blocks, i.e. find gaps. packet sequence
                # was sorted already..
                endtimes = (
                    packets[:-1]["time"] +
                    packets[:-1]["number_of_samples"].astype(np.int64) *
                    delta_nanoseconds)
                # check if next starttime matches seamless to last chunk
                # 1e-3 seconds == 1e6 nanoseconds is the smallest time
                # difference reftek130 format can represent, so anything larger
                # or equal means a gap/overlap.
                time_diffs_milliseconds_abs = np.abs(
                    packets[1:]["time"] - endtimes) / 1000000
                gaps = time_diffs_milliseconds_abs >= 1
                if np.any(gaps):
                    gap_split_indices = np.nonzero(gaps)[0] + 1
                    contiguous = np.array_split(packets, gap_split_indices)
                else:
                    contiguous = [packets]

                for packets_ in contiguous:
                    starttime = packets_[0]['time']

                    if headonly:
                        sample_data = np.array([], dtype=np.int32)
                        npts = packets_["number_of_samples"].sum()
                    else:
                        if encoding in ('C0', 'C2'):
                            sample_data = _unpack_C0_C2_data(packets_,
                                                             encoding)
                        elif encoding in ('16', '32'):
                            # rt130 stores in big endian
                            dtype = {'16': '>i2', '32': '>i4'}[encoding]
                            # just fix endianness and use correct dtype
                            sample_data = np.require(
                                packets_['payload'],
                                requirements=['C_CONTIGUOUS'])
                            # either int16 or int32
                            sample_data = sample_data.view(dtype)
                            # account for number of samples, i.e. some packets
                            # might not use the full payload size but have
                            # empty parts at the end that need to be cut away
                            number_of_samples_max = sample_data.shape[1]
                            sample_data = sample_data.flatten()
                            # go through packets starting at the back,
                            # otherwise indices of later packets would change
                            # while looping
                            for ind, num_samps in reversed([
                                    (ind, num_samps) for ind, num_samps in
                                    enumerate(packets_["number_of_samples"])
                                    if num_samps != number_of_samples_max]):
                                # looping backwards we can easily find the
                                # start of each packet, since the earlier
                                # packets are still untouched and at maximum
                                # sample length in our big array with all
                                # packets
                                start_of_packet = ind * number_of_samples_max
                                start_empty_part = start_of_packet + num_samps
                                end_empty_part = (start_of_packet +
                                                  number_of_samples_max)
                                sample_data = np.delete(
                                    sample_data,
                                    slice(start_empty_part, end_empty_part))
                        npts = len(sample_data)

                    tr = Trace(data=sample_data, header=copy.deepcopy(header))
                    # channel number is not included in the EH/ET packet
                    # payload, so add it to stats as well..
                    tr.stats.reftek130['channel_number'] = channel_number
                    if headonly:
                        tr.stats.npts = npts
                    tr.stats.starttime = UTCDateTime(ns=starttime)
                    # if component codes were explicitly provided, use them
                    # together with the stream label
                    if component_codes is not None:
                        tr.stats.channel = (eh.stream_name.strip() +
                                            component_codes[channel_number])
                    # otherwise check if channel code is set for the given
                    # channel (seems to be not the case usually)
                    elif eh.channel_code[channel_number] is not None:
                        tr.stats.channel = eh.channel_code[channel_number]
                    # otherwise fall back to using the stream label together
                    # with the number of the channel in the file (starting with
                    # 0, as Z-1-2 is common use for data streams not oriented
                    # against North)
                    else:
                        msg = ("No channel code specified in the data file "
                               "and no component codes specified. Using "
                               "stream label and number of channel in file as "
                               "channel codes.")
                        warnings.warn(msg)
                        tr.stats.channel = (
                            eh.stream_name.strip() + str(channel_number))
                    # check if endtime of trace is consistent
                    t_last = packets_[-1]['time']
                    npts_last = packets_[-1]['number_of_samples']
                    try:
                        if not headonly:
                            assert npts == len(sample_data)
                        if npts_last:
                            assert tr.stats.endtime == UTCDateTime(
                                ns=t_last) + (npts_last - 1) * delta
                        if npts:
                            assert tr.stats.endtime == (
                                tr.stats.starttime + (npts - 1) * delta)
                    except AssertionError:
                        msg = ("Reftek file has a trace with an inconsistent "
                               "endtime or number of samples. Please open an "
                               "issue on GitHub and provide your file for"
                               "testing.")
                        raise Reftek130Exception(msg)
                    st += tr

        return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
