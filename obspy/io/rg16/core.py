# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
from obspy.core import Stream, Trace, Stats, UTCDateTime

from .util import _read, _open_file, _read_block, _quick_merge

# blocks are specified as a list of tuples. Each tuple contains the following:
# (name, startbyte, length, format), explanation as follows:
# name - The name, these will be keys in a returned dict
# startbye - the byte position, relative to the block, to start reading
# length - the number of bytes to read
# format - format to interpret the data being read. see rg16.utils.read.


# header block, combines header block one and two
general_header_block = [
    ('channel_sets', 28, 1, 'bcd'),
    ('num_additional_headers', 11, 1, '>i.'),
    ('extended_headers', [30, 37], [1, 2], ['bcd', '>i2']),
    ('external_headers', [31, 39], [1, 3], ['bcd', '>i3']),
    ('record_length', 46, 3, '>i3'),
    ('base_scan', 22, 1, '>i1'),  # https://imgur.com/a/4aneG
]

# channel set header block
channel_header_block = [
    ('chan_num', 1, 1, 'bcd'),
    ('num_channels', 8, 2, 'bcd'),
    ('ru_channel_number', 30, 1, '>i1'),
]

# combines extended header 1, 2, and 3
extended_header_block = [
    ('num_records', 16 + 32, 4, '>i4'),
    ('num_files', 20 + 32, 4, '>i4'),
    ('collection_method', 32 + 15, 1, '>i1'),
    ('line_number', 64, 4, '>i4'),
    ('receiver_point', 68, 4, '>i4'),
    ('point_index', 69, 1, '>i1'),
]

# combines trace header blocks 0 (20 byte) and 1 to 10 (32 byte)
trace_header_block = [
    ('trace_number', 4, 2, 'bcd'),
    ('num_ext_blocks', 9, 1, '>i1'),
    ('line_number', 20 + 0, 3, '>i3'),
    ('point', 20 + 3, 3, '>i3'),
    ('index', 20 + 6, 1, '>i1'),
    ('samples', 20 + 7, 3, '>i3'),
    ('channel_code', 20 + 20, 1, '>i1'),  # https://imgur.com/a/4aneG
    ('trace_count', 20 + 21, 4, '>i4'),
    ('time', 20 + 2 * 32, 8, '>i8'),
]

# since UTCDateTime cannot be compared to np.inf in py27 get a large timestamp
# after which I will be dead (somebody else's problem)
BIG_TS = UTCDateTime('3000-01-01').timestamp

# map sampling rate to band code according to seed standard
BAND_MAP = {2000: 'G', 1000: 'G', 500: 'D', 250: 'D'}

# geophone instrument code
INSTRUMENT_CODE = 'P'

# mapping for "standard_orientation"
STANDARD_COMPONENT_MAP = {'2': 'Z', '3': 'N', '4': 'E'}


@_open_file
def _read_rg16(filename, headonly=False, starttime=None, endtime=None,
               merge=False, contacts_north=False, **kwargs):
    """
    Read Fairfield Nodal's Receiver Gather File Format version 1.6-1.

    :param filename: A path to the file or a buffer of an opened file.
    :type filename: str, buffer
    :param headonly: If True don't read data, only header information.
    :type headonly: bool
    :param starttime: If not None dont read traces that end before starttime.
    :type starttime: optional, obspy.UTCDateTime
    :param endtime: If None None dont read traces that start after endtime.
    :type endtime: optional, obspy.UTCDateTime
    :param merge:
        If True merge contiguous data blocks as they are found. For
        continuous data files having 100,000+ traces this will create
        more manageable streams.
    :type merge: bool
    :param contacts_north:
        Setting this parameter to True indicates the file either contains 1C
        traces or that the instruments were deployed with the gold contact
        terminals facing north. If this parameter is used, it will map the
        components to Z, N, and E (if 3C) as well as correct the polarity for
        the vertical component.
    :type contacts_north: bool
    :return: An ObsPy :class:`~obspy.core.stream.Stream` object.
    """
    # get timestamps
    time1 = UTCDateTime(starttime).timestamp if starttime else 0
    time2 = UTCDateTime(endtime).timestamp if endtime else BIG_TS
    # read general header information
    gheader = _read_block(filename, general_header_block)
    # byte number channel sets start at in file
    chan_set_start = (gheader['num_additional_headers'] + 1) * 32
    # get the byte number the extended headers start
    eheader_start = (gheader['channel_sets']) * 32 + chan_set_start
    # read trace headers
    ex_headers = gheader['extended_headers'] + gheader['external_headers']
    # get byte number trace headers start
    theader_start = eheader_start + (ex_headers * 32)
    # get traces and return stream
    traces = _make_traces(filename, theader_start, gheader, head_only=headonly,
                          starttime=time1, endtime=time2, merge=merge,
                          standard_orientation=contacts_north)
    return Stream(traces=traces)


@_open_file
def _is_rg16(filename, **kwargs):
    """
    Determine if a file or buffer contains an rg16 file.

    :param filename: A path to the file or a buffer of an opened file.
    :type filename: str, buffer
    :return: bool
    """
    try:
        sample_format = _read(filename, 2, 2, 'bcd')
        manufacturer_code = _read(filename, 16, 1, 'bcd')
        version = _read(filename, 42, 2, None)
    except ValueError:  # if file too small
        return False
    con1 = version == b'\x01\x06' and sample_format == 8058
    return con1 and manufacturer_code == 20


def _make_traces(fi, data_block_start, gheader, head_only=False,
                 starttime=None, endtime=None, merge=False,
                 standard_orientation=False):
    """
    Make obspy traces from trace blocks and headers.
    """
    traces = []  # list to store traces
    trace_position = data_block_start
    while True:  # read traces until parser falls of the end of file
        try:
            theader = _read_block(fi, trace_header_block, trace_position)
        except ValueError:  # this is the end, my only friend, the end
            break
        # get stats
        stats = _make_stats(theader, gheader, standard_orientation)
        # expected jump to next start position
        jumps = stats.npts * 4 + theader['num_ext_blocks'] * 32 + 20
        assert jumps != 0
        # if wrong starttime / endtime just keep going and update position
        if stats.endtime < starttime or stats.starttime > endtime:
            trace_position += jumps
            continue
        if head_only:  # empty np array for head only
            data = np.array([])
        else:  # else read data
            data_start = trace_position + 20 + theader['num_ext_blocks'] * 32
            data = _read(fi, data_start, theader['samples'] * 4, '>f4',
                         np.float32)
            if standard_orientation and stats.channel[-1] == 'Z':
                data = -data
        traces.append(Trace(data=data, header=stats))
        trace_position += jumps
    if merge:
        traces = _quick_merge(traces)
    return traces


def _make_stats(theader, gheader, standard_orientation):
    """
    Make Stats object from information from several blocks.
    """
    sampling_rate = int(1000. / (gheader['base_scan'] / 16.))

    # get channel code
    component = str(theader['channel_code'])
    if standard_orientation:
        component = STANDARD_COMPONENT_MAP[component]
    chan = BAND_MAP[sampling_rate] + INSTRUMENT_CODE + component

    statsdict = dict(
        starttime=UTCDateTime(theader['time'] / 1000000.),
        sampling_rate=sampling_rate,
        npts=theader['samples'],
        network=str(theader['line_number']),
        station=str(theader['point']),
        location=str(theader['index']),
        channel=chan,
    )
    return Stats(statsdict)


# Note: I am leaving this function in the code as it may be needed if the
# "read until end of file" approach needs to be revised. Also, there are
# some non-obvious peculiarities about the rg16 files I have worked with
# that are handled by this function.

# def _get_num_traces(fi, byte_start, gheader, eheader):
#     """
#     Get the number of traces contained in this file by reading trace sets.
#
#     Note: This function was created because multiplying channel_sets in
#     the general header by num_records in the extended header doesn't work for
#     some larger files.
#     """
#     channel_sets = gheader['channel_sets']
#     num_records = eheader['num_records']
#
#
#     # try reading the channel_header blocks. This is seems to be correct
#     # when there are millions of records in the file
#     channel_dicts = [read_block(fi, channel_header_block, byte_start + x*32)
#                      for x in range(channel_sets)]
#     num_traces1 = np.sum([x['num_channels'] for x in channel_dicts])
#
#     # try multiplying general_header and num_records. This seems to be right
#     # when there arent that many treaces in the file
#     num_traces2 = channel_sets * num_records
#
#     return max(num_traces1, num_traces2)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
