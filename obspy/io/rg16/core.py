# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
from obspy.core import Stream, Trace, Stats, UTCDateTime

from .util import read, open_file, read_block, quick_merge

# --------------------- define specs of needed blocks


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

# default channel mapping
COMPONENT_MAP = {2: 'Z', 3: '1', 4: '2'}

# map sampling rate to band code according to seed standard
BAND_MAP = {2000: 'G', 1000: 'G', 500: 'D', 250: 'D'}

INSTRUMENT_CODE = 'P'


# ------------------- read and format check functions


@open_file
def read_rg16(fi, headonly=False, starttime=None, endtime=None, merge=False,
              component_map=None, **kwargs):
    """
    Read fairfield nodal's Receiver Gather File Format version 1.6-1.

    :param fi: A path to the file to read or a buffer of an opened file.
    :type fi: str, buffer
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
    :return: An ObsPy :class:`~obspy.core.stream.Stream` object.
    :param component_map:
        A mapping from the component codes used in rg16 (2, 3, 4) to desired
        component code based on deployment orientation.
        The default is (Z, 1, 2). See https://imgur.com/a/4aneG.
    """
    if not is_rg16(fi):
        raise ValueError('read_fcnt was not passed a Fairfield RG 1.6 file')
    # get timestamps
    time1 = UTCDateTime(starttime).timestamp if starttime else 0
    time2 = UTCDateTime(endtime).timestamp if endtime else BIG_TS
    # read general header information
    gheader = read_block(fi, general_header_block)
    # byte number channel sets start at in file
    chan_set_start = (gheader['num_additional_headers'] + 1) * 32
    # get the byte number the extended headers start
    eheader_start = (gheader['channel_sets']) * 32 + chan_set_start
    # read trace headers
    ex_headers = gheader['extended_headers'] + gheader['external_headers']
    # get byte number trace headers start
    theader_start = eheader_start + (ex_headers * 32)
    # get traces and return stream
    assert component_map is None or set(component_map) == set(COMPONENT_MAP)
    traces = _make_traces(fi, theader_start, gheader, head_only=headonly,
                          starttime=time1, endtime=time2, merge=merge,
                          channel_mapping=component_map)
    return Stream(traces=traces)


@open_file
def is_rg16(fi, **kwargs):
    """
    Determine if a file or buffer contains an rg16 file.

    :param fi: A path to the file to read or a buffer of an opened file.
    :type fi: str, buffer
    :return: bool
    """
    try:
        fi.seek(0)
        sample_format = read(fi, 2, 2, 'bcd')
        manufacturer_code = read(fi, 16, 1, 'bcd')
        version = read(fi, 42, 2, None)
    except ValueError:  # if file too small
        return False
    con1 = version == b'\x01\x06' and sample_format == 8058
    return con1 and manufacturer_code == 20


# ------------ helper functions for formatting specific blocks


def _make_traces(fi, data_block_start, gheader, head_only=False,
                 starttime=None, endtime=None, merge=False,
                 channel_mapping=None):
    """ make obspy traces from trace blocks and headers """
    traces = []  # list to store traces
    trace_position = data_block_start
    while True:  # read traces until parser falls of the end of file
        try:
            theader = read_block(fi, trace_header_block, trace_position)
        except ValueError:  # this is the end, my only friend, the end
            break
        # get stats
        stats = _make_stats(theader, gheader, channel_mapping)
        # expected jump to next start position
        jumps = stats.npts * 4 + theader['num_ext_blocks'] * 32 + 20
        # if wrong starttime / endtime just keep going and update position
        if stats.endtime < starttime or stats.starttime > endtime:
            trace_position += jumps
            continue
        if head_only:  # empty np array for head only
            data = np.array([])
        else:  # else read data
            data_start = trace_position + 20 + theader['num_ext_blocks'] * 32
            data = read(fi, data_start, theader['samples'] * 4, '>f4')
        traces.append(Trace(data=data, header=stats))
        trace_position += jumps
    if merge:
        traces = quick_merge(traces)
    return traces


def _make_stats(theader, gheader, channel_mapping):
    """ make Stats object """
    sampling_rate = int(1000. / (gheader['base_scan'] / 16.))
    channel_code = _get_channel_code(theader['channel_code'], sampling_rate,
                                     channel_mapping)

    statsdict = dict(
        starttime=UTCDateTime(theader['time'] / 1000000.),
        sampling_rate=sampling_rate,
        npts=theader['samples'],
        network=str(theader['line_number']),
        station=str(theader['point']),
        location=str(theader['index']),
        channel=channel_code,
    )
    return Stats(statsdict)


def _get_channel_code(code, sampling_rate, channel_mapping):
    """ return a seed compliant (hopefully) channel code """
    component = (channel_mapping or COMPONENT_MAP)[code]
    band_code = BAND_MAP[sampling_rate]
    return band_code + INSTRUMENT_CODE + component


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
