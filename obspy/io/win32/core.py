# -*- coding: utf-8 -*-
"""
WIN32 bindings to ObsPy core module.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import struct
import numpy as np
from obspy import UTCDateTime, Stream, Trace
from obspy.core import AttribDict


def _is_win32(filename):
    """
    Checks whether a file is WIN32 format or not.

    :type filename: str
    :param filename: WIN32 file to be checked.
    :rtype: bool
    :return: ``True`` if a WIN32 file.
    """
    try:
        with open(filename, "rb") as fpin:
            # get full file size
            fpin.seek(0, 2)
            size = fpin.tell()
            fpin.seek(0)

            # check file header
            if fpin.read(4) != b'\x00\x00\x00\x00':
                return False

            length = 4
            # check headers of all one-second blocks
            while length < size:
                block_header = struct.unpack('>8sii', fpin.read(16))
                UTCDateTime(block_header[0][0:7].hex())

                # time length fixed at 10 (0x000A)
                if block_header[1] != 10:
                    return False

                # skip this block and check next block header
                block_length = block_header[2]
                fpin.seek(block_length, 1)
                length += 16 + block_length
    except Exception:
        return False
    return True


def _read_win32(filename, channel_table_filename=None, **kwargs):
    """
    Reads a WIN32 file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: WIN32 file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """

    data = {}
    network = {}
    starttime = {}
    sampling_rate = {}
    with open(filename, "rb") as fpin:
        # get full file size
        fpin.seek(0, 2)
        size = fpin.tell()
        fpin.seek(0)

        # read file header
        if fpin.read(4) != b'\x00\x00\x00\x00':
            raise ValueError("File not in WIN32 format.")

        length = 4
        # loop over one-second blocks
        while length < size:
            block_header = struct.unpack('>8sii', fpin.read(16))
            # Binary-coded decimal: YYYYMMDDhhmmssxx
            # sub-second is fixed at 0x00
            time = UTCDateTime(block_header[0][0:7].hex())
            if block_header[1] != 10:  # time length is fixed at 10 (0x000A)
                raise ValueError("File not in WIN32 format.")

            block_size = block_header[2]
            block_length = 0
            # loop over channels
            while block_length < block_size:
                channel_header = struct.unpack('>cc2s2s', fpin.read(6))
                org = channel_header[0].hex()
                net = channel_header[1].hex()
                chid = channel_header[2].hex()

                # sample_bytes is determined by first 4 bits
                # 0: 4 bit; 1: 8 bit; 2: 16 bit; 3: 24 bit; 4: 32 bit;
                sample_bytes = channel_header[3][0] >> 4
                if sample_bytes == 0:
                    sample_bytes == 0.5

                srate = int(channel_header[3].hex()[1:], 16)

                # read data in
                first_sample = struct.unpack('>i', fpin.read(4))[0]
                # sample_bytes = 0.5 and 4 is NOT full tested
                if sample_bytes == 0.5:
                    xlen = int(srate*sample_bytes)
                    buff = fpin.read(xlen)
                    samples = []
                    for i in range(xlen):
                        temp = np.frombuffer(buff[i:i+1], np.int8)[0] >> 4
                        samples.append(temp)
                        if i != xlen - 1:  # skip the last one
                            temp = (np.frombuffer(buff[i:i+1],
                                                  np.int8)[0] << 4) >> 4
                            samples.append(temp)
                    samples = np.array(samples).astype('int64')
                elif sample_bytes == 3:
                    buff = fpin.read(sample_bytes * (srate - 1))
                    samples = []
                    for i in range(srate-1):
                        temp = np.frombuffer(buff[3*i:3*(i+1)] + b' ',
                                             '>i')[0] >> 8
                        samples.append(temp)
                    samples = np.array(samples).astype('int64')
                elif sample_bytes in (1, 2, 4):
                    buff = fpin.read(sample_bytes * (srate - 1))
                    dtype = '>i{}'.format(sample_bytes)
                    samples = np.frombuffer(buff, dtype=dtype).astype('int64')

                # delta decompression
                array = np.insert(samples, 0, first_sample)
                array = np.cumsum(array)

                if chid not in data:
                    data[chid] = array.copy()
                else:
                    data[chid] = np.append(data[chid], array)

                if chid not in network:
                    network[chid] = '{}{}'.format(org, net)
                if chid not in starttime:
                    starttime[chid] = time
                if chid not in sampling_rate:
                    sampling_rate[chid] = srate

                block_length += 6 + 4 + sample_bytes * (srate-1)

            length += 16 + block_size

        if channel_table_filename:
            params = _read_channel_table(channel_table_filename)

        st = Stream()
        for i in data.keys():
            tr = Trace(data=data[i])
            tr.stats.starttime = starttime[i]
            tr.stats.sampling_rate = sampling_rate[i]
            tr.stats.network = network[i]
            tr.stats.channel = i
            if channel_table_filename:
                tr.stats.station = params[i][3]
                tr.stats.channel = params[i][4]

                tr.stats.win32 = AttribDict()
                tr.stats.win32.channel_id = i
                tr.stats.win32.record_flag = int(params[i][1])
                tr.stats.win32.delay_time = float(params[i][2])
                tr.stats.win32.monitor_waveform_amplitude = int(params[i][5])
                tr.stats.win32.adc_bit_size = int(params[i][6])
                tr.stats.win32.sensitivity = float(params[i][7])
                tr.stats.win32.unit = params[i][8]
                tr.stats.win32.nature_period = float(params[i][9])
                tr.stats.win32.dumping_constant = float(params[i][10])
                tr.stats.win32.preamplificacation = params[i][11]
                tr.stats.win32.lsb_value = float(params[i][12])
                tr.stats.win32.latitude = float(params[i][13])
                tr.stats.win32.longitude = float(params[i][14])
                tr.stats.win32.altitude = float(params[i][15])
                tr.stats.win32.staiton_correction_p = float(params[i][16])
                tr.stats.win32.station_correction_s = float(params[i][17])
                tr.stats.win32.station_name = params[i][18]

            st.append(tr)

        return st


def _read_channel_table(filename):
    """
    Read a channel table file and returns a parameter dict

    :type filename: str
    :param filename: channel table file to be read.
    :rtype: dict
    :return: channel related paramaters.
    """

    params = {}
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            items = line.split()
            params[items[0]] = items

    return params
