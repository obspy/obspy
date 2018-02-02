# -*- coding: utf-8 -*-
"""
Evt (Kinemetrics) format support for ObsPy.

:copyright:
    Royal Observatory of Belgium, 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from struct import unpack

import numpy as np

from obspy import Stream, Trace
from obspy.core.compatibility import from_buffer
from .evt_base import (EvtBadDataError, EvtBadHeaderError, EvtEOFError,
                       EvtVirtual)

WARNING_HEADER = "Only tested with files from ROB networks :" + \
    " - New Etna and old Etna" + \
    " - ByteOrder : MSB first (Motorola) " + \
    " - File Header of 2040 bytes (12 Channels)" + \
    " - .........." + \
    "Other situation may not work (not yet implemented)."

FRAME_STRUCT = b"BBHHLHHBBHB13s"

# 0 - 0x7c (0-34)
HEADER_STRUCT1 = b"3sB2H3B3x6Hh2Hh22x3B5x2H4hH2x2h6L16x"
# 0x7c - 0x22c (35-106)
HEADER_STRUCT2 = b"lLlL2l12x" * 12
# 0x22c-0x2c8(->139)
HEADER_STRUCT3 = b"3L4H2L4x2H5s33sh2f4h4B2LB17s2B2B6xh22x"
# 0x2c8 - 0x658 (140 - 464) (27*12)
HEADER_STRUCT4 = b"5sbH5h3H4BHBx8f2B10x" * 12
# 0x658 - 0x688
HEADER_STRUCT5 = b"3B5x6H2hb2Bxh3Hl8x"
# 0x688 - 0x6c4 (3*15)
HEADER_STRUCT6 = b"cBh" * 15
# 0x6c4 - 0x7f8
HEADER_STRUCT7 = b"64s16s16s16s16s16s24s24s24s24s3B3b5h4xH46s"


class Evt(object):
    """
    Class to read Evt (Kinemetrics) formatted files.
    """
    def __init__(self):
        self.e_tag = EvtTag()
        self.e_header = EvtHeader()
        self.e_frame = EvtFrameHeader()
        self.e_data = EvtData()
        self.samplingrate = 0

    def calibration(self):
        """
        Apply calibrations on data matrix

        Note about calibration:
            fullscale of instrument = +/- 2.5V & +/-20V
            data : 4 bytes - 24 bits
            calibration in volts = data * fullscale / 2**23

            sensitivity = volts/g
            calibration in MKS units = (data_in_volts / sensitivity) * g

        """
        for i in range(self.e_header.nchannels):
            calib_volts = 8388608.0 / self.e_header.chan_fullscale[i]
            # 8388608 = 2**23
            calib_mks = (calib_volts *
                         self.e_header.chan_sensitivity[i]) / (9.81)
            # 9.81 = mean value of g on earth
            self.data[i] /= calib_mks

    def read_file(self, filename_or_object, raw=False):
        """
        Reads an Evt file to the internal data structure

        :type filename_or_object: str or file-like object
        :param filename_or_object: Evt file to be read
        :type raw : bool
        :param raw : True if raw data (no corrections, int32)
                     False if data in m/s2 (default)
        :rtype: obspy.core.stream.Stream
        :return: Obspy Stream with data
        """
        # Support reading from filenames of file-like objects.
        if hasattr(filename_or_object, "seek") and \
                hasattr(filename_or_object, "tell") and \
                hasattr(filename_or_object, "read"):
            is_fileobject = True
            file_pointer = filename_or_object
        else:
            is_fileobject = False
            file_pointer = open(filename_or_object, "rb")

        try:
            self.e_tag.read(file_pointer)
            endian = self.e_tag.endian
            self.e_header.unset_dict()
            self.e_header.read(file_pointer, self.e_tag.length, endian)

            self.data = np.ndarray([self.e_header.nchannels, 0])

            while True:
                try:
                    self.e_tag.read(file_pointer)
                    retparam = self.e_frame.read(file_pointer,
                                                 self.e_tag.length, endian)
                    if self.samplingrate == 0:
                        self.samplingrate = retparam[0]
                    elif self.samplingrate != retparam[0]:
                        raise EvtBadHeaderError("Sampling rate not constant")
                    data = self.e_data.read(file_pointer,
                                            self.e_tag.datalength,
                                            endian, retparam)
                    self.data = np.hstack((self.data, data))
                except EvtEOFError:
                    break
        finally:
            if not is_fileobject:
                file_pointer.close()

        if self.e_frame.count() != self.e_header.duration:
            raise EvtBadDataError("Bad number of blocks")

        if not raw:
            self.calibration()

        traces = []
        for i in range(self.e_header.nchannels):
            cur_trace = Trace(data=self.data[i])
            cur_trace.stats.channel = str(i)
            cur_trace.stats.station = self.e_header.stnid
            cur_trace.stats.sampling_rate = float(self.samplingrate)
            cur_trace.stats.starttime = self.e_header.starttime
            cur_trace.stats.kinemetrics_evt = self.e_header.make_obspy_dict(i)
            traces.append(cur_trace)

        return Stream(traces=traces)


class EvtData(object):
    """
    Class to manage data in Evt file
    """
    def read(self, file_p, length, endian, param):
        """
        read data from file_p

        :param file_p: file pointer
        :param length: length to be read
        :param endian: endian type in datafile
        :type param: list
        :param param: sampling rate,sample size, block time, channels
        :rtype: list of list
        :return: list of data
        """
        buff = file_p.read(length)
        samplerate = param[0]
        numbyte = param[1]
        numchan = param[3]
        num = (samplerate // 10) * numbyte * numchan
        if length != num:
            raise EvtBadDataError("Bad data length")

        if numbyte == 2:
            data = from_buffer(buff, ">h").reshape((-1, numchan)).T
        elif numbyte == 4:
            data = from_buffer(buff, ">i").reshape((-1, numchan)).T
        elif numbyte == 3:
            data = np.empty((numchan, samplerate // 10))
            for j in range(samplerate // 10):
                for k in range(numchan):
                    i = (j * numchan) + k
                    val = unpack(b">i", buff[i * 3:(i * 3) + 3] + b'\0')[0] \
                        >> 8
                    data[k, j] = val

        return data


class EvtHeader(EvtVirtual):
    """
    Class to manage header of Evt file
    """
    HEADER = {'instrument': [1, ['_instrument', '']],
              'a2dbits': [4, ''],
              'samplebytes': [5, ''],
              'installedchan': [7, ''],
              'maxchannels': [8, ''],
              'batteryvoltage': [13, ''],
              'temperature': [16, ''],
              'gpsstatus': [18, ['_gpsstatus', '']],
              'gpslastlock': [33, ['_time', -1]],
              'starttime': [107, ['_time', 112]],
              'triggertime': [108, ['_time', 113]],
              'duration': [109, ''],
              'nscans': [115, ''],
              'serialnumber': [116, ''],
              'nchannels': [117, ''],
              'stnid': [118, ['_strnull', '']],
              'comment': [119, ['_strnull', '']],
              'elevation': [120, ''],
              'latitude': [121, ''],
              'longitude': [122, ''],
              'chan_id': [140, ['_arraynull', [12, 27, 140]]],
              'chan_north': [143, ['_array', [12, 27, 143]]],
              'chan_east': [144, ['_array', [12, 27, 144]]],
              'chan_up': [145, ['_array', [12, 27, 145]]],
              'chan_azimuth': [147, ['_array', [12, 27, 147]]],
              'chan_gain': [150, ['_array', [12, 27, 150]]],
              'chan_fullscale': [157, ['_array', [12, 27, 157]]],
              'chan_sensitivity': [158, ['_array', [12, 27, 158]]],
              'chan_damping': [159, ['_array', [12, 27, 159]]],
              'chan_natfreq': [160, ['_array', [12, 27, 160]]],
              'chan_calcoil': [164, ['_array', [12, 27, 164]]],
              'chan_range': [165, ['_array', [12, 27, 165]]],
              'chan_sensorgain': [166, ['_array', [12, 27, 166]]]}

    def __init__(self):
        EvtVirtual.__init__(self)

    def read(self, file_p, length, endian):
        """
        read the Header of Evt file
        """
        buff = file_p.read(length)
        self.endian = endian
        if length == 2040:  # File Header 12 channel
            self.analyse_header12(buff)
        elif length == 2736:  # File Header 18 channel
            raise NotImplementedError("16 Channel not implemented")
        else:
            raise EvtBadHeaderError("Bad Header length " + length)

    def analyse_header12(self, head_buff):
        val = unpack(self.endian + HEADER_STRUCT1, head_buff[0:0x7c])
        self.set_dict(val, 0)
        val = unpack(self.endian + HEADER_STRUCT2, head_buff[0x7c:0x22c])
        self.set_dict(val, 35)
        val = unpack(self.endian + HEADER_STRUCT3, head_buff[0x22c:0x2c8])
        self.set_dict(val, 107)
        val = unpack(self.endian + HEADER_STRUCT4, head_buff[0x2c8:0x658])
        self.set_dict(val, 140)
        # Those three do not do anything... (For futur extensions)
        # val = unpack(self.endian+HEADER_STRUCT5, head_buff[0x658:0x688])
        # val = unpack(self.endian+HEADER_STRUCT6, head_buff[0x688:0x6c4])
        # val = unpack(self.endian+HEADER_STRUCT7, head_buff[0x6c4:0x7f8])

    def make_obspy_dict(self, numchan):
        """
        Make an ObsPy dictionary from header dictionary for 1 channel

        :param numchan: channel to be converted
        :rtype: dictionary
        """
        dico = {}
        for key in self.HEADER:
            value = self.HEADER[key][2]
            if isinstance(value, list):
                dico[key] = value[numchan]
            else:
                dico[key] = value
        return dico

    def _gpsstatus(self, value, unused_a, unused_b, unused_c):
        """
        Transform bitarray for gpsstatus in human readable string

        :param value: gps status
        :rtype: string
        """
        dico = {1: 'Checking', 2: 'Present', 4: 'Error', 8: 'Failed',
                16: 'Not Locked', 32: 'ON'}
        retval = ""
        for key in sorted(dico):
            if value & key:
                retval += dico[key] + " "
        return retval


class EvtFrameHeader(EvtVirtual):
    """
    Class to manage frame header in Evt file
    """
    HEADER = {'frametype': [0, ''],
              'instrumentcode': [1, ['_instrument', '']],
              'recorderid': [2, ''],
              'framesize': [3, ''],
              'blocktime': [4, ['_time', 9]],
              'channelbitmap': [5, ''],
              'streampar': [6, ''],
              'framestatus': [7, ''],
              'framestatus2': [8, ''],
              'msec': [9, ''],
              'channelbitmap1': [10, ''],
              'timecode': [11, '']}

    def __init__(self):
        EvtVirtual.__init__(self)
        self.numframe = 0
        self.endian = 0

    def count(self):
        """
        return the number of frames read
        """
        return self.numframe

    def read(self, file_p, length, endian):
        """
        read a frame

        :rtype: list
        :return: samplingrate, samplesize, blocktime, channels
        """
        buff = file_p.read(length)
        self.endian = endian
        if length == 32:  # Frame Header
            self.analyse_frame32(buff)
        else:
            raise EvtBadHeaderError("Bad Header length " + length)
        samplingrate = self.streampar & 4095
        samplesize = (self.framestatus >> 6) + 1
        if samplesize not in [2, 3, 4]:
            samplesize = 3  # default value = 3
        return (samplingrate, samplesize, self.blocktime, self.channels())

    def analyse_frame32(self, head_buff):
        self.numframe += 1
        val = unpack(self.endian + FRAME_STRUCT, head_buff)
        self.set_dict(val, 0)
        if not self.verify(verbose=False):
            raise EvtBadHeaderError("Bad Frame values")

    def verify(self, verbose=False):
        if self.frametype not in [3, 4, 5]:
            if verbose:
                print("FrameType ", self.frametype, " not Known")
            return False
        return True

    def channels(self):
        numchan = 12
        if self.frametype == 4:
            raise NotImplementedError("16 Channels not implemented")
        chan = 0
        for i in range(numchan):
            pow_of_2 = 2 ** i
            if self.channelbitmap & pow_of_2:
                chan += 1
        return chan


class EvtTag(EvtVirtual):
    """
    Class to read the TAGs of Evt Files
    """
    HEADER = {'order': [1, ''],
              'version': [2, ''],
              'instrument': [3, ['_instrument', '']],
              'type': [4, ''],
              'length': [5, ''],
              'datalength': [6, ''],
              'id': [7, ''],
              'checksum': [8, ''],
              'endian': [9, '']}

    def __init__(self):
        EvtVirtual.__init__(self)
        self.endian = 0

    def read(self, file_p):
        """
        :type file_p: str
        :param file_p: file descriptor of Evt file.
        """
        mystr = file_p.read(16)
        if len(mystr) < 16:
            raise EvtEOFError
        sync, byte_order = unpack(b"cB", mystr[0:2])
        if sync == b'\x00':
            raise EvtEOFError
        if sync != b'K':
            raise EvtBadHeaderError('Sync error')
        if byte_order == 1:
            endian = b">"
        elif byte_order == 0:
            endian = b"<"
        else:
            raise EvtBadHeaderError
        val = unpack(endian + b"cBBBLHHHH", mystr)
        self.set_dict(val)
        self.endian = endian
        if not self.verify(verbose=False):
            raise EvtBadHeaderError("Bad Tag values")

    def verify(self, verbose=False):
        if self.type not in [1, 2]:
            if verbose:
                print("Type of Header ", self.type, " not known")
            return False
        if self.type == 1 and self.length not in [2040, 2736]:
            if verbose:
                print("Bad Header file length : ", self.length)
            return False
        return True
