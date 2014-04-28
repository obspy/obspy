# -*- coding: utf-8 -*-
"""
EVT (Kinemetrics) support for ObsPy.

:copyright:
Royal Observatory of Belgium, 2013
:license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)
"""

from struct import unpack
import numpy as np

from obspy import Trace, Stream

from . evt_base import *


WARNING_HEADER = "Only tested with files from ROB networks :" + \
    " - New ETna and old Etna" + \
    " - ByteOrder : MSB first (Motorola) " + \
    " - File Header of 2040 bytes (12 Channels)" + \
    " - .........." + \
    "Other situation may not work (not implemented)."


class EVT(object):
    """
    Class to read EVT (Kinemetrics) formatted files.
    """

    def __init__(self):
        self.ETag = EVT_TAG()
        self.EHeader = EVT_HEADER()
        self.EFrame = EVT_FRAME()
        self.EData = EVT_DATA()
        self.samplingrate = 0

    def saveascii(self):
        """
        for test purposes
        """
        fp = open("output.txt", "w")
        fp.write("####################\n")
        print >> fp, self.EHeader
        fp.write("####################\n\n")
        for i in range(self.data.shape[1]):
            tmp = ""
            for j in range(self.data.shape[0]):
                tmp += str(self.data[j][i]) + "    "
            tmp += "\n"
            fp.write(tmp)
        fp.close()

    def calibration(self):
        """
        Apply calibrations on data matrix
        """
        for i in range(self.EHeader.nchannels):
#            grange = 8.0/(2 ** self.EHeader.chan_range[i])
            calibV = 8388608.0 / self.EHeader.chan_fullscale[i]
            calibMKS = (calibV*self.EHeader.chan_sensitivity[i])/(9.81)
            self.data[i] /= calibMKS

    def readFile(self, filename, Raw=False):
        """
        :type filename: string
        :param filename: EVT file to be read.
        :type raw : boolean
        :param raw : True if Raw datas (no corrections, int32)
                     False if data in m/s2 (default)
        :rtype: obspy.core.stream.Stream
        :return: Obspy STream with data
        """
        self.file_pointer = open(filename, "rb")
        self.ETag.read(self.file_pointer)
        endian = self.ETag.endian
        self.EHeader.unsetdico()
        self.EHeader.read(self.file_pointer, self.ETag.lenght, endian)

        self.data = np.ndarray([self.EHeader.nchannels, 0])

        while True:
            try:
                self.ETag.read(self.file_pointer)
                retparam = self.EFrame.read(self.file_pointer,
                                            self.ETag.lenght, endian)
                if (self.samplingrate == 0):
                    self.samplingrate = retparam[0]
                elif (self.samplingrate != retparam[0]):
                    raise EVTBadHeaderError("Sampling rate not constant")
                datal = self.EData.read(self.file_pointer,
                                        self.ETag.datalenght, endian, retparam)
                npdata = np.array(datal)
                self.data = np.hstack((self.data, npdata))  # append data
            except EVTEOFError:
                break
        if (self.EFrame.count() != self.EHeader.duration):
            raise EVTBadDataError("Bad number of blocks")

        if not Raw:
            self.calibration()
        self.file_pointer.close()

        traces = []
        for i in range(self.EHeader.nchannels):
            t = Trace(data=self.data[i])
            t.stats.channel = str(i)
            t.stats.station = self.EHeader.stnid
            t.stats.sampling_rate = float(self.samplingrate)
            t.stats.starttime = self.EHeader.startime
            t.stats.evt = self.EHeader.makeobspydico(i)
            traces.append(t)

        return Stream(traces=traces)


###############################################################################

class EVT_DATA(object):
    def read(self, fp, length, endian, param):
        """
        read data from fp
        :param fp: file pointer
        :param length: length to be readed
        :param endian: endian type in datafile
        :type param: list
        :param param: sampling rate,sample size, block time, channels
        :rtype: list of list
        :return: list of data
        """

        buff = fp.read(length)
        samplrate = param[0]
        numbyte = param[1]
        numchan = param[3].count(1)
        num = (samplrate/10)*numbyte*numchan
        data = [[] for _ in range(numchan)]
        if (length != num):
            raise EVTBadDataError("Bad data lenght")
        for j in range(samplrate/10):
            for k in range(numchan):
                i = (j*numchan)+k
                if numbyte == 2:
                    val = unpack(">i", buff[i*2:(i*2)+2] + '\0\0')[0] >> 8
                elif numbyte == 3:
                    val = unpack(">i", buff[i*3:(i*3)+3] + '\0')[0] >> 8
                elif numbyte == 4:
                    val = unpack(">i", buff[i*4:(i*4)+4])[0]
                else:
                    raise EVTBadDataError("Bad data format")
                data[k].append(val)
        return data


#===========================================================================
HEADER_STRUCT1 = "3sB2H3B3x6Hh2Hh22x3B5x2H4hH2x2h6L16x"  # 0 - 0x7c (0-34)
HEADER_STRUCT2 = "lLlL2l12x"*12  # 0x7c - 0x22c (35-106)
HEADER_STRUCT3 = "3L4H2L4x2H5s33sh2f4h4B2LB17s2B2B6xh22x"  # 0x22c-0x2c8(->139)
HEADER_STRUCT4 = "5sbH5h3H4BHBx8f2B10x"*12  # 0x2c8 - 0x658 (140 - 464) (27*12)
HEADER_STRUCT5 = "3B5x6H2hb2Bxh3Hl8x"  # 0x658 - 0x688
HEADER_STRUCT6 = "cBh"*15  # 0x688 - 0x6c4 (3*15)
HEADER_STRUCT7 = "64s16s16s16s16s16s24s24s24s24s3B3b5h4xH46s"  # 0x6c4 - 0x7f8


class EVT_HEADER(EVT_Virtual):
    HEADER_DICT = {'instrument': [1, ['_instrument', '']],
                   'a2dbits': [4, ""],
                   'samplebytes': [5, ""],
                   'installedchan': [7, ""],
                   'maxchannels': [8, ""],
                   'batteryvoltage': [13, ""],
                   'temperature': [16, ""],
                   'gpsstatus': [18, ['_gpsstatus', '']],
                   'gpslastlock': [33, ['_time', -1]],
                   'startime': [107, ['_time', 112]],
                   'triggertime': [108, ['_time', 113]],
                   'duration': [109, ""],
                   'nscans': [115, ""],
                   'serialnumber': [116, ""],
                   'nchannels': [117, ""],
                   'stnid': [118, ['_strnull', '']],
                   'comment': [119, ['_strnull', '']],
                   'elevation': [120, ""],
                   'latitude': [121, ""],
                   'longitude': [122, ""],
                   'chan_id': [140, ['_arraynull', [12, 27, 140]]],
                   'chan_north': [143, ['_arraynull', [12, 27, 143]]],
                   'chan_east': [144, ['_arraynull', [12, 27, 144]]],
                   'chan_up': [145, ['_arraynull', [12, 27, 145]]],
                   'chan_azimuth': [147, ['_arraynull', [12, 27, 147]]],
                   'chan_gain': [150, ['_array', [12, 27, 150]]],
                   'chan_fullscale': [157, ['_array', [12, 27, 157]]],
                   'chan_sensitivity': [158, ['_array', [12, 27, 158]]],
                   'chan_damping': [159, ['_array', [12, 27, 159]]],
                   'chan_natfreq': [160, ['_array', [12, 27, 160]]],
                   'chan_calcoil': [164, ['_array', [12, 27, 164]]],
                   'chan_range': [165, ['_array', [12, 27, 165]]],
                   'chan_sensogain': [166, ['_array', [12, 27, 166]]]}

    def __init__(self):
        EVT_Virtual.__init__(self)
        self.diconame = "self.HEADER_DICT"

    def read(self, fp, length, endian):
        """
        read Header of Evt file
        """
        buff = fp.read(length)
        self.endian = endian
        if (length == 2040):  # File Header 12 channel
            self.analyse_header12(buff)
        elif (length == 2736):  # File Header 18 channel
            raise EVTNotImplementedError("16 Channel not implemented")
        else:
            raise EVTBadHeaderError("Bad Header lenght " + length)

    def analyse_header12(self, head_buff):
        val = unpack(self.endian+HEADER_STRUCT1, head_buff[0:0x7c])
        self.setdico(list(val), 0)
        val = unpack(self.endian+HEADER_STRUCT2, head_buff[0x7c:0x22c])
        self.setdico(list(val), 35)
        val = unpack(self.endian+HEADER_STRUCT3, head_buff[0x22c:0x2c8])
        self.setdico(list(val), 107)
        val = unpack(self.endian+HEADER_STRUCT4, head_buff[0x2c8:0x658])
        self.setdico(list(val), 140)
        val = unpack(self.endian+HEADER_STRUCT5, head_buff[0x658:0x688])
        val = unpack(self.endian+HEADER_STRUCT6, head_buff[0x688:0x6c4])
        val = unpack(self.endian+HEADER_STRUCT7, head_buff[0x6c4:0x7f8])

    def makeobspydico(self, numchan):
        """
        Make an obpsy dictionnary from header dictionnary for 1 channel
        :param numchan: channel to be converted
        :rtype: dictionnary
        """
        dico = {}
        for key in self.HEADER_DICT:
            try:
                value = self.HEADER_DICT[key][2]
                if type(value) is list:
                    dico[key] = value[numchan]
                else:
                    dico[key] = value
            except:
                pass
        return dico

    def _gpsstatus(self, value, a, b, c):
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

#===========================================================================
FRAME_STRUCT = "BBHHLHHBBHB13s"


class EVT_FRAME(EVT_Virtual):
    FRAME_DICT = {'frametype': [0, ''],
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

    endian = 0

    def __init__(self):
        EVT_Virtual.__init__(self)
        self.diconame = "self.FRAME_DICT"
        self.numframe = 0

    def count(self):
        """
        return the number of frames readed
        """
        return self.numframe

    def read(self, fp, length, endian):
        """
        read a frame
        :rtype: list
        :return: samplingrate, samplesize, blocktime, channels
        """
        buff = fp.read(length)
        self.endian = endian
        if (length == 32):  # Frame Header
            self.analyse_frame32(buff)
        else:
            raise EVTBadHeaderError("Bad Header lenght " + length)
        samplingrate = self.streampar & 4095
        samplesize = (self.framestatus >> 6)+1
        return (samplingrate, samplesize, self.blocktime, self.channels())

    def analyse_frame32(self, head_buff):
        self.numframe += 1
        val = unpack(self.endian+FRAME_STRUCT, head_buff)
        self.setdico(list(val), 0)
        if self.verify(0) == 1:
            raise EVTBadHeaderError("Bad Frame values")

    def verify(self, verbose=1):
        retval = 0
        if self.frametype not in [3, 4, 5]:
            if verbose:
                print "FrameType ", self.frametype, " not Known"
            retval = 1
        return retval

    def channels(self):
        numchan = 12
        if (self.frametype == 4):
            raise EVTNotImplementedError("16 Channels not implemented")
        chan = []
        for i in range(numchan):
            p2 = 2**i
            if (self.channelbitmap & p2):
                chan.append(1)
            else:
                chan.append(0)
        return chan


#=======================================================================
class EVT_TAG(EVT_Virtual):
    """
    Class to read the TAGs of EVT Files
    """
    TAG_DICT = {'order': [1, ""],
                'version': [2, ""],
                'instrument': [3, ['_instrument', '']],
                'type': [4, ""],
                'lenght': [5, ""],
                'datalenght': [6, ""],
                'id': [7, ""],
                'checksum': [8, ""],
                'endian': [9, ""]}
    INST_COUNT = 0

    def __init__(self):
        EVT_Virtual.__init__(self)
        self.diconame = "self.TAG_DICT"

    def read(self, fp):
        """
        :type fp: string
        :param fp: file descriptor of EVT file.
        """
        mystr = fp.read(16)
        sync, byteOrder = unpack("cB", mystr[0:2])
        if(len(mystr) < 16) or (sync == '\x00'):
            raise EVTEOFError
        if (sync != 'K'):
            raise EVTBadHeaderError('Sync error : <'+sync+'>')
        if (byteOrder == 1):
            endian = ">"
        elif (byteOrder == 0):
            endian = "<"
        else:
            raise EVTBadHeaderError
        val = list(unpack(endian + "cBBBLHHHH", mystr))
        self.setdico(val)
        self.endian = endian
        if self.verify(0) == 1:
            raise EVTBadHeaderError("Bad Tag values")
        return 1

    def verify(self, verbose=1):
        retval = 0
        if self.type not in [1, 2]:
            if verbose:
                print "Type of Header ", self.type, " not known"
            retval = 1
        if (self.type == 1) and (self.lenght not in [2040, 2736]):
            if verbose:
                print "Bad Header file lenght : ", self.lenght
            retval = 1
        return retval
