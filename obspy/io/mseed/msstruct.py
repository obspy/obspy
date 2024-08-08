# -*- coding: utf-8 -*-
"""
Convenience class for handling MSRecord and MSFileparam.
"""
import ctypes as C  # NOQA
import os

import numpy as np

from obspy import UTCDateTime
from obspy.core.compatibility import from_buffer
from .headers import HPTMODULUS, MS_NOERROR, MSFileParam, MSRecord, clibmseed


def _get_ms_file_info(f, real_name):
    """
    Takes a Mini-SEED filename as an argument and returns a dictionary
    with some basic information about the file. Also suitable for Full
    SEED.

    This is an exact copy of a method of the same name in utils. Due to
    circular imports this method cannot be import from utils.
    XXX: Figure out a better way!

    :param f: File pointer of opened file in binary format
    :param real_name: Realname of the file, needed for calculating size
    """
    # get size of file
    info = {'filesize': os.path.getsize(real_name)}
    pos = f.tell()
    f.seek(0)
    rec_buffer = from_buffer(f.read(512), dtype=np.int8)
    info['record_length'] = clibmseed.ms_detect(rec_buffer, 512)
    # Calculate Number of Records
    info['number_of_records'] = int(info['filesize'] //
                                    info['record_length'])
    info['excess_bytes'] = info['filesize'] % info['record_length']
    f.seek(pos)
    return info


class _MSStruct(object):
    """
    Class for handling MSRecord and MSFileparam.

    It consists of a MSRecord and MSFileparam and an attached python file
    pointer.

    :ivar msr: MSRecord
    :ivar msf: MSFileparam
    :ivar file: filename
    :ivar offset: Current offset

    :param filename: file to attach to
    :param init_msrmsf: initialize msr and msf structure
        by a first pass of read. Setting this option to
        false will result in errors when setting e.g.
        the offset before a call to read
    """
    def __init__(self, filename, init_msrmsf=True):
        # Initialize MSRecord structure
        self.msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        self.msf = C.POINTER(MSFileParam)()  # null pointer
        self.file = filename
        # dummy read once, to avoid null pointer in ms.msf for e.g.
        # ms.offset
        if init_msrmsf:
            self.read(-1, 0, 1, 0)
            self.offset = 0

    def get_end(self):
        """
        Return endtime
        """
        self.read(-1, 0, 1, 0)
        dtime = clibmseed.msr_endtime(self.msr)
        return UTCDateTime(dtime / HPTMODULUS)

    def get_start(self):
        """
        Return starttime
        """
        self.read(-1, 0, 1, 0)
        dtime = clibmseed.msr_starttime(self.msr)
        return UTCDateTime(dtime / HPTMODULUS)

    def file_info(self):
        """
        For details see util._get_ms_file_info
        """
        fp = open(self.file, 'rb')
        self.info = _get_ms_file_info(fp, self.file)
        fp.close()
        return self.info

    def file_pos_from_rec_num(self, record_number=0):
        """
        Return byte position of file given a certain record_number.

        The byte position can be used to seek to certain points in the file
        """
        if not hasattr(self, 'info'):
            self.info = self.file_info()
        # Calculate offset of the record to be read.
        if record_number < 0:
            record_number = self.info['number_of_records'] + record_number
        if record_number < 0 or \
           record_number >= self.info['number_of_records']:
            raise ValueError('Please enter a valid record_number')
        return record_number * self.info['record_length']

    def read(self, reclen=-1, dataflag=1, skipnotdata=1, verbose=0,
             raise_flag=True):
        """
        Read MSRecord using the ms_readmsr_r function. The following
        parameters are directly passed to ms_readmsr_r.

        :param ms: _MSStruct (actually consists of a LP_MSRecord,
            LP_MSFileParam and an attached file pointer).
            Given an existing ms the function is much faster.
        :param reclen: If reclen is 0 the length of the first record is auto-
            detected. All subsequent records are then expected to have the
            same record length. If reclen is negative the length of every
            record is automatically detected. Defaults to -1.
        :param dataflag: Controls whether data samples are unpacked, defaults
            to 1.
        :param skipnotdata: If true (not zero) any data chunks read that to do
            not have valid data record indicators will be skipped. Defaults to
            True (1).
        :param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        :param record_number: Number of the record to be read. The first record
            has the number 0. Negative numbers will start counting from the end
            of the file, e.g. -1 is the last complete record.
        """
        errcode = clibmseed.ms_readmsr_r(C.pointer(self.msf),
                                         C.pointer(self.msr),
                                         self.file.encode('ascii', 'strict'),
                                         reclen, None, None,
                                         skipnotdata, dataflag, verbose)
        if raise_flag:
            if errcode != MS_NOERROR:
                raise Exception("Error %d in ms_readmsr_r" % errcode)
        return errcode

    def __del__(self):
        """
        Method for deallocating MSFileParam and MSRecord structure.
        """
        errcode = clibmseed.ms_readmsr_r(C.pointer(self.msf),
                                         C.pointer(self.msr),
                                         None, -1, None, None, 0, 0, 0)
        if errcode != MS_NOERROR:
            raise Exception("Error %d in ms_readmsr_r" % errcode)

    def set_offset(self, value):
        self.msf.contents.readoffset = C.c_int(value)

    def get_offset(self):
        return int(self.msf.contents.readoffset)

    offset = property(get_offset, set_offset)
