# -*- coding: utf-8 -*-
"""
Convenience class for iterating over miniSEED records in a file.

Version 2017.114
"""
import ctypes as C  # NOQA

from obspy import UTCDateTime
from obspy.io.mseed.headers import (HPTMODULUS, MS_NOERROR, MS_ENDOFFILE,
                                    MSFileParam, MSRecord, clibmseed)


class _MSRIterator(object):
    """
    Class for iterating through miniSEED records in a file.

    :ivar msr: MSRecord (pointer to)
    :ivar msf: MSFileparam (pointer to)
    :ivar file: filename
    :ivar offset: Current offset

    :param filename: File to read
    :param startoffset: Offset in bytes to start reading the file
    :param reclen: If reclen is 0 the length of the first record is auto-
        detected. All subsequent records are then expected to have the
        same record length. If reclen is negative the length of every
        record is automatically detected. Defaults to -1.
    :param dataflag: Controls whether data samples are unpacked, defaults
        to False.
    :param skipnotdata: If true (not zero) any data chunks read that to do
        not have valid data record indicators will be skipped. Defaults to
        True (1).
    :param verbose: Controls verbosity from 0 to 2. Defaults to None (0).

    .. rubric:: Notes

    The elements of the MSRecord struct are available through `contents`
    on the pointer to the MSrecord, e.g. `_MSRIterator.msr.contents.reclen`.

    The raw record as a byte array is available at:
    `_MSRIterator.msr.contents.record`
    and can be used with Python I/O routines using:
    `ctypes.string_at(_MSRIterator.msr.contents.record,
    _MSRIterator.msr.contents.reclen))

    .. rubric:: Example

    from msriterator import _MSRIterator

    mseedfile = "test.mseed"

    for msri in _MSRIterator(filename=mseedfile, dataflag=False):

        print ("{:d}: {}, reclen: {}, samples: {}, starttime: {}, endtime: {}".
               format(msri.get_offset(),
                      msri.get_srcname(quality=False),
                      msri.msr.contents.reclen,
                      msri.msr.contents.samplecnt,
                      msri.get_starttime(),
                      msri.get_endtime()))
    """

    def __init__(self, filename, startoffset=0,
                 reclen=-1, dataflag=0, skipnotdata=1, verbose=0,
                 raise_errors=True):

        self.msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        self.msf = C.POINTER(MSFileParam)()  # NULL pointer
        self.file = filename

        # Best guess at off_t type
        self._c_off_t = C.c_long

        self.fpos = self._c_off_t(0)
        if startoffset != 0:
            self.fpos = self._c_off_t(-startoffset)

        self.reclen = reclen
        self.dataflag = 1 if dataflag else 0
        self.skipnotdata = 1 if skipnotdata else 0
        self.verbose = verbose
        self.raise_errors = raise_errors

    def __iter__(self):
        return self

    def __next__(self):
        """
        Read next record from file.
        """
        errcode = clibmseed.ms_readmsr_r(C.pointer(self.msf),
                                         C.pointer(self.msr),
                                         self.file.encode('ascii', 'strict'),
                                         self.reclen,
                                         C.byref(self.fpos),
                                         None,
                                         self.skipnotdata,
                                         self.dataflag,
                                         self.verbose)

        if errcode == MS_ENDOFFILE:
            raise StopIteration()

        if self.raise_errors:
            if errcode != MS_NOERROR:
                raise Exception("Error %d in ms_readmsr_r" % errcode)

        return self

    def __del__(self):
        """
        Method for deallocating MSFileParam and MSRecord structures.
        """
        errcode = clibmseed.ms_readmsr_r(C.pointer(self.msf),
                                         C.pointer(self.msr),
                                         None, -1, None, None, 0, 0, 0)
        if errcode != MS_NOERROR:
            raise Exception("Error %d in ms_readmsr_r" % errcode)

    def get_srcname(self, quality=False):
        """
        Return record start time
        """
        srcname = C.create_string_buffer(50)

        clibmseed.msr_srcname.argtypes = [C.POINTER(MSRecord),
                                          C.POINTER(C.c_char),
                                          C.c_int]
        clibmseed.msr_srcname.restype = C.POINTER(C.c_char)

        if quality:
            clibmseed.msr_srcname(self.msr, srcname, 1)
        else:
            clibmseed.msr_srcname(self.msr, srcname, 0)

        return srcname.value.decode('utf-8')

    def get_starttime(self):
        """
        Return record start time as UTCDateTime
        """
        hptime = self.msr.contents.starttime
        return UTCDateTime(hptime / HPTMODULUS)

    def get_endtime(self):
        """
        Return record end time as UTCDateTime
        """
        hptime = clibmseed.msr_endtime(self.msr)
        return UTCDateTime(hptime / HPTMODULUS)

    def get_startepoch(self):
        """
        Return record start time as epoch time
        """
        return self.msr.contents.starttime / HPTMODULUS

    def get_endepoch(self):
        """
        Return record end time as epoch time
        """
        hptime = clibmseed.msr_endtime(self.msr)
        return hptime / HPTMODULUS

    def set_offset(self, value):
        """
        Set file reading position
        """
        self.fpos = self._c_off_t(-value)

    def get_offset(self):
        """
        Return offset into file for current record
        """
        return self.fpos.value

    offset = property(get_offset, set_offset)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
