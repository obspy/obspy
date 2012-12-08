#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: libgse2.py
#  Purpose: Python wrapper for gse_functions of Stefan Stange
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Moritz Beyreuther
#---------------------------------------------------------------------
"""
Lowlevel module internally used for handling GSE2 files

Python wrappers for gse_functions - The GSE2 library of Stefan Stange.
Currently CM6 compressed GSE2 files are supported, this should be
sufficient for most cases. Gse_functions is written in C and
interfaced via python-ctypes.

See: http://www.orfeus-eu.org/Software/softwarelib.html#gse

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from distutils import sysconfig
from obspy import UTCDateTime
from obspy.core.util import c_file_p
import ctypes as C
import doctest
import numpy as np
import os
import platform
import warnings


# Import shared libgse2
# create library names
lib_names = [
     # platform specific library name
    'libgse2-%s-%s-py%s' % (platform.system(), platform.architecture()[0],
        ''.join([str(i) for i in platform.python_version_tuple()[:2]])),
     # fallback for pre-packaged libraries
    'libgse2']
# get default file extension for shared objects
lib_extension, = sysconfig.get_config_vars('SO')
# initialize library
clibgse2 = None
for lib_name in lib_names:
    try:
        clibgse2 = C.CDLL(os.path.join(os.path.dirname(__file__), os.pardir,
                                       'lib', lib_name + lib_extension))
    except Exception, e:
        pass
    else:
        break
if not clibgse2:
    msg = 'Could not load shared library for obspy.gse2.\n\n %s' % (e)
    raise ImportError(msg)


class ChksumError(StandardError):
    """
    Exception type for mismatching checksums
    """
    pass


class GSEUtiError(StandardError):
    """
    Exception type for other errors in GSE_UTI
    """
    pass


# gse2 header struct
class HEADER(C.Structure):
    """
    Ctypes based GSE2 header structure for internal usage.
    """
    _fields_ = [
        ('d_year', C.c_int),
        ('d_mon', C.c_int),
        ('d_day', C.c_int),
        ('t_hour', C.c_int),
        ('t_min', C.c_int),
        ('t_sec', C.c_float),
        ('station', C.c_char * 6),
        ('channel', C.c_char * 4),
        ('auxid', C.c_char * 5),
        ('datatype', C.c_char * 4),
        ('n_samps', C.c_int),
        ('samp_rate', C.c_float),
        ('calib', C.c_float),
        ('calper', C.c_float),
        ('instype', C.c_char * 7),
        ('hang', C.c_float),
        ('vang', C.c_float),
    ]


# ctypes, PyFile_AsFile: convert python file pointer/ descriptor to C file
# pointer descriptor
C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
C.pythonapi.PyFile_AsFile.restype = c_file_p

# reading C memory into buffer which can be converted to numpy array
C.pythonapi.PyBuffer_FromMemory.argtypes = [C.c_void_p, C.c_int]
C.pythonapi.PyBuffer_FromMemory.restype = C.py_object

## gse_functions read_header
clibgse2.read_header.argtypes = [c_file_p, C.POINTER(HEADER)]
clibgse2.read_header.restype = C.c_int

## gse_functions decomp_6b
clibgse2.decomp_6b.argtypes = [
    c_file_p, C.c_int,
    np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='C_CONTIGUOUS')]
clibgse2.decomp_6b.restype = C.c_int

# gse_functions rem_2nd_diff
clibgse2.rem_2nd_diff.argtypes = [
    np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='C_CONTIGUOUS'),
    C.c_int]
clibgse2.rem_2nd_diff.restype = C.c_int

# gse_functions check_sum
clibgse2.check_sum.argtypes = [
    np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='C_CONTIGUOUS'),
    C.c_int, C.c_int32]
clibgse2.check_sum.restype = C.c_int  # do not know why not C.c_int32

# gse_functions buf_init
clibgse2.buf_init.argtypes = [C.c_void_p]
clibgse2.buf_init.restype = C.c_void_p

# gse_functions diff_2nd
clibgse2.diff_2nd.argtypes = [
    np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='C_CONTIGUOUS'),
    C.c_int, C.c_int]
clibgse2.diff_2nd.restype = C.c_void_p

# gse_functions compress_6b
clibgse2.compress_6b.argtypes = [
    np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='C_CONTIGUOUS'),
    C.c_int]
clibgse2.compress_6b.restype = C.c_int

## gse_functions write_header
clibgse2.write_header.argtypes = [c_file_p, C.POINTER(HEADER)]
clibgse2.write_header.restype = C.c_void_p

## gse_functions buf_dump
clibgse2.buf_dump.argtypes = [c_file_p]
clibgse2.buf_dump.restype = C.c_void_p

# gse_functions buf_free
clibgse2.buf_free.argtypes = [C.c_void_p]
clibgse2.buf_free.restype = C.c_void_p

# module wide variable, can be imported by:
# >>> from obspy.gse2 import gse2head
gse2head = [_i[0] for _i in HEADER._fields_]


def isGse2(f):
    """
    Checks whether a file is GSE2 or not. Returns True or False.

    :type f : file pointer
    :param f : file pointer to start of GSE2 file to be checked.
    """
    pos = f.tell()
    widi = f.read(4)
    f.seek(pos)
    if widi != 'WID2':
        raise TypeError("File is not in GSE2 format")


def writeHeader(f, head):
    """
    Rewriting the write_header Function of gse_functions.c

    Different operating systems are delivering different output for the
    scientific format of floats (fprinf libc6). Here we ensure to deliver
    in a for GSE2 valid format independent of the OS. For speed issues we
    simple cut any number ending with E+0XX or E-0XX down to E+XX or E-XX.
    This fails for numbers XX>99, but should not occur.

    :type f: File pointer
    :param f: File pointer to to GSE2 file to write
    :type head: Ctypes struct
    :param head: Ctypes structure to write
    """
    calib = "%10.2e" % (head.calib)
    header = "WID2 %4d/%02d/%02d %02d:%02d:%06.3f %-5s %-3s %-4s %-3s %8d " + \
             "%11.6f %s %7.3f %-6s %5.1f %4.1f\n"
    f.write(header % (
            head.d_year,
            head.d_mon,
            head.d_day,
            head.t_hour,
            head.t_min,
            head.t_sec,
            head.station,
            head.channel,
            head.auxid,
            head.datatype,
            head.n_samps,
            head.samp_rate,
            calib,
            head.calper,
            head.instype,
            head.hang,
            head.vang))


def uncompress_CM6(f, n_samps):
    """
    Uncompress n_samps of CM6 compressed data from file pointer fp.

    :type f: File Pointer
    :param f: File Pointer
    :type n_samps: Int
    :param n_samps: Number of samples
    """
    # transform to a C file pointer
    fp = C.pythonapi.PyFile_AsFile(f)
    data = np.empty(n_samps, dtype='int32')
    n = clibgse2.decomp_6b(fp, n_samps, data)
    if n != n_samps:
        raise GSEUtiError("Mismatching length in lib.decomp_6b")
    clibgse2.rem_2nd_diff(data, n_samps)
    return data


def verifyChecksum(fh, data, version=2):
    """
    Calculate checksum from data, as in gse_driver.c line 60

    :type fh: File Pointer
    :param fh: File Pointer
    :type version: Int
    :param version: GSE version, either 1 or 2, defaults to 2.
    """
    chksum_data = clibgse2.check_sum(data, len(data), C.c_int32(0))
    # find checksum within file
    buf = fh.readline()
    chksum_file = 0
    CHK_LINE = 'CHK%d' % version
    while buf:
        if buf.startswith(CHK_LINE):
            chksum_file = int(buf.strip().split()[1])
            break
        buf = fh.readline()
    if chksum_data != chksum_file:
        # 2012-02-12, should be deleted in a year from now
        if abs(chksum_data) == abs(chksum_file):
            msg = "Checksum differs only in absolute value. If this file " + \
                "was written with ObsPy GSE2, this is due to a bug in " + \
                "the obspy.gse2.write routine (resolved with [3431]), " + \
                "and thus this message can be safely ignored."
            warnings.warn(msg, UserWarning)
            return
        msg = "Mismatching checksums, CHK %d != CHK %d"
        raise ChksumError(msg % (chksum_data, chksum_file))
    return


def read(f, verify_chksum=True):
    """
    Read GSE2 file and return header and data.

    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases. Data are in circular frequency counts, for
    correction of calper multiply by 2PI and calper: data * 2 * pi *
    header['calper'].

    :type f: File Pointer
    :param f: Open file pointer of GSE2 file to read, opened in binary mode,
              e.g. f = open('myfile','rb')
    :type test_chksum: Bool
    :param verify_chksum: If True verify Checksum and raise Exception if it
                          is not correct
    :rtype: Dictionary, Numpy.ndarray int32
    :return: Header entries and data as numpy.ndarray of type int32.
    """
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    errcode = clibgse2.read_header(fp, C.pointer(head))
    if errcode != 0:
        raise GSEUtiError("Error in lib.read_header")
    data = uncompress_CM6(f, head.n_samps)
    # test checksum only if enabled
    if verify_chksum:
        verifyChecksum(f, data, version=2)
    headdict = {}
    for i in head._fields_:
        headdict[i[0]] = getattr(head, i[0])
    # cleaning up
    del fp, head
    return headdict, data


def write(headdict, data, f, inplace=False):
    """
    Write GSE2 file, given the header and data.

    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases. Data are in circular frequency counts, for
    correction of calper multiply by 2PI and calper:
    data * 2 * pi * header['calper'].

    Warning: The data are actually compressed in place for performance
    issues, if you still want to use the data afterwards use data.copy()

    :note: headdict dictionary entries C{'datatype', 'n_samps',
           'samp_rate'} are absolutely necessary
    :type data: numpy.ndarray dtype int32
    :param data: Contains the data.
    :type f: File Pointer
    :param f: Open file pointer of GSE2 file to write, opened in binary
              mode, e.g. f = open('myfile','wb')
    :type inplace: Bool
    :param inplace: If True, do compression not on a copy of the data but
                    on the data itself --- note this will change the data
                    values and make them therefore unusable
    :type headdict: Dictionary
    :param headdict: Header containing the following entries::
        'd_year': int,
        'd_mon': int,
        'd_mon': int,
        'd_day': int,
        't_hour': int,
        't_min': int,
        't_sec': float,
        'station': char*6,
        'station': char*6,
        'channel': char*4,
        'auxid': char*5,
        'datatype': char*4,
        'n_samps': int,
        'samp_rate': float,
        'calib': float,
        'calper': float,
        'instype': char*7,
        'hang': float,
        'vang': float
    """
    fp = C.pythonapi.PyFile_AsFile(f)
    n = len(data)
    clibgse2.buf_init(None)
    #
    chksum = clibgse2.check_sum(data, n, C.c_int32(0))
    # Maximum values above 2^26 will result in corrupted/wrong data!
    # do this after chksum as chksum does the type checking for numpy array
    # for you
    if not inplace:
        data = data.copy()
    if data.max() > 2 ** 26:
        raise OverflowError("Compression Error, data must be less equal 2^26")
    clibgse2.diff_2nd(data, n, 0)
    ierr = clibgse2.compress_6b(data, n)
    assert ierr == 0, "Error status after compression is NOT 0 but %d" % ierr
    # set some defaults if not available and convert header entries
    headdict.setdefault('datatype', 'CM6')
    headdict.setdefault('vang', -1)
    headdict.setdefault('calper', 1.0)
    headdict.setdefault('calib', 1.0)
    head = HEADER()
    for _i in headdict.keys():
        if _i in gse2head:
            setattr(head, _i, headdict[_i])
    # This is the actual function where the header is written. It avoids
    # the different format of 10.4e with fprintf on Windows and Linux.
    # For further details, see the __doc__ of writeHeader
    writeHeader(f, head)
    clibgse2.buf_dump(fp)
    f.write("CHK2 %8ld\n\n" % chksum)
    clibgse2.buf_free(None)
    del fp, head


def readHead(f):
    """
    Return (and read) only the header of gse2 file as dictionary.

    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases.

    :type file: File Pointer
    :param file: Open file pointer of GSE2 file to read, opened in binary
                 mode, e.g. f = open('myfile','rb')
    :rtype: Dictionary
    :return: Header entries.
    """
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    clibgse2.read_header(fp, C.pointer(head))
    headdict = {}
    for i in head._fields_:
        headdict[i[0]] = getattr(head, i[0])
    del fp, head
    return headdict


def getStartAndEndTime(f):
    """
    Return start and endtime/date of GSE2 file

    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases.

    :type f: File Pointer
    :param f: Open file pointer of GSE2 file to read, opened in binary
              mode, e.g. f = open('myfile','rb')
    :rtype: List
    :return: C{[startdate,stopdate,startime,stoptime]} Start and Stop time as
             Julian seconds and as date string.
    """
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    clibgse2.read_header(fp, C.pointer(head))
    seconds = int(head.t_sec)
    microseconds = int(1e6 * (head.t_sec - seconds))
    startdate = UTCDateTime(head.d_year, head.d_mon, head.d_day,
                            head.t_hour, head.t_min, seconds, microseconds)
    stopdate = UTCDateTime(startdate.timestamp +
                           head.n_samps / float(head.samp_rate))
    del fp, head
    return [startdate, stopdate, startdate.timestamp, stopdate.timestamp]


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
