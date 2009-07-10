#!/usr/bin/python
#-------------------------------------------------------------------
# Filename: libgse2.py
#  Purpose: Python wrapper for gse_functions of Stefan Stange
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Moritz Beyreuther
#---------------------------------------------------------------------
""" 
Python wrapper for gse_functions - The GSE2 library of Stefan Stange.
Currently supports only CM6 compressed GSE2 files, this should be
sufficient for most cases.


This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import platform, os
import ctypes as C
import numpy as N
from obspy.core.util import UTCDateTime, c_file_p, formatScientific


if platform.system() == 'Windows':
    lib_name = 'gse_functions.win32.dll'
else:
    if platform.architecture()[0] == '64bit':
        lib_name = 'gse_functions.lin64.so'
    else:
        lib_name = 'gse_functions.so'

lib = C.CDLL(os.path.join(os.path.dirname(__file__), lib_name))

# Exception type for mismatching checksums
class ChksumError(StandardError):
    pass

# ctypes, PyFile_AsFile: convert python file pointer/ descriptor to C file
# pointer descriptor
C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
C.pythonapi.PyFile_AsFile.restype = c_file_p

## gse_functions read_header
lib.read_header.argtypes = [c_file_p, C.c_void_p]
lib.read_header.restype = C.c_int

## gse_functions decomp_6b
lib.decomp_6b.argtypes = [c_file_p, C.c_int, C.c_void_p]
lib.decomp_6b.restype = C.c_int

# gse_functions rem_2nd_diff
lib.rem_2nd_diff.argtypes = [C.c_void_p, C.c_int]
lib.rem_2nd_diff.restype = C.c_int

# gse_functions check_sum
lib.check_sum.argtypes = [C.c_void_p, C.c_int, C.c_longlong]
lib.check_sum.restype = C.c_int # do not know why not C.c_longlong

# gse_functions buf_init
lib.buf_init.argtypes = [C.c_void_p]
lib.buf_init.restype = C.c_void_p

# gse_functions diff_2nd
lib.diff_2nd.argtypes = [C.c_void_p, C.c_int, C.c_int]
lib.diff_2nd.restype = C.c_void_p

# gse_functions compress_6b
lib.compress_6b.argyptes = [C.c_void_p, C.c_int]
lib.compress_6b.restype = C.c_int

## gse_functions write_header
lib.write_header.argtypes = [c_file_p, C.c_void_p]
lib.write_header.restype = C.c_void_p

## gse_functions buf_dump
lib.buf_dump.argtypes = [c_file_p]
lib.buf_dump.restype = C.c_void_p

# gse_functions buf_free
lib.buf_free.argtypes = [C.c_void_p]
lib.buf_free.restype = C.c_void_p

# gse2 header struct
class HEADER(C.Structure):
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

# module wide variable, can be imported by:
# >>> from obspy.gse2 import gse2head
gse2head = [_i[0] for _i in HEADER._fields_]


def isGse2(f):
    widi = f.read(4)
    if widi != 'WID2':
        raise TypeError("File is not in GSE2 format")
    f.seek(0)


def writeHeader(f, head):
    """
    Rewriting the write_header Function of gse_functions.c

    Different operation systems are delivering different output for the
    scientific format of floats (fprinf libc6). Here we ensure to deliver
    in a for GSE2 valid format independent of the OS. For speed issues we
    simple cut any number ending with E+0XX or E-0XX down to E+XX or E-XX.
    This fails for numbers XX>99, but should not occur.
    """
    calib = formatScientific("%10.4e" % head.calib)
    f.write("WID2 %4d/%02d/%02d %02d:%02d:%06.3f %-5s %-3s %-4s %-3s %8d %11.6f %s %7.3f %-6s %5.1f %4.1f\n" % (
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
            head.vang
        )
    )


def read(infile, test_chksum=False):
    """
    Read GSE2 file and return header and data. 
    
    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases. Data are in circular frequency counts, for
    correction of calper multiply by 2PI and calper: data * 2 * pi *
    header['calper'].
    
    @type file: String
    @param file: Filename of GSE2 file to read, can also be a file pointer
    @type test_chksum: Bool
    @param test_chksum: If True: Test Checksum and raise Exception
    @rtype: Dictionary, Numpy.ndarray int32
    @return: Header entries and data as numpy.ndarray of type int32.
    """
    if type(infile) == file:
        f = infile
    else:
        f = open(infile, "rb")
    isGse2(f)
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    lib.read_header(fp, C.pointer(head))
    #data = (C.c_long * head.n_samps)()
    data = N.zeros(head.n_samps,dtype='int32')
    LP_data = data.ctypes.data_as(C.c_void_p) # Pointer to data
    n = lib.decomp_6b(fp, head.n_samps, LP_data)
    assert n == head.n_samps, "Missmatching length in lib.decomp_6b"
    lib.rem_2nd_diff(LP_data, head.n_samps)
    chksum = C.c_longlong()
    chksum = lib.check_sum(LP_data, head.n_samps, chksum)
    chksum2 = int(f.readline().strip().split()[1])
    if test_chksum and chksum != chksum2:
        msg = "Missmatching Checksums, CHK1 %d; CHK2 %d; %d != %d"
        raise ChksumError(msg % (chksum, chksum2, chksum, chksum2))
    f.close()
    headdict = {}
    for i in head._fields_:
        headdict[i[0]] = getattr(head, i[0])
    #
    # cleaning up
    del fp, head
    if not type(infile) == file:
        f.close()
    # return headdict , data[0:n]
    return headdict , data


def write(headdict, data, outfile):
    """
    Write GSE2 file, given the header and data.
    
    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases. Data are in circular frequency counts, for
    correction of calper multiply by 2PI and calper: 
    data * 2 * pi * header['calper'].
    
    Warning: The data are actually compressed in place for performance
    issues, if you still want to use the data afterwards use data.copy()
    
    @requires: headdict dictionary entries C{'datatype', 'n_samps', 
        'samp_rate'} are absolutely necessary
    @type data: Iterable of longs
    @param data: Contains the data.
    @type outfile: String, File
    @param outfile: Name of GSE2 file to write, can also be a file pointer
    @type headdict: Dictonary
    @param headdict: Header containing the following entries C{
        {
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
        }
        }
    """
    #@requires: headdict dictionary entries datatype, n_samps and samp_rate
    n = len(data)
    # see that data are of type numpy.ndarray, dtype int32
    assert type(data) == N.ndarray, "Error, data need to be int32 numpy ndarray"
    assert data.dtype == 'int32', "Error, data need to be int32 numpy ndarray"
    # Maximum values above 2^26 will result in corrupted/wrong data!
    if data.max() > 2 ** 26:
        raise OverflowError("Compression Error, data must be less equal 2^26")
    tr = data.ctypes.data_as(C.c_void_p)
    #tr = N.ctypeslib.as_ctypes(data)
    lib.buf_init(None)
    if type(outfile) == file:
        f = outfile
    else:
        f = open(outfile, "wb")
    fp = C.pythonapi.PyFile_AsFile(f)
    chksum = C.c_longlong()
    chksum = abs(lib.check_sum(tr, n, chksum))
    lib.diff_2nd(tr, n, 0)
    ierr = lib.compress_6b(tr, n)
    assert ierr == 0, "Error status after compression is NOT 0 but %d" % ierr
    #
    head = HEADER()
    for _i in headdict.keys():
        if _i in gse2head:
            setattr(head, _i, headdict[_i])
    # We leave this function ONLY for type checking, as the file pointer is
    # seeked to pos the header is overwritten!
    pos = f.tell()
    lib.write_header(fp, C.pointer(head))
    f.seek(pos)
    # This is the actual function where the header is written. It avoids
    # the different format of 10.4e with fprintf on windows and linux.
    # For further details, see the __doc__ of writeHeader
    writeHeader(f, head)
    lib.buf_dump(fp)
    f.write("CHK2 %8ld\n\n" % chksum)
    lib.buf_free(None)
    del fp, head
    if not type(outfile) == file:
        f.close()
    return 0


def readHead(file):
    """
    Return (and read) only the header of gse2 file as dictionary.

    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases.

    @type file: String
    @param file: Name of GSE2 file.
    @rtype: Dictonary
    @return: Header entries.
    """
    f = open(file, "rb")
    isGse2(f)
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    lib.read_header(fp, C.pointer(head))
    f.close()
    headdict = {}
    for i in head._fields_:
        headdict[i[0]] = getattr(head, i[0])
    del fp, head
    return headdict


def getStartAndEndTime(file):
    """
    Return start and endtime/date of gse2 file
    
    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases.

    @type file: String
    @param file: Name of GSE2 file.
    @rtype: List
    @return: C{[startdate,stopdate,startime,stoptime]} Start and Stop time as
        Julian seconds and as date string.
    """
    f = open(file, "rb")
    isGse2(f)
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    lib.read_header(fp, C.pointer(head))
    f.close()
    seconds = int(head.t_sec)
    microseconds = int(1e6 * (head.t_sec - seconds))
    startdate = UTCDateTime(head.d_year, head.d_mon, head.d_day,
                            head.t_hour, head.t_min, seconds, microseconds)
    stopdate = UTCDateTime(startdate.timestamp +
                           head.n_samps / float(head.samp_rate))
    del fp, head
    return [startdate, stopdate, startdate.timestamp, stopdate.timestamp]
