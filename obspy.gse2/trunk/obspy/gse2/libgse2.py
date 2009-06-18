#!/usr/bin/python
#-------------------------------------------------------------------
# Filename: libgse2.py
#  Purpose: Python wrapper for gse_functions of Stefan Stange
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008 Moritz Beyreuther, Stefan Stange
#---------------------------------------------------------------------
""" 
Contains wrappers for gse_functions - The GSE2 library. Currently supports
only CM6 compressed GSE2 files, this should be sufficient for most cases.

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

import sys, os, time, ctypes as C
#import numpy as N
#C = N.ctypeslib.ctypes

if sys.platform == 'win32':
    lib_name = 'gse_functions.win32.dll'
else:
#    if platform.architecture()[0] == '64bit':
#        lib_name = 'gse_functions.lin64.so'
#    else:
#        lib_name = 'gse_functions.so'
    lib_name = 'gse_functions.so'

lib = C.CDLL(os.path.join(os.path.dirname(__file__), lib_name))

#if sys.platform=='win32':
#    lib = C.cdll.gse_functions
#else:
#    lib = C.CDLL(os.path.join(os.path.dirname(__file__),'gse_functions.so'))

# Exception type for mismatching checksums
class ChksumError(StandardError):
    pass

# C file pointer class
class FILE(C.Structure): # Never directly used
    """C file pointer class for type checking with argtypes"""
    pass
c_file_p = C.POINTER(FILE)

# ctypes, PyFile_AsFile: convert python file pointer to C file pointer
C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
C.pythonapi.PyFile_AsFile.restype = c_file_p

# gse_functions read_header
lib.read_header.argtypes = [c_file_p, C.c_void_p]
lib.read_header.restype = C.c_int

# gse_functions decomp_6b
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

# gse_functions labs
#lib.labs.argtypes = [C.c_int]
#lib.labs.restype = C.c_int

# gse_functions diff_2nd
lib.diff_2nd.argtypes = [C.c_void_p, C.c_int, C.c_int]
lib.diff_2nd.restype = C.c_void_p

# gse_functions compress_6b
lib.compress_6b.argyptes = [C.c_void_p, C.c_int]
lib.compress_6b.restype = C.c_int

# gse_functions write_header
lib.write_header.argtypes = [c_file_p, C.c_void_p]
lib.write_header.restype = C.c_void_p

# gse_functions buf_dump
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

def read(file, test_chksum=False):
    """
    Read GSE2 file and return header and data. 
    
    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases. Data are in circular frequency counts, for
    correction of calper multiply by 2PI and calper: data * 2 * pi *
    header['calper'].

    @type file: String
    @param file: Filename of GSE2 file to read.
    @type test_chksum: Bool
    @param test_chksum: If True: Test Checksum and raise Exception
    @rtype: Dictionary, Interable
    @return: Header entries and data as longs.
    """
    f = open(file, "rb")
    isGse2(f)
    fp = C.pythonapi.PyFile_AsFile(f)
    head = HEADER()
    lib.read_header(fp, C.pointer(head))
    data = (C.c_long * head.n_samps)()
    n = lib.decomp_6b(fp, head.n_samps, data)
    assert n == head.n_samps, "Missmatching length in lib.decomp_6b"
    lib.rem_2nd_diff(data, head.n_samps)
    chksum = C.c_longlong()
    chksum = lib.check_sum(data, head.n_samps, chksum)
    chksum2 = int(f.readline().strip().split()[1])
    if test_chksum and chksum != chksum2:
        raise ChksumError("Missmatching Checksums, CHK1 %d; CHK2 %d; %d != %d" % (chksum, chksum2, chksum, chksum2))
    f.close()
    headdict = {}
    for i in head._fields_:
        headdict[i[0]] = getattr(head, i[0])
    del fp, head
    return headdict , data[0:n]
    # from numpy 1.2.1 it's possible to use:
    ##import numpy as N
    ##return headdict , N.ctypeslib.as_array(data)

def write(headdict, data, file):
    """
    Write GSE2 file, given the header and data.
    
    Currently supports only CM6 compressed GSE2 files, this should be
    sufficient for most cases. Data are in circular frequency counts, for
    correction of calper multiply by 2PI and calper: 
    data * 2 * pi * header['calper'].

    @requires: headdict dictionary entries C{'datatype', 'n_samps', 'samp_rate'} are
        absolutely necessary
    @type data: Iterable of longs
    @param data: Contains the data.
    @type file: String
    @param file: Name of GSE2 file to write.
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
    # Maximum values above 2^26 will result in corrupted/wrong data!
    if max(data) > 2 ** 26:
        raise OverflowError("Compression Error, data must be less equal 2^26")
    tr = (C.c_long * n)()
    try:
        tr[0:n] = data
    except TypeError:
        raise TypeError("GSE2 data must be of type int or long, cast data to long!")
    #tr = data.ctypes.data_as(C.c_void_p)
    #tr = N.ctypeslib.as_ctypes(data)
    lib.buf_init(None)
    f = open(file, "wb")
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
    lib.write_header(fp, C.pointer(head))
    lib.buf_dump(fp)
    f.write("CHK2 %8ld\n\n" % chksum)
    f.close()
    lib.buf_free(None)
    del fp, head
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
    os.environ['TZ'] = 'UTC'
    time.tzset()
    dmsec = head.t_sec - int(head.t_sec)
    datestr = "%04d%02d%02d%02d%02d%02d" % (head.d_year, head.d_mon, head.d_day,
                                       head.t_hour, head.t_min, head.t_sec)
    startime = float(time.mktime(time.strptime(datestr, '%Y%m%d%H%M%S')) + dmsec)
    stoptime = startime + head.n_samps / float(head.samp_rate)
    startdate = "%s.%s" % (time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(startime)),
                           ("%.3f" % startime).split('.')[1])
    stopdate = "%s.%s" % (time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(stoptime)),
                           ("%.3f" % stoptime).split('.')[1])
    del fp, head
    return [startdate, stopdate, startime, stoptime]

#import pdb;pdb.set_trace()

del c_file_p

if __name__ == '__main__':
    import numpy
    numpy.random.seed(815)
    data = numpy.random.random_integers(0, 2 ** 26, 1000).tolist()
    write({'n_samps':len(data)}, data, "test.gse.1")
