#!/usr/bin/env python
# --------------------------------------------------------------------
# Filename: libgse2.py
#  Purpose: Python wrapper for gse_functions of Stefan Stange
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Moritz Beyreuther
# ---------------------------------------------------------------------
"""
Lowlevel module internally used for handling GSE2 files.

Python wrappers for gse_functions - The GSE2 library of Stefan Stange.
Currently CM6 compressed GSE2 files are supported, this should be
sufficient for most cases. Gse_functions is written in C and
interfaced via python-ctypes.

See: http://www.orfeus-eu.org/software/seismo_softwarelibrary.html#gse

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C
import doctest
import sys
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule
from obspy.core.util.libnames import _load_cdll


# Import shared libgse2
clibgse2 = _load_cdll("gse2")

clibgse2.decomp_6b_buffer.argtypes = [
    C.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.CFUNCTYPE(C.c_char_p, C.POINTER(C.c_char), C.c_void_p), C.c_void_p]
clibgse2.decomp_6b_buffer.restype = C.c_int

clibgse2.rem_2nd_diff.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int]
clibgse2.rem_2nd_diff.restype = C.c_int

clibgse2.check_sum.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.c_int32]
clibgse2.check_sum.restype = C.c_int  # do not know why not C.c_int32

clibgse2.diff_2nd.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int, C.c_int]
clibgse2.diff_2nd.restype = C.c_void_p

clibgse2.compress_6b_buffer.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int,
    C.CFUNCTYPE(C.c_int, C.c_char)]
clibgse2.compress_6b_buffer.restype = C.c_int


class ChksumError(Exception):
    """
    Exception type for mismatching checksums
    """
    pass


class GSEUtiError(Exception):
    """
    Exception type for other errors in GSE_UTI
    """
    pass


# example header of tests/data/loc_RNON20040609200559.z:
#
# WID2 2009/05/18 06:47:20.255 RNHA  EHN      CM6      750  200.000000
# 0123456789012345678901234567890123456789012345678901234567890123456789
# 0         10        20        30        40        50        60
#  9.49e-02   1.000    M24  -1.0 -0.0
# 0123456789012345678901234567890123456789012345678901234567890123456789
# 70        80        90        100
def _str(s):
    return s.strip()

GSE2_FIELDS = [
    # local used date fields
    ('year', 5, 9, int),
    ('month', 10, 12, int),
    ('day', 13, 15, int),
    ('hour', 16, 18, int),
    ('minute', 19, 21, int),
    ('second', 22, 24, int),
    ('microsecond', 25, 28, int),
    # global ObsPy stats names
    ('station', 29, 34, _str),
    ('channel', 35, 38, lambda s: s.strip().upper()),
    ('gse2.auxid', 39, 43, _str),
    ('gse2.datatype', 44, 48, _str),
    ('npts', 48, 56, int),
    ('sampling_rate', 57, 68, float),
    ('calib', 69, 79, float),
    ('gse2.calper', 80, 87, float),
    ('gse2.instype', 88, 94, _str),
    ('gse2.hang', 95, 100, float),
    ('gse2.vang', 101, 105, float),
]


def is_gse2(f):
    """
    Checks whether a file is GSE2 or not. Returns True or False.

    :type f: file
    :param f: file pointer to start of GSE2 file to be checked.
    """
    pos = f.tell()
    widi = f.read(4)
    f.seek(pos)
    if widi != b'WID2':
        raise TypeError("File is not in GSE2 format")


def read_header(fh):
    """
    Reads GSE2 header from file pointer and returns it as dictionary.

    The method searches for the next available WID2 field beginning from the
    current file position.
    """
    # search for WID2 field
    line = fh.readline()
    while line:
        if line.startswith(b'WID2'):
            # valid GSE2 header
            break
        line = fh.readline()
    else:
        raise EOFError
    # fetch header
    header = {}
    header['gse2'] = {}
    for key, start, stop, fct in GSE2_FIELDS:
        value = fct(line[slice(start, stop)])
        if 'gse2.' in key:
            header['gse2'][key[5:]] = value
        else:
            header[key] = value
    # convert and remove date entries from header dict
    header['microsecond'] *= 1000
    date = {k: header.pop(k) for k in
            "year month day hour minute second microsecond".split()}
    header['starttime'] = UTCDateTime(**date)
    # search for STA2 line (mandatory but often omitted in practice)
    # according to manual this has to follow immediately after WID2
    pos = fh.tell()
    line = fh.readline()
    if line.startswith(b'STA2'):
        header2 = parse_sta2(line)
        header['network'] = header2.pop("network")
        header['gse2'].update(header2)
    # in case no STA2 line is encountered we need to rewind the file pointer,
    # otherwise we might miss the DAT2 line afterwards.
    else:
        fh.seek(pos)
    # Py3k: convert to unicode
    header['gse2'] = dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                          for k, v in header['gse2'].items())
    return dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                for k, v in header.items())


def write_header(f, headdict):
    """
    Rewriting the write_header Function of gse_functions.c

    Different operating systems are delivering different output for the
    scientific format of floats (fprintf libc6). Here we ensure to deliver
    in a for GSE2 valid format independent of the OS. For speed issues we
    simple cut any number ending with E+0XX or E-0XX down to E+XX or E-XX.
    This fails for numbers XX>99, but should not occur.

    :type f: file
    :param f: File pointer to to GSE2 file to write
    :type headdict: dict
    :param headdict: ObsPy header
    """
    calib = "%10.2e" % (headdict['calib'])
    date = headdict['starttime']
    fmt = "WID2 %4d/%02d/%02d %02d:%02d:%06.3f %-5s %-3s %-4s %-3s %8d " + \
          "%11.6f %s %7.3f %-6s %5.1f %4.1f\n"
    f.write((fmt % (
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second + date.microsecond / 1e6,
            headdict['station'],
            headdict['channel'],
            headdict['gse2']['auxid'],
            headdict['gse2']['datatype'],
            headdict['npts'],
            headdict['sampling_rate'],
            calib,
            headdict['gse2']['calper'],
            headdict['gse2']['instype'],
            headdict['gse2']['hang'],
            headdict['gse2']['vang'])).encode('ascii', 'strict')
            )
    try:
        sta2_line = compile_sta2(headdict)
    except:
        msg = "GSE2: Error while compiling the STA2 header line, omitting it."
        warnings.warn(msg)
    else:
        f.write(sta2_line)


def uncompress_cm6(f, n_samps):
    """
    Uncompress n_samps of CM6 compressed data from file pointer fp.

    :type f: file
    :param f: File Pointer
    :type n_samps: int
    :param n_samps: Number of samples
    """
    def read83(cbuf, vptr):  # @UnusedVariable
        line = f.readline()
        if line == b'':
            return None
        # avoid buffer overflow through clipping to 82
        sb = C.create_string_buffer(line[:82])
        # copy also null termination "\0", that is max 83 bytes
        C.memmove(C.addressof(cbuf.contents), C.addressof(sb), len(line) + 1)
        return C.addressof(sb)

    cread83 = C.CFUNCTYPE(C.c_char_p, C.POINTER(C.c_char), C.c_void_p)(read83)
    if n_samps == 0:
        data = np.empty(0, dtype=np.int32)
    else:
        # aborts with segmentation fault when n_samps == 0
        data = np.empty(n_samps, dtype=np.int32)
        n = clibgse2.decomp_6b_buffer(n_samps, data, cread83, None)
        if n != n_samps:
            raise GSEUtiError("Mismatching length in lib.decomp_6b")
        clibgse2.rem_2nd_diff(data, n_samps)
    return data


def compress_cm6(data):
    """
    CM6 compress data

    :type data: :class:`numpy.ndarray`, dtype=int32
    :param data: the data to write
    :returns: NumPy chararray containing compressed samples
    """
    data = np.ascontiguousarray(data, np.int32)
    n = len(data)
    count = [0]  # closure, must be container
    # 4 character bytes per int32_t
    carr = np.zeros(n * 4, dtype=native_str('c'))

    def writer(char):
        carr[count[0]] = char
        count[0] += 1
        return 0
    cwriter = C.CFUNCTYPE(C.c_int, C.c_char)(writer)
    ierr = clibgse2.compress_6b_buffer(data, n, cwriter)
    if ierr != 0:
        msg = "Error status after compress_6b_buffer is NOT 0 but %d"
        raise GSEUtiError(msg % ierr)
    cnt = count[0]
    if cnt < 80:
        return carr[:cnt].view(native_str('|S%d' % cnt))
    else:
        return carr[:(cnt // 80 + 1) * 80].view(native_str('|S80'))


def verify_checksum(fh, data, version=2):
    """
    Calculate checksum from data, as in gse_driver.c line 60

    :type fh: file
    :param fh: File Pointer
    :type version: int
    :param version: GSE version, either 1 or 2, defaults to 2.
    """
    chksum_data = clibgse2.check_sum(data, len(data), C.c_int32(0))
    # find checksum within file
    buf = fh.readline()
    chksum_file = 0
    chk_line = ('CHK%d' % version).encode('ascii', 'strict')
    while buf:
        if buf.startswith(chk_line):
            chksum_file = int(buf.strip().split()[1])
            break
        buf = fh.readline()
    if chksum_data != chksum_file:
        # 2012-02-12, should be deleted in a year from now
        if abs(chksum_data) == abs(chksum_file):
            msg = "Checksum differs only in absolute value. If this file " + \
                "was written with ObsPy GSE2, this is due to a bug in " + \
                "the obspy.io.gse2.write routine (resolved with [3431]), " + \
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

    :type f: file
    :param f: Open file pointer of GSE2 file to read, opened in binary mode,
              e.g. f = open('myfile','rb')
    :type test_chksum: bool
    :param verify_chksum: If True verify Checksum and raise Exception if it
                          is not correct
    :rtype: Dictionary, :class:`numpy.ndarray`, dtype=int32
    :return: Header entries and data as numpy.ndarray of type int32.
    """
    headdict = read_header(f)
    data = uncompress_cm6(f, headdict['npts'])
    # test checksum only if enabled
    if verify_chksum:
        verify_checksum(f, data, version=2)
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
    :type data: :class:`numpy.ndarray`, dtype=int32
    :param data: Contains the data.
    :type f: file
    :param f: Open file pointer of GSE2 file to write, opened in binary
              mode, e.g. f = open('myfile','wb')
    :type inplace: bool
    :param inplace: If True, do compression not on a copy of the data but
                    on the data itself --- note this will change the data
                    values and make them therefore unusable
    :type headdict: dict
    :param headdict: ObsPy Header
    """
    n = len(data)
    #
    chksum = clibgse2.check_sum(data, n, C.c_int32(0))
    # Maximum values above 2^26 will result in corrupted/wrong data!
    # do this after chksum as chksum does the type checking for NumPy array
    # for you
    if not inplace:
        data = data.copy()
    if data.max() > 2 ** 26:
        raise OverflowError("Compression Error, data must be less equal 2^26")
    clibgse2.diff_2nd(data, n, 0)
    data_cm6 = compress_cm6(data)
    # set some defaults if not available and convert header entries
    headdict.setdefault('calib', 1.0)
    headdict.setdefault('gse2', {})
    headdict['gse2'].setdefault('auxid', '')
    headdict['gse2'].setdefault('datatype', 'CM6')
    headdict['gse2'].setdefault('calper', 1.0)
    headdict['gse2'].setdefault('instype', '')
    headdict['gse2'].setdefault('hang', -1)
    headdict['gse2'].setdefault('vang', -1)
    # This is the actual function where the header is written. It avoids
    # the different format of 10.4e with fprintf on Windows and Linux.
    # For further details, see the __doc__ of write_header
    write_header(f, headdict)
    f.write(b"DAT2\n")
    for line in data_cm6:
        f.write(line + b"\n")
    f.write(("CHK2 %8ld\n\n" % chksum).encode('ascii', 'strict'))


def parse_sta2(line):
    """
    Parses a string with a GSE2 STA2 header line.

    Official Definition::

        Position Name     Format    Description
           1-4   "STA2"   a4        Must be "STA2"
          6-14   Network  a9        Network identifier
         16-34   Lat      f9.5      Latitude (degrees, S is negative)
         36-45   Lon      f10.5     Longitude (degrees, W is negative)
         47-58   Coordsys a12       Reference coordinate system (e.g., WGS-84)
         60-64   Elev     f5.3      Elevation (km)
         66-70   Edepth   f5.3      Emplacement depth (km)

    Corrected Definition (end column of "Lat" field wrong)::

        Position Name     Format    Description
           1-4   "STA2"   a4        Must be "STA2"
          6-14   Network  a9        Network identifier
         16-24   Lat      f9.5      Latitude (degrees, S is negative)
         26-35   Lon      f10.5     Longitude (degrees, W is negative)
         37-48   Coordsys a12       Reference coordinate system (e.g., WGS-84)
         50-54   Elev     f5.3      Elevation (km)
         56-60   Edepth   f5.3      Emplacement depth (km)

    However, many files in practice do not adhere to these defined fixed
    positions. Here are some real-world examples:

    >>> l = "STA2           -999.0000 -999.00000              -.999 -.999"
    >>> for k, v in sorted(parse_sta2(l).items()):  \
            # doctest: +NORMALIZE_WHITESPACE
    ...     print(k, v)
    coordsys
    edepth -0.999
    elev -0.999
    lat -999.0
    lon -999.0
    network
    >>> l = "STA2 ABCD       12.34567   1.234567 WGS-84       -123.456 1.234"
    >>> for k, v in sorted(parse_sta2(l).items()):
    ...     print(k, v)
    coordsys WGS-84
    edepth 1.234
    elev -123.456
    lat 12.34567
    lon 1.234567
    network ABCD
    """
    header = {}
    try:
        lat = line[15:24].strip()
        if lat:
            lat = float(lat)
        else:
            lat = None
        lon = line[25:35].strip()
        if lon:
            lon = float(lon)
        else:
            lon = None
        elev_edepth = line[48:].strip().split()
        elev, edepth = elev_edepth or (None, None)
        if elev:
            elev = float(elev)
        else:
            elev = None
        if edepth:
            edepth = float(edepth)
        else:
            edepth = None
        header['network'] = line[5:14].strip()
        header['lat'] = lat
        header['lon'] = lon
        header['coordsys'] = line[36:48].strip()
        header['elev'] = elev
        header['edepth'] = edepth
    except:
        msg = 'GSE2: Invalid STA2 header, ignoring.'
        warnings.warn(msg)
        return {}
    else:
        return header


def compile_sta2(stats):
    """
    Returns a STA2 line as a string (including newline at end) from a
    :class:`~obspy.core.stats.Stats` object.
    """
    fmt1 = "STA2 %-9s %9s %10s %-12s "
    fmt2 = "%5s %5s\n"
    # compile first part, problems can only arise with invalid lat/lon values
    # or if coordsys has more than 12 characters. raise in case of problems.
    lat = stats['gse2'].get('lat')
    lon = stats['gse2'].get('lon')
    coordsys = stats['gse2'].get('coordsys')
    line = fmt1 % (
        stats['network'],
        lat is not None and '{:9.5f}'.format(lat) or '',
        lon is not None and '{:10.5f}'.format(lon) or '',
        coordsys or '')
    if len(line) != 49:
        msg = ("GSE2: Invalid header values, unable to compile valid "
               "STA2 line. Omitting STA2 line in output")
        warnings.warn(msg)
        raise Exception()
    # compile second part, in many cases it is impossible to adhere to manual.
    # follow common practice, just not adhere to fixed format strictly.
    elev = stats['gse2'].get('elev')
    edepth = stats['gse2'].get('edepth')
    line = line + fmt2 % (
        elev is not None and '{:5.3f}'.format(elev) or '',
        edepth is not None and '{:5.3f}'.format(edepth) or '')
    for key, value in zip(('elev', 'edepth'), (elev, edepth)):
        if value is None:
            continue
        if len('%5.3f' % stats['gse2'][key]) > 5:
            msg = ("Bad value in GSE2 '%s' header field detected. "
                   "The last two header fields of the STA2 line in the "
                   "output file will deviate from the official fixed "
                   "column format description (because they can not be "
                   "represented as '%%f5.3' properly).") % key
            warnings.warn(msg)
    return line.encode('ascii', 'strict')


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    original_module=sys.modules[__name__],
    import_map={},
    function_map={
        'isGse2': 'obspy.io.gse2.libgse2.is_gse2',
        'readHeader': 'obspy.io.gse2.libgse2.read_header',
        'verifyChecksum': 'obspy.io.gse2.libgse2.verify_checksum',
        'writeHeader': 'obspy.io.gse2.libgse2.write_header',
        'compile_STA2': 'obspy.io.gse2.libgse2.compile_sta2',
        'compress_CM6': 'obspy.io.gse2.libgse2.compress_cm6',
        'parse_STA2': 'obspy.io.gse2.libgse2.parse_sta2',
        'uncompress_CM6': 'obspy.io.gse2.libgse2.uncompress_cm6'
    })


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
