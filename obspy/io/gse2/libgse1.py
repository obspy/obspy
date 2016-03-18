#!/usr/bin/env python
# -------------------------------------------------------------------
# Filename: libgse1.py
#  Purpose: Python wrapper for reading GSE1 files
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Moritz Beyreuther
# ---------------------------------------------------------------------
"""
Low-level module internally used for handling GSE1 files

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import doctest
import sys

from obspy import UTCDateTime
from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule

from .libgse2 import uncompress_cm6, verify_checksum, read_integer_data


def read(fh, verify_chksum=True):
    """
    Read GSE1 file and return header and data.

    Currently supports only CM6 compressed and plain integer GSE1 files, this
    should be sufficient for most cases. Data are in circular frequency counts,
    for correction of calper multiply by 2PI and calper: data * 2 * pi *
    header['calper'].

    :type fh: file
    :param fh: Open file pointer of GSE1 file to read, opened in binary mode,
        e.g. fh = open('myfile','rb')
    :type verify_chksum: bool
    :param verify_chksum: If True verify Checksum and raise Exception if not
        correct
    :rtype: Dictionary, :class:`numpy.ndarray`, dtype=int32
    :return: Header entries and data as numpy.ndarray of type int32.
    """
    header = read_header(fh)
    dtype = header['gse1']['datatype']
    if dtype == 'CMP6':
        data = uncompress_cm6(fh, header['npts'])
    elif dtype == 'INTV':
        data = read_integer_data(fh, header['npts'])
    else:
        msg = "Unsupported data type %s in GSE1 file" % (dtype)
        raise NotImplementedError(msg)
    # test checksum only if enabled
    if verify_chksum:
        verify_checksum(fh, data, version=1)
    return header, data


def read_header(fh):
    """
    Reads GSE1 header from file pointer and returns it as dictionary.

    The method searches for the next available WID1 field beginning from the
    current file position.
    """
    # search for WID1 field
    line = fh.readline()
    while line:
        if line.startswith(b"WID1"):
            # valid GSE1 header
            break
        line = fh.readline()
    else:
        raise EOFError
    # fetch header
    header = {}
    header['gse1'] = {}
    # first line
    year = int(line[5:10])
    julday = int(line[10:13])
    hour = int(line[14:16])
    minute = int(line[17:19])
    second = int(line[20:22])
    millisec = int(line[23:26])
    header['starttime'] = UTCDateTime(year=year, julday=julday,
                                      hour=hour, minute=minute,
                                      second=second,
                                      microsecond=millisec * 1000)
    header['npts'] = int(line[27:35])
    header['station'] = line[36:42].strip()
    header['gse1']['instype'] = line[43:51].strip()
    _chan = line[52:54].strip()
    _chan = "%03s" % _chan.decode().upper()
    header['channel'] = _chan.encode('ascii', 'strict')
    header['sampling_rate'] = float(line[55:66])
    header['gse1']['type'] = line[67:73].strip()
    header['gse1']['datatype'] = line[74:78].strip()
    header['gse1']['dflag'] = int(line[79:80])
    # second line
    line = fh.readline()
    header['calib'] = float(line[0:10])
    header['gse1']['units'] = float(line[10:17])
    header['gse1']['cperiod'] = float(line[18:27])
    header['gse1']['lat'] = float(line[28:37])
    header['gse1']['lon'] = float(line[38:47])
    header['gse1']['alt'] = float(line[48:57])
    header['gse1']['unknown1'] = float(line[58:65])
    header['gse1']['unknown2'] = float(line[66:73])
    header['gse1']['unknown3'] = float(line[74:81])
    header['gse1']['unknown4'] = float(line[74:80])
    # Py3k: convert to unicode
    header['gse1'] = dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                          for k, v in header['gse1'].items())
    return dict((k, v.decode()) if isinstance(v, bytes) else (k, v)
                for k, v in header.items())


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    original_module=sys.modules[__name__],
    import_map={},
    function_map={
        'readHeader': 'obspy.io.gse2.libgse1.read_header',
        'readIntegerData': 'obspy.io.gse2.libgse1.read_integer_data'})


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
