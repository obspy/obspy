#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: libgse1.py
#  Purpose: Python wrapper for reading GSE1 files
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2011 Moritz Beyreuther
#---------------------------------------------------------------------
"""
Lowlevel module internally used for handling GSE1 files

:license: GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)
"""

import doctest
from obspy.gse2.libgse2 import uncompress_CM6
from obspy.core import UTCDateTime

def read(f, verify_chksum=True):
    """
    Read GSE1 file and return header and data.

    Currently supports only CM6 compressed GSE1 files, this should be
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
    head = readHead(f)
    data = uncompress_CM6(f, head['npts'], verify_chksum=False)
    return head, data


def readHead(f):
    """
    Reads GSE1 header from filepointer and returns it at dictionary.
    The current position must start with WID1 and is not checked any
    further.
    """
    head = {}
    head['gse1'] = {}
    # first line
    line = f.readline()
    year = int(line[5:10])
    julday = int(line[10:13])
    hour = int(line[14:16])
    minute = int(line[17:19])
    second = int(line[20:22])
    millisec = int(line[23:26])
    head['starttime'] = UTCDateTime(year=year, julday=julday,
                                    hour=hour, minute=minute,
                                    second=second,
                                    microsecond=millisec*1000)
    head['npts'] = int(line[27:35])
    head['station'] = line[36:42].strip()
    head['gse1']['instype'] = line[43:51].strip()
    head['channel'] = "%03s" % line[52:54].strip().upper()
    head['sampling_rate'] = float(line[55:66])
    head['gse1']['type'] = line[67:73].strip()
    head['gse1']['datatype'] = line[74:78].strip()
    head['gse1']['dflag'] = int(line[79:80])
    # second line
    line = f.readline()
    head['calib'] = float(line[0:10])
    head['gse1']['units'] = float(line[10:17])
    head['gse1']['cperiod'] = float(line[18:27])
    head['gse1']['lat'] = float(line[28:37])
    head['gse1']['lon'] = float(line[38:47])
    head['gse1']['alt'] = float(line[48:57])
    head['gse1']['unkown1'] = float(line[58:65])
    head['gse1']['unkown2'] = float(line[66:73])
    head['gse1']['unkown3'] = float(line[74:81])
    head['gse1']['unkown4'] = float(line[74:80])
    return head

if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
