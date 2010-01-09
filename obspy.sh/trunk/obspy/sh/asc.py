# -*- coding: utf-8 -*-

from obspy.core import Stream, Trace, UTCDateTime, Stats
import numpy as np


MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
          'OCT', 'NOV', 'DEC' ]


def isASC(filename):
    """
    Checks whether a file is ASC or not. Returns True or False.

    :param filename: ASC file to be read.
    """
    # first six chars should contain 'DELTA:'
    try:
        temp = open(filename, 'rb').read(6)
    except:
        return False
    if temp != 'DELTA:':
        return False
    return True


def readASC(filename, headonly=False, start=0, length=None, **kwargs):
    """
    Reads a ASC file and returns an L{obspy.Stream} object.

    :param filename: ASC file to be read.
    :rtype: L{obspy.Stream}.
    :return: A ObsPy Stream object.
    """
    fh = open(filename, 'rb')
    # create Stats
    header = Stats()
    header['network'] = ''
    header['location'] = ''
    header['station'] = ''
    header['sh'] = {}
    calib = header['sh']['calib'] = 1
    channel = [' ', ' ', ' ']
    npts = 0
    starttime = UTCDateTime()
    # get headers
    temp = fh.read(1)
    while not temp.isdigit():
        fh.seek(-1, 1)
        temp = fh.readline()
        key, value = temp.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key == 'DELTA':
            header['sampling_rate'] = 1.0 / float(value)
        elif key == 'LENGTH':
            npts = int(value)
        elif key == 'CALIB':
            calib = header['sh']['calib'] = float(value)
        elif key == 'STATION':
            header['station'] = value
        elif key == 'COMP':
            channel[2] = value[0]
        elif key == 'CHAN1':
            channel[0] = value[0]
        elif key == 'CHAN2':
            channel[1] = value[0]
        elif key == 'START':
            # 01-JAN-2009_01:01:01.0
            day = int(value[0:2])
            month = MONTHS.index(value[3:6]) + 1
            year = int(value[7:11])
            hour = int(value[12:14])
            mins = int(value[15:17])
            secs = float(value[18:])
            starttime = UTCDateTime(year, month, day, hour, mins) + secs
        # read next char
        temp = fh.read(1)
    else:
        # jump back to original position
        fh.seek(-1, 1)
    # set channel
    header['channel'] = ''.join(channel)
    # set npts
    npts = npts - start
    if length and length < npts:
        npts = length
    header['npts'] = npts
    # calculate start time
    header['starttime'] = starttime + (start / float(header['sampling_rate']))
    # create stream object
    stream = Stream()
    if headonly:
        # skip data
        stream.append(Trace(header=header))
    else:
        # fetch data
        data = np.loadtxt(fh, dtype='float32').ravel()[start:start + npts]
        # apply calibration factor
        data = data * calib
        stream.append(Trace(data=data, header=header))
    return stream


def writeASC(stream_object, filename, **kwargs):
    """
    Writes a ASC file.

    :type stream_object: L{obspy.Stream}.
    :param stream_object: A ObsPy Stream object.
    :param filename: ASC file to be written.
    """
