# -*- coding: utf-8 -*-

from obspy.core import Trace, UTCDateTime
from obspy.gse2 import libgse2


def isGSE2(filename):
    """
    Checks whether a file is GSE2 or not. Returns True or False.
    
    @param filename: GSE2 file to be read.
    """
    # Open file.
    try:
        fh = open(filename)
    except:
        return False
    try:
        data = fh.read(4)
    except:
        data = False
    fh.close()
    if data == 'WID2':
        return True
    return False


convert_dict = {
    'station': 'station',
    'samp_rate':'sampling_rate',
    'n_samps': 'npts',
    'channel': 'channel',
}

gse2_extra = [
    'instype',
    'datatype',
    'vang',
    'auxid',
    'calper',
    'calib'
]


def readGSE2(filename, **kwargs):
    """
    Reads a GSE2 file and returns an obspy.Trace object.
    
    @param filename: GSE2 file to be read.
    """
    # read GSE2 file
    header, data = libgse2.read(filename)
    # assign all header entries to a new dictionary compatible with an Obspy
    # Trace object.
    new_header = {}
    for i, j in convert_dict.iteritems():
        new_header[j] = header[i]
    # assign gse specific header entries
    new_header['gse2'] = {}
    for i in gse2_extra:
        new_header['gse2'][i] = header[i]
    # Calculate start time.
    new_header['starttime'] = UTCDateTime(
        header['d_year'], header['d_mon'], header['d_day'],
        header['t_hour'], header['t_min'], 0) + header['t_sec']
    new_header['endtime'] = new_header['starttime'] + \
        header['n_samps'] / float(header['samp_rate'])
    return Trace(header=new_header, data=data)


def writeGSE2(stream_object, filename, **kwargs):
    #
    # Translate the common (renamed) entries
    fh = open(filename, 'ab')
    for trace in stream_object:
        header = {}
        for _j, _k in convert_dict.iteritems():
            try:
                header[_j] = trace.stats[_k]
            except:
                pass
        for _j in gse2_extra:
            try:
                header[_j] = trace.stats.gse2[_j]
            except:
                pass
        # year, month, day, hour, min, sec
        try:
            (header['d_year'],
             header['d_mon'],
             header['d_day'],
             header['t_hour'],
             header['t_min'],
             header['t_sec']) = trace.stats.starttime.timetuple()[0:6]
            header['t_sec'] += trace.stats.starttime.microsecond / 1.0e6
        except:
            raise
        try:
            libgse2.write(header, trace.data, fh)
        except AssertionError:
            libgse2.write(header, trace.data.astype('l'), fh)
    fh.close()
