# -*- coding: utf-8 -*-

from obspy.core import Trace
from obspy.gse2 import libgse2
from obspy.core.util import UTCDateTime


def isGSE2(filename):
    """
    Checks whether a file is GSE2 or not. Returns True or False.
    
    @param filename: GSE2 file to be read.
    """
    # Open file.
    try:
        f = open(filename)
    except:
        return False
    try:
        data = f.read(4)
    except:
        data = False
    f.close()
    if data == 'WID2':
        return True
    return False


def readGSE2(filename, **kwargs):
    """
    Reads a GSE2 file and returns an obspy.Trace object.
    
    @param filename: GSE2 file to be read.
    """
    # read GSE2 file
    header, data = libgse2.read(filename)
    # assign all header entries to a new dictionary compatible with an Obspy
    # Trace object.
    new_header = {'station': header['station'], 'sampling_rate' : \
                  header['samp_rate'], 'npts': header['n_samps'], 'channel': \
                  header['channel']}
    # convert time to seconds since epoch
    seconds = int(header['t_sec'])
    microseconds = int(1e6 * (header['t_sec'] - seconds))
    # Calculate start time.
    new_header['starttime'] = UTCDateTime(
            header['d_year'], header['d_mon'], header['d_day'],
            header['t_hour'], header['t_min'],
            seconds, microseconds
    )
    new_header['endtime'] = UTCDateTime.utcfromtimestamp(
        new_header['starttime'].timestamp() +
        header['n_samps'] / float(header['samp_rate'])
    )
    return Trace(header=new_header, data=data)


def writeGSE2(stream_object, filename, **kwargs):
    raise NotImplementedError

#def writeGSE2(filename, **kwargs):
#    #
#    # Assign all attributes which have same name in gse2head
#    #
#    header = {}
#    for _i in libgse2.gse2head:
#        try:
#            header[_i] = getattr(self.stats,_i)
#        except AttributeError:
#            pass
#    #
#    # Translate the common (renamed) entries
#    #
#    # sampling rate
#    if not header['samp_rate']:
#        try:
#            header['samp_rate'] = self.stats.sampling_rate
#        except:
#            raise ValueError("No sampling rate given")
#    # number of samples
#    if not header['n_samps']:
#        try:
#            header['n_samps'] = self.stats.npts
#        except:
#            header['n_samps'] = len(data)
#    # year, month, day, hour, min, sec
#    if not header['d_day']:
#        try:
#            (header['d_year'],
#             header['d_mon'],
#             header['d_day'],
#             header['t_hour'],
#             header['t_min'],
#             header['t_sec']) = self.stats.starttime.timetuple()[0:6]
#            header['t_sec'] += self.stats.starttime.microseconds/1.0e6
#        except:
#            pass
#    try:
#        libgse2.write(header,self.data,filename)
#    except TypeError:
#        libgse2.write(header,array(self.data,format='l'),filename)
