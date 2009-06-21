# -*- coding: utf-8 -*-

from obspy.gse2 import libgse2
from obspy.numpy import array
from obspy.util import Stats, DateTime
from obspy.core import Trace
import os

def isGSE2(filename):
    """
    Checks whether a file is GSE2 or not.
    """
    # Open file.
    f = open(filename)
    if f.read(4) == 'WID2':
        return True
    return False

def readGSE2(filename, **kwargs):
    # read GSE2 file
    header, data = libgse2.read(filename)
    # assign all header entries to a new dictionary compatible with an Obspy
    # Trace object.
    new_header = {'station': header['station'], 'sampling_rate' : \
                  header['samp_rate'], 'npts': header['n_samps'], 'channel': \
                  header['channel']}
    # convert time to seconds since epoch
    seconds = int(header['t_sec'])
    microseconds = int(1e6*(header['t_sec'] - seconds))
    new_header['starttime'] = DateTime(
            header['d_year'],header['d_mon'],header['d_day'],
            header['t_hour'],header['t_min'],
            seconds,microseconds
    )
    new_header['endtime'] = DateTime.utcfromtimestamp(
        new_header['starttime'].timestamp() +
        header['n_samps']/float(header['samp_rate'])
    )
    return Trace(header = new_header, data =array(data))

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
#            header['n_samps'] = self.stats.sampling_rate
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
