# -*- coding: utf-8 -*-

from obspy.gse2 import libgse2
from obspy.numpy import array
from obspy.util import Stats, DateTime
import os


class GSE2Trace(object):
    __format__ = 'GSE2'
    
    def __init__(self, filename=None, **kwargs):
        if filename:
            self.read(filename, **kwargs)
    
    def read(self, filename, **kwargs):
        if not os.path.exists(filename):
            msg = "File not found '%s'" % (filename)
            raise IOError(msg)
        # read GSE2 file
        header, data = libgse2.read(filename)
        # reset header information
        self.stats = Stats()
        # assign all header entries to stats
        for _i in header.keys():
            setattr(self.stats,_i,header[_i])
        # now assign the common attributes of the Trace class
        # station name
        self.stats.station = header['station']
        # sampling rate in Hz (float)
        self.stats.sampling_rate = header['samp_rate']
        # number of samples/data points (int)
        self.stats.npts = header['n_samps']
        # network ID
        self.stats.network = ""
        # location ID
        self.stats.location = ""
        # channel ID
        self.stats.channel = header['channel']
        # data quality indicator
        self.stats.dataquality = ""
        # convert time to seconds since epoch
        seconds = int(header['t_sec'])
        microseconds = int(1e6*(header['t_sec'] - seconds))
        self.stats.starttime = DateTime(
                header['d_year'],header['d_mon'],header['d_day'],
                header['t_hour'],header['t_min'],
                seconds,microseconds
        )
        self.stats.endtime = DateTime.utcfromtimestamp(
            self.stats.starttime.timestamp() +
            header['n_samps']/float(header['samp_rate'])
        )
        self.data=array(data)
    
    def write(self, filename, **kwargs):
        #
        # Assign all attributes which have same name in gse2head
        #
        header = {}
        for _i in libgse2.gse2head:
            try:
                header[_i] = getattr(self.stats,_i)
            except AttributeError:
                pass
        #
        # Translate the common (renamed) entries
        #
        # sampling rate
        if not header['samp_rate']:
            try:
                header['samp_rate'] = self.stats.sampling_rate
            except:
                raise ValueError("No sampling rate given")
        # number of samples
        if not header['n_samps']:
            try:
                header['n_samps'] = self.stats.sampling_rate
            except:
                header['n_samps'] = len(data)
        # year, month, day, hour, min, sec
        if not header['d_day']:
            try:
                (header['d_year'],
                 header['d_mon'],
                 header['d_day'],
                 header['t_hour'],
                 header['t_min'],
                 header['t_sec']) = self.stats.starttime.timetuple()[0:6]
                header['t_sec'] += self.stats.starttime.microseconds/1.0e6
            except:
                pass
        try:
            libgse2.write(header,self.data,filename)
        except TypeError:
            libgse2.write(header,array(self.data,format='l'),filename)
