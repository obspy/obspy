# -*- coding: utf-8 -*-

from obspy.gse2 import libgse2
from obspy.numpy import array
from obspy.util import Stats, Time
import os, time


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
        #os.environ['TZ'] = 'UTC'
        #time.tzset()
        #dmsec = header['t_sec'] - int(header['t_sec'])
        #datestr = "%04d%02d%02d%02d%02d%02d" % (
        #        header['d_year'],header['d_mon'],header['d_day'],
        #        header['t_hour'],header['t_min'],header['t_sec']
        #)
        #starttime = float(time.mktime(time.strptime(datestr,'%Y%m%d%H%M%S')) + dmsec)
        #endtime = starttime + header['n_samps']/float(header['samp_rate'])
        # start time of seismogram in seconds since 1970 (float)
        #self.stats.julday = float(starttime/1000000)
        #self.stats.starttime = starttime
        #self.stats.endtime = endtime
        # type, not actually used by libmseed
        datestr = "%04d%02d%02d%02d%02d%09.6f" % (
                header['d_year'],header['d_mon'],header['d_day'],
                header['t_hour'],header['t_min'],header['t_sec']
        )
        self.stats.starttime = Time(datestr)
        self.stats.endtime = Time(self.stats.starttime +
                                  header['n_samps']/float(header['samp_rate']))
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
                # seconds till epoch to time
                os.environ['TZ'] = 'UTC'
                dmsec = self.stats.starttime - int(self.stats.starttime) 
                time.tzset()
                (header['d_year'],
                 header['d_mon'],
                 header['d_day'],
                 header['t_hour'],
                 header['t_min'],
                 header['t_sec']) = time.gmtime(int(self.stats.starttime))[0:6]
                header['t_sec'] += dmsec
            except:
                pass
        try:
            libgse2.write(header,self.data,filename)
        except TypeError:
            libgse2.write(header,array(self.data,format='l'),filename)
