# -*- coding: utf-8 -*-

from obspy.gse2 import libgse2
from obspy.numpy import array
from obspy.util import Stats
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
        # station name
        self.stats.station = header['station']
        # convert time to seconds since epoch
        os.environ['TZ'] = 'UTC'
        time.tzset()
        dmsec = header['t_sec'] - int(header['t_sec'])
        datestr = "%04d%02d%02d%02d%02d%02d" % (
                header['d_year'],header['d_mon'],header['d_day'],
                header['t_hour'],header['t_min'],header['t_sec']
        )
        starttime = float(time.mktime(time.strptime(datestr,'%Y%m%d%H%M%S')) + dmsec)
        endtime = starttime + header['n_samps']/float(header['samp_rate'])
        # start time of seismogram in seconds since 1970 (float)
        self.stats.julday = float(starttime/1000000)
        self.stats.starttime = starttime
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
        # calib factor
        self.stats.calib = header['calib']
        # calper factor
        self.stats.calper = header['calper']
        # instrument type
        self.stats.instype = header['instype']
        # vang factor
        self.stats.vang = header['vang']
        # hang factor
        self.stats.hang = header['hang']
        # auxid
        self.stats.auxid = header['auxid']
        # ent time of seismogram in seconds since 1970 (float)
        self.stats.endtime = endtime
        # type, not actually used by libmseed
        #self.stats.type = header['type']
        # the actual seismogram data
        self.data=array()
        self.data.extend(data)
    
    def write(self, filename=None, **kwargs):
        raise NotImplementedError
