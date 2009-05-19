#!/usr/bin/env python
# -*- coding: utf-8 -*-

from calendar import timegm
import datetime


class Stats(dict):
    """
    A stats class which behaves like a dictionary.
    
    You may the following syntax to change or access data in this class:
      >>> stats = Stats()
      >>> stats.network = 'BW'
      >>> stats['station'] = 'ROTZ'
      >>> stats.get('network')
      'BW'
      >>> stats['network']
      'BW'
      >>> stats.station
      'ROTZ'
      >>> x = stats.keys()
      >>> x.sort()
      >>> x
      ['channel', 'dataquality', 'location', 'network', 'npts', 'sampling_rate', 'starttime', 'station']

    @type station: String
    @ivar station: Station name
    @type sampling_rate: Float
    @ivar sampling_rate: Sampling rate
    @type npts: Int
    @ivar npts: Number of data points
    @type network: String
    @ivar network: Stations network code
    @type location: String
    @ivar location: Stations location code
    @type channel: String
    @ivar channel: Channel
    @type dataquality: String
    @ivar dataquality: Data quality
    @type starttime: Datetime Object
    @ivar starttime: Starttime of seismogram
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        # fill some dummy values
        self.station = "dummy"
        self.sampling_rate = 1.0
        self.npts = -1
        self.network = "--"
        self.location = ""
        self.channel = "BHZ"
        self.dataquality = ""
        self.starttime = DateTime.utcfromtimestamp(0.0)


class DateTime(datetime.datetime):
    """
    A class handling conversion from utc datetime to utc timestamps. 
    
    This class inherits from datetime.datetime and refines the UTC timezone 
    support.
    
    You may use the following syntax to change or access data in this class:
        >>> DateTime(0.0)
        DateTime(1970, 1, 1, 0, 0)
        >>> DateTime(1970, 1, 1)
        DateTime(1970, 1, 1, 0, 0)
        >>> t = DateTime(1240561632.005)
        >>> t
        DateTime(2009, 4, 24, 8, 27, 12, 5000)
        >>> t.year
        2009
        >>> t.year, t.hour, t.month, t.hour, t.minute, t.second, t.microsecond
        (2009, 8, 4, 8, 27, 12, 5000)
        >>> t.timestamp() + 100
        1240561732.0050001
        >>> t2 = DateTime(t.timestamp()+60)
        >>> t2
        DateTime(2009, 4, 24, 8, 28, 12, 5000)
        >>> DateTime(datetime.datetime(2009, 5, 24, 8, 28, 12, 5001))
        DateTime(2009, 5, 24, 8, 28, 12, 5001)
        
    """
    def __new__(cls, *args, **kwargs):
        if len(args)==1:
            arg = args[0]
            if type(arg) in [int, long, float]:
                dt = datetime.datetime.utcfromtimestamp(arg)
                return datetime.datetime.__new__(cls, dt.year, dt.month, 
                                                 dt.day, dt.hour, 
                                                 dt.minute, dt.second, 
                                                 dt.microsecond)
            elif isinstance(arg, datetime.datetime):
                dt = arg
                return datetime.datetime.__new__(cls, dt.year, dt.month, 
                                                 dt.day, dt.hour, 
                                                 dt.minute, dt.second, 
                                                 dt.microsecond)
        return datetime.datetime.__new__(cls, *args, **kwargs)
    
    def timestamp(self):
        """
        Return UTC timestamp in floating point seconds
        
        @rtype: float
        @return: Timestamp in seconds
        """
        #XXX: datetime.strftime("%s") is not working in windows
        #os.environ['TZ'] = 'UTC'
        #return float(self.strftime("%s")) + self.microsecond/1.0e6
        return float(timegm(self.timetuple())) + self.microsecond/1.0e6


#class Time(float):
#    """
#    A class handling conversion from utc date to timestamps
#    
#    You may use the following syntax to change or access data in this class:
#        >>> Time(0.0)
#        0.0
#        >>> Time('19700101000000')
#        0.0
#        >>> t = Time(1240561632.005)
#        >>> t.date
#        '2009-04-24T08:27:12.005000'
#        >>> t.year
#        2009
#        >>> t.year, t.hour, t.month, t.hour, t.min, "%9.6f"%t.sec
#        (2009, 8, 4, 8, 27, '12.005000')
#        >>> t
#        1240561632.0050001
#        >>> t + 100
#        1240561732.0050001
#        >>> type(t + 100)
#        <type 'float'>
#        >>> t2 = Time(t+100)
#        >>> t2
#        1240561732.0050001
#        >>> type(t2)
#        <class '__main__.Time'>
#        >>> t2.date
#        '2009-04-24T08:28:52.005000'
#
#    @type year: integer
#    @ivar year: The year
#    @type month: integer
#    @ivar month: The month
#    @type day: integer
#    @ivar day: The day
#    @type hour: integer
#    @ivar hour: The hour
#    @type min: integer
#    @ivar min: The minute
#    @type sec: float
#    @ivar year: The seconds
#    @type date: string
#    @ivar date: The seconds
#    @rtype: float
#    @return: Timestamp (julian seconds/seconds since 1970)
#    """
#    def __new__(cls,T,format="%Y%m%d%H%M%S",tzone='UTC'):
#        os.environ['TZ'] = tzone
#        time.tzset()
#        # Given timestamp
#        if type(T) in [int,long,float]:
#            _timestamp = int(T)
#            _msec = T - int(T)
#        # Given datetime
#        elif type(T) is str:
#            try:
#                [date,_i] = T.split('.')
#                _msec = float("0.%s"%_i)
#            except ValueError:
#                date = T
#                _msec = 0.0
#            _timestamp = time.mktime(time.strptime(date,format))
#        # Given unknown type
#        else:
#            raise TypeError("Time in wrong format, must be number or string")
#        # New style class float initializiation
#        myself = float.__new__(cls,_timestamp + _msec)
#        myself.timetuple = time.gmtime(_timestamp)
#        for _i,_j in zip(['year','month','day','hour','min','sec','wday','yday','dst'], 
#                         myself.timetuple):
#            setattr(myself,_i,_j)
#        myself.sec += _msec
#        myself.date = "%s.%06d" % (time.strftime('%Y-%m-%dT%H:%M:%S',myself.timetuple),
#                                 _msec*1e6+.5)
#        return myself


def getParser():
    """
    Collects all obspy parser classes.
    """
    temp = []
    try:
        from obspy.mseed.core import MSEEDTrace
        temp.append(MSEEDTrace)
    except:
        pass
    try:
        from obspy.gse2.core import GSE2Trace
        temp.append(GSE2Trace)
    except:
        pass
    try:
        from obspy.sac.core import SACTrace
        temp.append(SACTrace)
    except:
        pass
    try:
        from obspy.wav.core import WAVTrace
        temp.append(WAVTrace)
    except:
        pass
    return tuple(temp)


def getStreamParser():
    """
    Collects all obspy parser classes that can handle streams.
    """
    temp = []
    try:
        from obspy.mseed.core import MSEEDStream
        temp.append(MSEEDStream)
    except:
        pass
    return tuple(temp)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
