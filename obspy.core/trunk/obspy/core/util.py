#!/usr/bin/env python
# -*- coding: utf-8 -*-

from calendar import timegm
import ctypes as C
import datetime
import traceback


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
      >>> x[0:3]
      ['channel', 'dataquality', 'endtime']

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
    @type starttime: obspy.UTCDateTime Object
    @ivar starttime: Starttime of seismogram
    @type endtime: obspy.UTCDateTime Object
    @ivar endtime: Endtime of seismogram
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
        self.starttime = UTCDateTime.utcfromtimestamp(0.0)
        self.endtime = UTCDateTime.utcfromtimestamp(86400.0)

    def is_attr(self, attr, typ, default, length=False, assertation=False,
                verbose=False):
        """
        True if attribute of certain type and optional length exists
        
        Function is probably most useful for checking if necessary
        attributes are provided, e.g. when writing seismograms to file

        >>> p=Stats()
        >>> p.julsec = 0.0
        >>> p.is_attr('julsec',float,1.0)
        True
        >>> p.is_attr('julsec',int,123)
        False
        >>> p.julsec
        123
        """
        # Check if attribute exists
        if not attr in self.__dict__.keys():
            if assertation:
                assert False, "%s attribute of Seismogram required" % attr
            if verbose:
                print "WARNING: %s attribute of Seismogram missing",
                print "forcing", attr, "=", default
            setattr(self, attr, default)
            return False
        # Check if attribute is of correct type
        if not isinstance(getattr(self, attr), typ):
            if assertation:
                assert False, "%s attribute of Seismogram not of type %s" % (attr, typ)
            if verbose:
                print "WARNING: %s attribute of Seismogram not of type %s" % (attr, typ),
                print "forcing", attr, "=", default
            setattr(self, attr, default)
            return False
        # Check if attribute has correct length
        if (length):
            if (len(getattr(self, attr)) != length):
                if assertation:
                    assert False, "%s attribute of Seismogram is != %i" % (attribute, length)
                if verbose:
                    print "%s attribute of Seismogram is != %i" % (attribute, length)
                setattr(self, attr, default)
                return False
        # If not test failed
        return True


class UTCDateTime(datetime.datetime):
    """
    A class handling conversion from utc datetime to utc timestamps. 
    
    This class inherits from datetime.datetime and refines the UTC timezone 
    support.
    
    You may use the following syntax to change or access data in this class:
        >>> UTCDateTime(0.0)
        UTCDateTime(1970, 1, 1, 0, 0)
        >>> UTCDateTime(1970, 1, 1)
        UTCDateTime(1970, 1, 1, 0, 0)
        >>> t = UTCDateTime(1240561632.005)
        >>> t
        UTCDateTime(2009, 4, 24, 8, 27, 12, 5000)
        >>> t.year
        2009
        >>> t.year, t.hour, t.month, t.hour, t.minute, t.second, t.microsecond
        (2009, 8, 4, 8, 27, 12, 5000)
        >>> t.timestamp + 100
        1240561732.0050001
        >>> t2 = UTCDateTime(t.timestamp+60)
        >>> t2
        UTCDateTime(2009, 4, 24, 8, 28, 12, 5000)
        >>> UTCDateTime(datetime.datetime(2009, 5, 24, 8, 28, 12, 5001))
        UTCDateTime(2009, 5, 24, 8, 28, 12, 5001)
        >>> UTCDateTime("19700101")
        UTCDateTime(1970, 1, 1, 0, 0)
        >>> UTCDateTime("1970-01-01 12:23:34")
        UTCDateTime(1970, 1, 1, 12, 23, 34)
        >>> UTCDateTime("20090701121212")
        UTCDateTime(2009, 7, 1, 12, 12, 12)
        >>> a = UTCDateTime("1970-01-01T12:23:34.123456")
        >>> a
        UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)
        >>> str(a)
        '1970-01-01 12:23:34.123456'
    """
    def __new__(cls, *args, **kwargs):
        if len(args) == 1:
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
            elif isinstance(arg, basestring):
                arg = arg.replace('T', '')
                arg = arg.replace('-', '')
                arg = arg.replace(':', '')
                arg = arg.replace(' ', '')
                ms = 0
                if '.' in arg:
                    parts = arg.split('.')
                    arg = parts[0].strip()
                    try:
                        ms = int(parts[1].strip())
                    except:
                        pass
                dt = None
                for pattern in ["%Y%m%d%H%M%S", "%Y%m%d"]:
                    try:
                        dt = datetime.datetime.strptime(arg, pattern)
                    except:
                        continue
                    else:
                        break
                if dt:
                    return datetime.datetime.__new__(cls, dt.year, dt.month,
                                                     dt.day, dt.hour,
                                                     dt.minute, dt.second, ms)
        elif len(args) == 0:
            dt = datetime.datetime.utcnow()
            return datetime.datetime.__new__(cls, dt.year, dt.month,
                                             dt.day, dt.hour,
                                             dt.minute, dt.second,
                                             dt.microsecond)
        # XXX
        print "UTCDateTime.__new__", args, kwargs
        return datetime.datetime.__new__(cls, *args, **kwargs)

    def getTimeStamp(self):
        """
        Returns UTC timestamp in floating point seconds.

        @rtype: float
        @return: Timestamp in seconds
        """
        return float(timegm(self.timetuple())) + self.microsecond / 1.0e6

    timestamp = property(getTimeStamp)

    def getDateTime(self):
        """
        Converts current UTCDateTime object in a Python datetime object.

        @rtype: datetime
        @return: Python datetime object of current UTCDateTime
        """
        return datetime.datetime(self.year, self.month, self.day, self.hour,
                                 self.minute, self.second, self.microsecond)

    datetime = property(getDateTime)

    def __add__(self, *args, **kwargs):
        """
        Adds seconds and microseconds from current UTCDateTime object.
        
            >>> a = UTCDateTime(0.0)
            >>> a
            UTCDateTime(1970, 1, 1, 0, 0)
            >>> a + 1
            UTCDateTime(1970, 1, 1, 0, 0, 1)
            >>> a + 1.123456
            UTCDateTime(1970, 1, 1, 0, 0, 1, 123456)
            >>> a + 60 * 60 * 24 * 31 + 0.1
            UTCDateTime(1970, 2, 1, 0, 0, 0, 100000)
            >>> UTCDateTime(0.5) + UTCDateTime(10.5)
            11.0
        
        @return: UTCDateTime
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                td = datetime.timedelta(seconds=arg)
                dt = datetime.datetime.__add__(self, td)
                return UTCDateTime(dt)
            elif isinstance(arg, float):
                sec = int(arg)
                msec = int((arg % 1) * 1000000)
                td = datetime.timedelta(seconds=sec, microseconds=msec)
                dt = datetime.datetime.__add__(self, td)
                return UTCDateTime(dt)
            elif isinstance(arg, UTCDateTime):
                return self.timestamp + arg.timestamp
        else:
            dt = datetime.datetime.__add__(self, *args, **kwargs)
            return UTCDateTime(dt)

    def __sub__(self, *args, **kwargs):
        """
        Substracts seconds and microseconds from current UTCDateTime object.
        
            >>> a = UTCDateTime(0.0) + 60 * 60 * 24 * 31
            >>> a
            UTCDateTime(1970, 2, 1, 0, 0)
            >>> a - 1
            UTCDateTime(1970, 1, 31, 23, 59, 59)
            >>> a - 1.123456
            UTCDateTime(1970, 1, 31, 23, 59, 58, 876544)
            >>> a - 60 * 60 * 24 * 31
            UTCDateTime(1970, 1, 1, 0, 0)
            >>> UTCDateTime(10.0) - UTCDateTime(9.5)
            0.5
        
        @return: UTCDateTime
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                td = datetime.timedelta(seconds=arg)
                dt = datetime.datetime.__sub__(self, td)
                return UTCDateTime(dt)
            elif isinstance(arg, float):
                sec = int(arg)
                msec = int((arg % 1) * 1000000)
                td = datetime.timedelta(seconds=sec, microseconds=msec)
                dt = datetime.datetime.__sub__(self, td)
                return UTCDateTime(dt)
            elif isinstance(arg, UTCDateTime):
                return self.timestamp - arg.timestamp
        else:
            dt = datetime.datetime.__sub__(self, *args, **kwargs)
            return UTCDateTime(dt)


def getFormatsAndMethods(verbose=False):
    """
    Collects all obspy parser classes.

    @type verbose: Bool
    @param verbose: Print error messages / exceptions while parsing.
    """
    temp = []
    failure = []
    # There is one try-except block for each supported file format.
    try:
        from obspy.mseed.core import isMSEED, readMSEED, writeMSEED
        # The first item is the name of the format, the second the checking function.
        temp.append(['MSEED', isMSEED, readMSEED, writeMSEED])
    except:
        failure.append(traceback.format_exc())
    try:
        from obspy.gse2.core import isGSE2, readGSE2, writeGSE2
        # The first item is the name of the format, the second the checking function.
        temp.append(['GSE2', isGSE2, readGSE2, writeGSE2])
    except:
        failure.append(traceback.format_exc())
    try:
        from obspy.wav.core import isWAV, readWAV, writeWAV
        # The first item is the name of the format, the second the checking function.
        temp.append(['WAV', isWAV, readWAV, writeWAV])
    except:
        failure.append(traceback.format_exc())
    try:
        from obspy.sac.core import isSAC, readSAC, writeSAC
        # The first item is the name of the format, the second the checking function.
        temp.append(['SAC', isSAC, readSAC, writeSAC])
    except:
        failure.append(traceback.format_exc())
    if verbose:
        for _i in xrange(len(failure)):
            print failure[_i]
    return temp


def scoreatpercentile(a, per, limit=(), sort=True):
    """
    Calculates the score at the given 'per' percentile of the sequence a.

    For example, the score at per = 50 is the median.

    If the desired quantile lies between two data points, we interpolate
    between them.

    If the parameter 'limit' is provided, it should be a tuple (lower,
    upper) of two values.  Values of 'a' outside this (closed) interval
    will be ignored.

        >>> a = [1, 2, 3, 4]
        >>> scoreatpercentile(a, 25)
        1.75
        >>> scoreatpercentile(a, 50)
        2.5
        >>> scoreatpercentile(a, 75)
        3.25
        >>> a = [6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36]
        >>> scoreatpercentile(a, 25)
        25.5
        >>> scoreatpercentile(a, 50)
        40
        >>> scoreatpercentile(a, 75)
        42.5

    This method is taken from scipy.stats.scoreatpercentile
    Copyright (c) Gary Strangman
    """
    if sort:
        values = sorted(a)
        if limit:
            values = values[(limit[0] < a) & (a < limit[1])]
    else:
        values = a

    def _interpolate(a, b, fraction):
        return a + (b - a) * fraction;

    idx = per / 100. * (len(values) - 1)
    if (idx % 1 == 0):
        return values[int(idx)]
    else:
        return _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)


# C file pointer class
class FILE(C.Structure): # Never directly used
    """C file pointer class for type checking with argtypes"""
    pass
c_file_p = C.POINTER(FILE)

# Define ctypes arg- and restypes.
#C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
#C.pythonapi.PyFile_AsFile.restype = c_file_p

def formatScientific(value):
    """
    Formats floats in a fixed exponential format.
    
    Different operation systems are delivering different output for the
    exponential format of floats. Here we ensure to deliver in a for SEED
    valid format independent of the OS. For speed issues we simple cut any 
    number ending with E+0XX or E-0XX down to E+XX or E-XX. This fails for 
    numbers XX>99, but should not occur, because the SEED standard does 
    not allow this values either.
    
    Python 2.5.2 (r252:60911, Feb 21 2008, 13:11:45) 
    [MSC v.1310 32 bit (Intel)] on win32
    > '%E' % 2.5
    '2.500000E+000'
    
    Python 2.5.2 (r252:60911, Apr  2 2008, 18:38:52)
    [GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on linux2
    > '%E' % 2.5
    '2.500000E+00'

    Doctest:
    >>> formatScientific("3.4e+002")
    '3.4e+02'
    >>> formatScientific("3.4E+02")
    '3.4E+02'
    """
    if 'e' in value:
        mantissa, exponent = value.split('e')
        return "%se%+03d" % (mantissa, int(exponent))
    elif 'E' in value:
        mantissa, exponent = value.split('E')
        return "%sE%+03d" % (mantissa, int(exponent))
    else:
        msg = "Can't format scientific %s in field %s" % (data, self.field_name)
        raise TypeError(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
