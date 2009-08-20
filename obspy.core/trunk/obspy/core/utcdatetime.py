# -*- coding: utf-8 -*-

from calendar import timegm
import datetime


class UTCDateTime(datetime.datetime):
    """
    A class providing a smooth interface to UTC based datetime objects. 
    
    This class inherits from Python's L{datetime.datetime} and refines the UTC 
    time zone support.
    
    Usage:
        >>> UTCDateTime(0.0)
        UTCDateTime(1970, 1, 1, 0, 0)
        >>> UTCDateTime(1970, 1, 1)
        UTCDateTime(1970, 1, 1, 0, 0)
        >>> UTCDateTime(datetime.datetime(2009, 5, 24, 8, 28, 12, 5001))
        UTCDateTime(2009, 5, 24, 8, 28, 12, 5001)
        >>> UTCDateTime("19700101")
        UTCDateTime(1970, 1, 1, 0, 0)
        >>> UTCDateTime("1970-01-01 12:23:34")
        UTCDateTime(1970, 1, 1, 12, 23, 34)
        >>> UTCDateTime("20090701121212")
        UTCDateTime(2009, 7, 1, 12, 12, 12)
        >>> UTCDateTime("1970-01-01T12:23:34.123456")
        UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)
    
        >>> t = UTCDateTime(1240561632.005)
        >>> t
        UTCDateTime(2009, 4, 24, 8, 27, 12, 5000)
        >>> t.year
        2009
        >>> t.year, t.hour, t.month, t.hour, t.minute, t.second, t.microsecond
        (2009, 8, 4, 8, 27, 12, 5000)
        >>> t.timestamp + 100
        1240561732.0050001
        >>> UTCDateTime(t.timestamp+60)
        UTCDateTime(2009, 4, 24, 8, 28, 12, 5000)
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
                arg = arg.replace('Z', '')
                ms = 0
                if '.' in arg:
                    parts = arg.split('.')
                    arg = parts[0].strip()
                    try:
                        ms = float('.' + parts[1].strip())
                    except:
                        pass
                # The module copy.deepcopy passes a (binary) string to
                # UTCDateTime which contains the class specifications. If 
                # argument is not a digit by now, it must be a binary string 
                # and we pass it to L{datetime.datetime},
                if not arg.isdigit():
                    return datetime.datetime.__new__(cls, *args, **kwargs)
                dt = None
                for pattern in ["%Y%m%d%H%M%S", "%Y%m%d"]:
                    try:
                        dt = datetime.datetime.strptime(arg, pattern)
                    except:
                        continue
                    else:
                        break
                if dt:
                    dt = UTCDateTime(dt) + ms
                    return datetime.datetime.__new__(cls, dt.year, dt.month,
                                                     dt.day, dt.hour,
                                                     dt.minute, dt.second,
                                                     dt.microsecond)
        elif len(args) == 0 and len(kwargs) == 0:
            dt = datetime.datetime.utcnow()
            return datetime.datetime.__new__(cls, dt.year, dt.month,
                                             dt.day, dt.hour,
                                             dt.minute, dt.second,
                                             dt.microsecond)
        # check for julian day kwargs
        if 'julday' in kwargs and 'year' in kwargs:
            try:
                temp = "%4d%03d" % (int(kwargs['year']),
                                    int(kwargs['julday']))
                dt = datetime.datetime.strptime(temp, '%Y%j')
            except:
                pass
            else:
                kwargs['month'] = dt.month
                kwargs['day'] = dt.day
                kwargs.pop('julday')
        return datetime.datetime.__new__(cls, *args, **kwargs)

    def getTimeStamp(self):
        """
        Returns UTC time stamp in floating point seconds.
        
        @rtype: float
        @return: Time stamp in seconds
        """
        return float(timegm(self.timetuple())) + self.microsecond / 1.0e6

    timestamp = property(getTimeStamp)

    def getDateTime(self):
        """
        Converts current UTCDateTime to a Python L{datetime.datetime} object.

        @rtype: datetime
        @return: Python datetime object of current UTCDateTime
        """
        return datetime.datetime(self.year, self.month, self.day, self.hour,
                                 self.minute, self.second, self.microsecond)

    datetime = property(getDateTime)

    def getDate(self):
        """
        Converts current UTCDateTime to a Python L{datetime.date} object.

        @rtype: date
        @return: Python date object of current UTCDateTime
        """
        return datetime.date(self.year, self.month, self.day)

    date = property(getDate)

    def getTime(self):
        """
        Converts current UTCDateTime to a Python L{datetime.time} object.

        @rtype: time
        @return: Python time object of current UTCDateTime
        """
        return datetime.time(self.hour, self.minute, self.second,
                             self.microsecond)

    time = property(getTime)

    def __add__(self, *args, **kwargs):
        """
        Adds seconds and microseconds to current L{UTCDateTime} object.
        
        Adding two L{UTCDateTime} objects results into a time span in seconds.
        
        Usage:
            >>> a = UTCDateTime(0.0)
            >>> a
            UTCDateTime(1970, 1, 1, 0, 0)
            >>> a + 1.123456
            UTCDateTime(1970, 1, 1, 0, 0, 1, 123456)
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
                msec = int((arg - sec) * 1000000)
                td = datetime.timedelta(seconds=sec, microseconds=msec)
                dt = datetime.datetime.__add__(self, td)
                return UTCDateTime(dt)
            elif isinstance(arg, UTCDateTime):
                return self.timestamp + arg.timestamp
            else:
                dt = datetime.datetime.__add__(self, arg)
                return UTCDateTime(dt)
        else:
            dt = datetime.datetime.__add__(self, *args, **kwargs)
            return UTCDateTime(dt)

    def __sub__(self, *args, **kwargs):
        """
        Subtracts seconds and microseconds from current L{UTCDateTime} object.
        
        Subtracting two L{UTCDateTime} objects from each other results into a 
        relative time span in seconds.
        
        Usage:
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
                dt = datetime.datetime.__sub__(self, arg)
                return UTCDateTime(dt)
        else:
            dt = datetime.datetime.__sub__(self, *args, **kwargs)
            return UTCDateTime(dt)

    def __str__(self):
        """
        Returns string representation of the L{UTCDateTime} object.
        """
        text = datetime.datetime.__str__(self)
        if not '.' in text:
            text += '.000000'
        return text.replace(' ', 'T') + 'Z'

    def formatArcLink(self):
        """
        Returns string representation for the ArcLink protocol.
        """
        return "%d,%d,%d,%d,%d,%d,%d" % (self.year, self.month, self.day,
                                      self.hour, self.minute, self.second,
                                      self.microsecond)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

