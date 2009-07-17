# -*- coding: utf-8 -*-

import datetime
from calendar import timegm


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
                        ms = float('.' + parts[1].strip())
                    except:
                        pass
                # The module copy.deepcopy passes a (binary) string to
                # datetime which contains the class specifictions. If 
                # arg is not a digit by now, it must be a binary string and
                # we pass it to datetime
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
        elif len(args) == 0:
            dt = datetime.datetime.utcnow()
            return datetime.datetime.__new__(cls, dt.year, dt.month,
                                             dt.day, dt.hour,
                                             dt.minute, dt.second,
                                             dt.microsecond)
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

    def __str__(self):
        """
        Returns string representation of the UTCDateTime object.
        """
        text = datetime.datetime.__str__(self)
        if not '.' in text:
            text += '.000000'
        return text.replace(' ', 'T')


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

