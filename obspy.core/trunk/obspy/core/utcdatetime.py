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
        >>> UTCDateTime("2009,010,19:59:42.1800")
        UTCDateTime(2009, 1, 10, 19, 59, 42, 180000)
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
        """
        Creates a new UTCDateTime object.
        """
        if len(args) == 0 and len(kwargs) == 0:
            dt = datetime.datetime.utcnow()
            return UTCDateTime._new(cls, dt)
        elif len(args) == 1:
            value = args[0]
            # check types
            if type(value) in [int, long, float]:
                # got a timestamp
                dt = datetime.datetime.utcfromtimestamp(value)
                return UTCDateTime._new(cls, dt)
            elif isinstance(value, datetime.datetime):
                # got a Python datetime.datetime object
                return UTCDateTime._new(cls, value)
            elif isinstance(value, datetime.date):
                # got a Python datetime.date object
                return datetime.datetime.__new__(cls, value.year, value.month,
                                                 value.day)
            elif isinstance(value, basestring):
                # got a string instance
                value = value.strip()
                # check for ISO8601 date string
                if value.count("T") == 1 or 'iso8601' in kwargs:
                    try:
                        dt = UTCDateTime._parseISO8601(value)
                        return UTCDateTime._new(cls, dt)
                    except:
                        if 'iso8601' in kwargs:
                            raise
                        pass
                # try to apply some standard patterns
                value = value.replace('T', ' ')
                value = value.replace('-', ' ')
                value = value.replace(':', ' ')
                value = value.replace(',', ' ')
                value = value.replace('Z', ' ')
                value = value.replace('W', ' ')
                # check for ordinal date (julian date)
                parts = value.split(' ')
                if len(parts) == 1 and len(value) == 7 and value.isdigit():
                    # looks like an compact ordinal date string
                    patterns = ["%Y%j"]
                elif len(parts) > 1 and len(parts[1]) == 3 and \
                   parts[1].isdigit():
                    # looks like an ordinal date string
                    patterns = ["%Y%j%H%M%S", "%Y%j"]
                else:
                    # standard date string
                    patterns = ["%Y%m%d%H%M%S", "%Y%m%d"]
                value = value.replace(' ', '')
                ms = 0
                if '.' in value:
                    parts = value.split('.')
                    value = parts[0].strip()
                    try:
                        ms = float('.' + parts[1].strip())
                    except:
                        pass
                # The module copy.deepcopy passes a (binary) string to
                # UTCDateTime which contains the class specifications. If 
                # argument is not a digit by now, it must be a binary string 
                # and we pass it to L{datetime.datetime},
                if not value.isdigit():
                    return datetime.datetime.__new__(cls, *args, **kwargs)
                dt = None
                for pattern in patterns:
                    try:
                        dt = datetime.datetime.strptime(value, pattern)
                    except:
                        continue
                    else:
                        break
                if dt:
                    dt = UTCDateTime(dt) + ms
                    return UTCDateTime._new(cls, dt)
        # check for ordinal/julian date kwargs
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

    @staticmethod
    def _new(cls, dt):
        """
        I'm just a small helper method to create readable source code.
        """
        return datetime.datetime.__new__(cls, dt.year, dt.month, dt.day,
                                         dt.hour, dt.minute, dt.second,
                                         dt.microsecond)

    @staticmethod
    def _parseISO8601(value):
        """
        Parses an ISO8601:2004 date time string.
        """
        # remove trailing 'Z'
        value = value.replace('Z', '')
        # split between date and time
        try:
            (date, time) = value.split("T")
        except:
            date = value
            time = ""
        # split microseconds
        ms = 0
        if '.' in time:
            (time, ms) = time.split(".")
            ms = float('0.' + ms.strip())
        # remove all hyphens in date
        date = date.replace('-', '')
        length_date = len(date)
        # remove colons in time
        time = time.replace(':', '')
        length_time = len(time)
        # guess date pattern
        if date.count('W') == 1 and length_date == 8:
            # we got a week date: YYYYWwwD
            # remove week indicator 'W'
            date = date.replace('W', '')
            date_pattern = "%Y%W%w"
            year = int(date[0:4])
            # [Www] is the week number prefixed by the letter 'W', from W01 
            # through W53.
            # strpftime %W == Week number of the year (Monday as the first day 
            # of the week) as a decimal number [00,53]. All days in a new year 
            # preceding the first Monday are considered to be in week 0.
            week = int(date[4:6]) - 1
            # [D] is the weekday number, from 1 through 7, beginning with 
            # Monday and ending with Sunday.
            # strpftime %w == Weekday as a decimal number [0(Sunday),6]
            day = int(date[6])
            if day == 7:
                day = 0
            date = "%04d%02d%1d" % (year, week, day)
        elif length_date == 7 and date.isdigit():
            # we got a ordinal date: YYYYDDD
            date_pattern = "%Y%j"
        elif length_date == 8 and date.isdigit():
            # we got a calendar date: YYYYMMDD
            date_pattern = "%Y%m%d"
        else:
            raise Exception("Wrong or incomplete ISO8601:2004 date format")
        # check for time zone information
        # note that the zone designator is the actual offset from UTC and 
        # does not include any information on daylight saving time
        delta = 0
        if time.count('+') == 1:
            (time, tz) = time.split('+')
            delta = -1 * (int(tz[0:2]) * 60 + int(tz[2:]))
        elif time.count('-') == 1:
            (time, tz) = time.split('-')
            delta = (int(tz[0:2]) * 60 + int(tz[2:]))
        # guess time pattern
        if length_time == 6 and time.isdigit():
            time_pattern = "%H%M%S"
        elif length_time == 4 and time.isdigit():
            time_pattern = "%H%M"
        elif length_time == 2 and time.isdigit():
            time_pattern = "%H"
        elif length_time == 0:
            time_pattern = ""
        else:
            raise Exception("Wrong or incomplete ISO8601:2004 time format")
        # parse patterns
        dt = datetime.datetime.strptime(date + 'T' + time,
                                        date_pattern + 'T' + time_pattern)
        # add microseconds and eventually correct time zone 
        return UTCDateTime(dt) + (delta + ms)

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

    def getJulday(self):
        return int(self.strftime("%j"))

    julday = property(getJulday)

    def formatArcLink(self):
        """
        Returns string representation for the ArcLink protocol.
        """
        return "%d,%d,%d,%d,%d,%d,%d" % (self.year, self.month, self.day,
                                         self.hour, self.minute, self.second,
                                         self.microsecond)

    def formatSEED(self, compact=False):
        """
        Returns string representation for a SEED volume.
        """
        if not compact:
            if not self.time:
                return "%04d,%03d" % (self.year, self.julday)
            return "%04d,%03d,%02d:%02d:%02d.%04d" % (self.year, self.julday,
                                                      self.hour, self.minute,
                                                      self.second,
                                                      self.microsecond // 100)
        temp = "%04d,%03d" % (self.year, self.julday)
        if not self.time:
            return temp
        temp += ",%02d" % self.hour
        if self.microsecond:
            return temp + ":%02d:%02d.%04d" % (self.minute, self.second,
                                               self.microsecond // 100)
        elif self.second:
            return temp + ":%02d:%02d" % (self.minute, self.second)
        elif self.minute:
            return temp + ":%02d" % (self.minute)
        return temp


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

