# -*- coding: utf-8 -*-
"""
Module containing a UTC-based datetime class.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from calendar import timegm
import datetime


class UTCDateTime(datetime.datetime):
    """
    A UTC-based datetime object.

    This class inherits from Python :class:`datetime.datetime` class and
    refines the UTC time zone (Coordinated Universal Time) support. It features
    the full `ISO8601:2004`_ specification and some additional string patterns
    during object initialization.

    :type args: int, float, string, :class:`datetime.datetime`, optional
    :param args: The creation of a new `UTCDateTime` object depends from the
        given input parameters. All possible options are summarized in the
        examples section underneath.
    :type iso8601: boolean, optional
    :param iso8601: Enforce `ISO8601:2004`_ detection. Works only with a string
        as first input argument.

    .. rubric:: Supported Operations

    ``UTCDateTime = UTCDateTime + delta``
        Adds/removes ``delta`` seconds (given as int or float) to/from the
        current ``UTCDateTime`` object and returns a new ``UTCDateTime``
        object.
        See also: :meth:`~obspy.core.utcdatetime.UTCDateTime.__add__`.

    ``delta = UTCDateTime - UTCDateTime``
        Calculates the time difference in seconds between two ``UTCDateTime``
        objects. The time difference is given as float data type and may also
        contain a negative number.
        See also: :meth:`~obspy.core.utcdatetime.UTCDateTime.__sub__`.

    .. rubric:: Examples

    (1) Using a timestamp.

        >>> UTCDateTime(0)
        UTCDateTime(1970, 1, 1, 0, 0)

        >>> UTCDateTime(1240561632)
        UTCDateTime(2009, 4, 24, 8, 27, 12)

        >>> UTCDateTime(1240561632.5)
        UTCDateTime(2009, 4, 24, 8, 27, 12, 500000)

    (2) Using a `ISO8601:2004`_ string. The detection may be enforced by
        setting the ``iso8601`` parameter to True.

        * Calendar date representation.

            >>> UTCDateTime("2009-12-31T12:23:34.5")
            UTCDateTime(2009, 12, 31, 12, 23, 34, 500000)

            >>> UTCDateTime("20091231T122334.5")           # compact
            UTCDateTime(2009, 12, 31, 12, 23, 34, 500000)

            >>> UTCDateTime("2009-12-31T12:23:34.5Z")      # w/o time zone
            UTCDateTime(2009, 12, 31, 12, 23, 34, 500000)

            >>> UTCDateTime("2009-12-31T12:23:34+01:15")   # w/ time zone
            UTCDateTime(2009, 12, 31, 11, 8, 34)

        * Ordinal date representation.

            >>> UTCDateTime("2009-365T12:23:34.5")
            UTCDateTime(2009, 12, 31, 12, 23, 34, 500000)

            >>> UTCDateTime("2009365T122334.5")            # compact
            UTCDateTime(2009, 12, 31, 12, 23, 34, 500000)

            >>> UTCDateTime("2009001", iso8601=True)       # enforce ISO8601
            UTCDateTime(2009, 1, 1, 0, 0)

        * Week date representation.

            >>> UTCDateTime("2009-W53-7T12:23:34.5")
            UTCDateTime(2010, 1, 3, 12, 23, 34, 500000)

            >>> UTCDateTime("2009W537T122334.5")           # compact
            UTCDateTime(2010, 1, 3, 12, 23, 34, 500000)

            >>> UTCDateTime("2009W011", iso8601=True)      # enforce ISO8601
            UTCDateTime(2008, 12, 29, 0, 0)

    (3) Using not ISO8601 compatible strings.

        >>> UTCDateTime("1970-01-01 12:23:34")
        UTCDateTime(1970, 1, 1, 12, 23, 34)

        >>> UTCDateTime("1970,01,01,12:23:34")
        UTCDateTime(1970, 1, 1, 12, 23, 34)

        >>> UTCDateTime("1970,001,12:23:34")
        UTCDateTime(1970, 1, 1, 12, 23, 34)

        >>> UTCDateTime("20090701121212")
        UTCDateTime(2009, 7, 1, 12, 12, 12)

        >>> UTCDateTime("19700101")
        UTCDateTime(1970, 1, 1, 0, 0)

    (4) Using multiple arguments in the following order: `year, month,
        day[, hour[, minute[, second[, microsecond]]]`. The year, month and day
        arguments are required.

        >>> UTCDateTime(1970, 1, 1)
        UTCDateTime(1970, 1, 1, 0, 0)

        >>> UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)
        UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)

    (5) Using the following keyword arguments: `year, month, day, julday, hour,
        minute, second, microsecond`. Either the combination of year, month and
        day, or year and julday are required.

        >>> UTCDateTime(year=1970, month=1, day=1, minute=15, microsecond=20)
        UTCDateTime(1970, 1, 1, 0, 15, 0, 20)

        >>> UTCDateTime(year=2009, julday=234, hour=14, minute=13)
        UTCDateTime(2009, 8, 22, 14, 13)

    (6) Using a Python :class:`datetime.datetime` object.

        >>> dt = datetime.datetime(2009, 5, 24, 8, 28, 12, 5001)
        >>> UTCDateTime(dt)
        UTCDateTime(2009, 5, 24, 8, 28, 12, 5001)

    .. _ISO8601:2004: http://en.wikipedia.org/wiki/ISO_8601
    """

    def __new__(cls, *args, **kwargs):
        """
        Creates a new UTCDateTime object.
        """
        # iso8601 flag
        iso8601 = 'iso8601' in kwargs
        if iso8601:
            kwargs.pop('iso8601')
        if len(args) == 0 and len(kwargs) == 0:
            dt = datetime.datetime.utcnow()
            return UTCDateTime._new(cls, dt)
        elif len(args) == 1 and len(kwargs) == 0:
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
                if value.count("T") == 1 or iso8601:
                    try:
                        dt = UTCDateTime._parseISO8601(value)
                        return UTCDateTime._new(cls, dt)
                    except:
                        if iso8601:
                            raise
                # try to apply some standard patterns
                value = value.replace('T', ' ')
                value = value.replace('-', ' ')
                value = value.replace(':', ' ')
                value = value.replace(',', ' ')
                value = value.replace('Z', ' ')
                value = value.replace('W', ' ')
                # check for ordinal date (julian date)
                parts = value.split(' ')
                # check for patterns
                if len(parts) == 1 and len(value) == 7 and value.isdigit():
                    # looks like an compact ordinal date string
                    pattern = "%Y%j"
                elif len(parts) > 1 and len(parts[1]) == 3 and \
                   parts[1].isdigit():
                    # looks like an ordinal date string
                    value = ''.join(parts)
                    if len(parts) > 2:
                        pattern = "%Y%j%H%M%S"
                    else:
                        pattern = "%Y%j"
                else:
                    # some parts should have 2 digits
                    for i in range(1, min(len(parts), 6)):
                        if len(parts[i]) == 1:
                            parts[i] = '0' + parts[i]
                    # standard date string
                    value = ''.join(parts)
                    if len(value) > 8:
                        pattern = "%Y%m%d%H%M%S"
                    else:
                        pattern = "%Y%m%d"
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
                # and we pass it to datetime.datetime,
                if not ''.join(parts).isdigit():
                    return datetime.datetime.__new__(cls, *args, **kwargs)
                dt = datetime.datetime.strptime(value, pattern)
                dt = UTCDateTime(dt) + ms
                return UTCDateTime._new(cls, dt)
        # check for ordinal/julian date kwargs
        if 'julday' in kwargs:
            if 'year' in kwargs:
                # year given as kwargs
                year = kwargs['year']
            elif len(args) == 1:
                # year is first (and only) argument
                year = args[0]
            try:
                temp = "%4d%03d" % (int(year),
                                    int(kwargs['julday']))
                dt = datetime.datetime.strptime(temp, '%Y%j')
            except:
                pass
            else:
                kwargs['month'] = dt.month
                kwargs['day'] = dt.day
                kwargs.pop('julday')

        # check if seconds are given as float value
        if len(args) == 6 and isinstance(args[5], float):
            kwargs['microsecond'] = int(args[5] % 1 * 1000000)
            kwargs['second'] = int(args[5])
            args = args[0:5]
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
        # remove all hyphens in date
        date = date.replace('-', '')
        # remove colons in time
        time = time.replace(':', '')
        # guess date pattern
        length_date = len(date)
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
        elif length_date == 7 and date.isdigit() and value.count('-') != 2:
            # we got a ordinal date: YYYYDDD
            date_pattern = "%Y%j"
        elif length_date == 8 and date.isdigit():
            # we got a calendar date: YYYYMMDD
            date_pattern = "%Y%m%d"
        else:
            raise ValueError("Wrong or incomplete ISO8601:2004 date format")
        # check for time zone information
        # note that the zone designator is the actual offset from UTC and
        # does not include any information on daylight saving time
        delta = 0
        if time.count('+') == 1:
            (time, tz) = time.split('+')
            delta = -1 * (int(tz[0:2]) * 60 * 60 + int(tz[2:]) * 60)
        elif time.count('-') == 1:
            (time, tz) = time.split('-')
            delta = int(tz[0:2]) * 60 * 60 + int(tz[2:]) * 60
        # split microseconds
        ms = 0
        if '.' in time:
            (time, ms) = time.split(".")
            ms = float('0.' + ms.strip())
        # guess time pattern
        length_time = len(time)
        if length_time == 6 and time.isdigit():
            time_pattern = "%H%M%S"
        elif length_time == 4 and time.isdigit():
            time_pattern = "%H%M"
        elif length_time == 2 and time.isdigit():
            time_pattern = "%H"
        elif length_time == 0:
            time_pattern = ""
        else:
            raise ValueError("Wrong or incomplete ISO8601:2004 time format")
        # parse patterns
        dt = datetime.datetime.strptime(date + 'T' + time,
                                        date_pattern + 'T' + time_pattern)
        # add microseconds and eventually correct time zone
        return UTCDateTime(dt) + (delta + ms)

    def getTimeStamp(self):
        """
        Returns UTC timestamp in seconds.

        :rtype: float
        :return: Timestamp in seconds

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 123456)
        >>> dt.getTimeStamp()
        1222864235.123456
        """
        return float(timegm(self.timetuple())) + self.microsecond / 1.0e6

    timestamp = property(getTimeStamp)

    def getDateTime(self):
        """
        Returns a Python datetime object from the current UTCDateTime object.

        :rtype: :class:`datetime.datetime`
        :return: Python datetime object of current UTCDateTime object.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.getDateTime()
        datetime.datetime(2008, 10, 1, 12, 30, 35, 45020)
        """
        return datetime.datetime(self.year, self.month, self.day, self.hour,
                                 self.minute, self.second, self.microsecond)

    datetime = property(getDateTime)

    def getDate(self):
        """
        Returns a Python date object from the current UTCDateTime object.

        :rtype: :class:`datetime.date`
        :return: Python date object of current UTCDateTime object.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.getDate()
        datetime.date(2008, 10, 1)
        """
        return datetime.date(self.year, self.month, self.day)

    date = property(getDate)

    def getTime(self):
        """
        Returns a Python time object from the current UTCDateTime object.

        :rtype: :class:`datetime.time`
        :return: Python time object of current UTCDateTime object.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.getTime()
        datetime.time(12, 30, 35, 45020)
        """
        return datetime.time(self.hour, self.minute, self.second,
                             self.microsecond)

    time = property(getTime)

    def getJulday(self):
        """
        Returns the Julian day of the current UTCDateTime object.

        :rtype: int
        :return: Julian day of current UTCDateTime

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.getJulday()
        275
        """
        try:
            return int(self.strftime("%j"))
        except:
            # calculate it
            return (self.date - UTCDateTime(self.year, 1, 1).date).days + 1

    julday = property(getJulday)

    def __add__(self, value):
        """
        Adds seconds and microseconds to current UTCDateTime object.

        :type value: int, float
        :param value: Seconds to add
        :rtype: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :return: New UTCDateTime object.

        .. rubric:: Example

        >>> dt = UTCDateTime(1970, 1, 1, 0, 0)
        >>> dt + 2
        UTCDateTime(1970, 1, 1, 0, 0, 2)

        >>> UTCDateTime(1970, 1, 1, 0, 0) + 1.123456
        UTCDateTime(1970, 1, 1, 0, 0, 1, 123456)
        """
        try:
            frac = value % 1.0
        except:
            dt = datetime.datetime.__add__(self, value)
            return self.__class__(dt)
        if frac == 0.0:
            td = datetime.timedelta(seconds=int(value))
            dt = datetime.datetime.__add__(self, td)
            return self.__class__(dt)
        else:
            sec = int(value)
            msec = int(round((value - sec) * 1000000))
            td = datetime.timedelta(seconds=sec, microseconds=msec)
            dt = datetime.datetime.__add__(self, td)
            return self.__class__(dt)

    def __sub__(self, value):
        """
        Subtracts seconds and microseconds from current UTCDateTime object.

        :type value: int, float or :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param value: Seconds or UTCDateTime object to subtract. Subtracting an
            UTCDateTime objects results into a relative time span in seconds.
        :rtype: :class:`~obspy.core.utcdatetime.UTCDateTime` or float
        :return: New UTCDateTime object or relative time span in seconds.

        .. rubric:: Example

        >>> dt = UTCDateTime(1970, 1, 2, 0, 0)
        >>> dt - 2
        UTCDateTime(1970, 1, 1, 23, 59, 58)

        >>> UTCDateTime(1970, 1, 2, 0, 0) - 1.123456
        UTCDateTime(1970, 1, 1, 23, 59, 58, 876544)

        >>> UTCDateTime(1970, 1, 2, 0, 0) - UTCDateTime(1970, 1, 1, 0, 0)
        86400.0
        """
        if isinstance(value, int):
            td = datetime.timedelta(seconds=value)
            dt = datetime.datetime.__sub__(self, td)
            return self.__class__(dt)
        elif isinstance(value, float):
            sec = int(value)
            msec = int(round((value - sec) * 1000000))
            td = datetime.timedelta(seconds=sec, microseconds=msec)
            dt = datetime.datetime.__sub__(self, td)
            return self.__class__(dt)
        elif isinstance(value, UTCDateTime):
            return round(self.timestamp - value.timestamp, 6)
        else:
            dt = datetime.datetime.__sub__(self, value)
            return self.__class__(dt)

    def __str__(self):
        """
        Returns ISO8601 string representation from current UTCDateTime object.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> str(dt)
        '2008-10-01T12:30:35.045020Z'
        """
        text = datetime.datetime.__str__(self)
        if not '.' in text:
            text += '.000000'
        return text.replace(' ', 'T') + 'Z'

    def formatFissures(self):
        """
        Returns string representation for the Fissures protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.formatFissures()
        '2008275T123035.0450Z'
        """
        return "%04d%03dT%02d%02d%02d.%04dZ" % \
                (self.year, self.julday, self.hour, self.minute, self.second,
                 self.microsecond // 100)

    def formatArcLink(self):
        """
        Returns string representation for the ArcLink protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.formatArcLink()
        '2008,10,1,12,30,35,45020'
        """
        return "%d,%d,%d,%d,%d,%d,%d" % (self.year, self.month, self.day,
                                         self.hour, self.minute, self.second,
                                         self.microsecond)

    def formatSEED(self, compact=False):
        """
        Returns string representation for a SEED volume.

        :type compact: boolean, optional
        :param compact: Delivers a compact SEED date string if enabled. Default
            value is set to False.
        :rtype: string
        :return: Datetime string in the SEED format.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.formatSEED()
        '2008,275,12:30:35.0450'

        >>> dt = UTCDateTime(2008, 10, 1, 0, 30, 0, 0)
        >>> dt.formatSEED(compact=True)
        '2008,275,00:30'
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

    def formatIRISWebService(self):
        """
        Returns string representation usable for the IRIS Web services.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 5, 27, 12, 30, 35, 45020)
        >>> dt.formatIRISWebService()
        '2008-05-27T12:30:35.045'
        """
        return "%04d-%02d-%02dT%02d:%02d:%02d.%03d" % \
                (self.year, self.month, self.day, self.hour, self.minute,
                 self.second, self.microsecond // 1000)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
