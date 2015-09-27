# -*- coding: utf-8 -*-
"""
Module containing a UTC-based datetime class.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import datetime
import math
import time


TIMESTAMP0 = datetime.datetime(1970, 1, 1, 0, 0)


class UTCDateTime(object):
    """
    A UTC-based datetime object.

    This datetime class is based on the POSIX time, a system for describing
    instants in time, defined as the number of seconds elapsed since midnight
    Coordinated Universal Time (UTC) of Thursday, January 1, 1970. Using a
    single float timestamp allows higher precision as the default Python
    :class:`datetime.datetime` class. It features the full `ISO8601:2004`_
    specification and some additional string patterns during object
    initialization.

    :type args: int, float, str, :class:`datetime.datetime`, optional
    :param args: The creation of a new `UTCDateTime` object depends from the
        given input parameters. All possible options are summarized in the
        `Examples`_ section below.
    :type iso8601: bool, optional
    :param iso8601: Enforce `ISO8601:2004`_ detection. Works only with a string
        as first input argument.
    :type precision: int, optional
    :param precision: Sets the precision used by the rich comparison operators.
        Defaults to ``6`` digits after the decimal point. See also `Precision`_
        section below.

    .. versionchanged:: 0.5.1
        UTCDateTime is no longer based on Python's datetime.datetime class
        instead uses timestamp as a single floating point value which allows
        higher precision.

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

    .. rubric:: _`Examples`

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

        >>> UTCDateTime("20110818_03:00:00")
        UTCDateTime(2011, 8, 18, 3, 0)

    (4) Using multiple arguments in the following order: `year, month,
        day[, hour[, minute[, second[, microsecond]]]`. The year, month and day
        arguments are required.

        >>> UTCDateTime(1970, 1, 1)
        UTCDateTime(1970, 1, 1, 0, 0)

        >>> UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)
        UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)

    (5) Using the following keyword arguments: `year, month, day, julday, hour,
        minute, second, microsecond`. Either the combination of year, month and
        day, or year and Julian day are required.

        >>> UTCDateTime(year=1970, month=1, day=1, minute=15, microsecond=20)
        UTCDateTime(1970, 1, 1, 0, 15, 0, 20)

        >>> UTCDateTime(year=2009, julday=234, hour=14, minute=13)
        UTCDateTime(2009, 8, 22, 14, 13)

    (6) Using a Python :class:`datetime.datetime` object.

        >>> dt = datetime.datetime(2009, 5, 24, 8, 28, 12, 5001)
        >>> UTCDateTime(dt)
        UTCDateTime(2009, 5, 24, 8, 28, 12, 5001)

    .. rubric:: _`Precision`

    The :class:`UTCDateTime` class works with a default precision of ``6``
    digits which effects the comparison of date/time values, e.g.:

    >>> dt = UTCDateTime(0)
    >>> dt2 = UTCDateTime(0.00001)
    >>> dt3 = UTCDateTime(0.0000001)
    >>> print(dt.precision)
    6
    >>> dt == dt2  # 5th digit is within current precision
    False
    >>> dt == dt3  # 7th digit will be neglected
    True

    You may change that behavior either by,

    (1) using the ``precision`` keyword during object initialization:

        >>> dt = UTCDateTime(0, precision=4)
        >>> dt2 = UTCDateTime(0.00001, precision=4)
        >>> print(dt.precision)
        4
        >>> dt == dt2
        True

    (2) or set it the class attribute ``DEFAULT_PRECISION`` for all new
        :class:`UTCDateTime` objects using a monkey patch:

        >>> UTCDateTime.DEFAULT_PRECISION = 4
        >>> dt = UTCDateTime(0)
        >>> dt2 = UTCDateTime(0.00001)
        >>> print(dt.precision)
        4
        >>> dt == dt2
        True

        Don't forget to reset ``DEFAULT_PRECISION`` if not needed anymore!

        >>> UTCDateTime.DEFAULT_PRECISION = 6

    .. _ISO8601:2004: http://en.wikipedia.org/wiki/ISO_8601
    """
    timestamp = 0.0
    DEFAULT_PRECISION = 6

    def __init__(self, *args, **kwargs):
        """
        Creates a new UTCDateTime object.
        """
        # set default precision
        self.precision = kwargs.pop('precision', self.DEFAULT_PRECISION)
        # iso8601 flag
        iso8601 = kwargs.pop('iso8601', False) is True
        # check parameter
        if len(args) == 0 and len(kwargs) == 0:
            # use current time if no time is given
            self.timestamp = time.time()
            return
        elif len(args) == 1 and len(kwargs) == 0:
            value = args[0]
            # check types
            try:
                # got a timestamp
                self.timestamp = value.__float__()
                return
            except:
                pass
            if isinstance(value, datetime.datetime):
                # got a Python datetime.datetime object
                self._from_datetime(value)
                return
            elif isinstance(value, datetime.date):
                # got a Python datetime.date object
                dt = datetime.datetime(value.year, value.month, value.day)
                self._from_datetime(dt)
                return
            elif isinstance(value, (bytes, str)):
                if not isinstance(value, (str, native_str)):
                    value = value.decode()
                # got a string instance
                value = value.strip()
                # check for ISO8601 date string
                if value.count("T") == 1 or iso8601:
                    try:
                        self.timestamp = self._parse_ISO_8601(value).timestamp
                        return
                    except:
                        if iso8601:
                            raise
                # try to apply some standard patterns
                value = value.replace('T', ' ')
                value = value.replace('_', ' ')
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
                # all parts should be digits now - here we filter unknown
                # patterns and pass it directly to Python's  datetime.datetime
                if not ''.join(parts).isdigit():
                    dt = datetime.datetime(*args, **kwargs)
                    self._from_datetime(dt)
                    return
                dt = datetime.datetime.strptime(value, pattern)
                self._from_datetime(dt, ms)
                return
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
            _frac, _sec = math.modf(round(args[5], 6))
            kwargs['microsecond'] = int(round(_frac * 1e6))
            kwargs['second'] = int(_sec)
            args = args[0:5]
        dt = datetime.datetime(*args, **kwargs)
        self._from_datetime(dt)

    def _set(self, **kwargs):
        """
        Sets current timestamp using kwargs.
        """
        year = kwargs.get('year', self.year)
        month = kwargs.get('month', self.month)
        day = kwargs.get('day', self.day)
        hour = kwargs.get('hour', self.hour)
        minute = kwargs.get('minute', self.minute)
        second = kwargs.get('second', self.second)
        microsecond = kwargs.get('microsecond', self.microsecond)
        julday = kwargs.get('julday', None)
        if julday:
            self.timestamp = UTCDateTime(year=year, julday=julday, hour=hour,
                                         minute=minute, second=second,
                                         microsecond=microsecond).timestamp
        else:
            self.timestamp = UTCDateTime(year, month, day, hour, minute,
                                         second, microsecond).timestamp

    def _from_datetime(self, dt, ms=0):
        """
        Use Python datetime object to set current time.

        :type dt: :class:`datetime.datetime`
        :param dt: Python datetime object.
        :type ms: float
        :param ms: extra seconds to add to current UTCDateTime object.
        """
        # see datetime.timedelta.total_seconds
        try:
            td = (dt - TIMESTAMP0)
        except TypeError:
            td = (dt.replace(tzinfo=None) - dt.utcoffset()) - TIMESTAMP0
        self.timestamp = (td.microseconds + (td.seconds + td.days * 86400) *
                          1000000) / 1000000.0 + ms

    @staticmethod
    def _parse_ISO_8601(value):
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
        if time.count('+') == 1 and '+' in time[-6:]:
            (time, tz) = time.rsplit('+')
            delta = -1
        elif time.count('-') == 1 and '-' in time[-6:]:
            (time, tz) = time.rsplit('-')
            delta = 1
        else:
            delta = 0
        if delta:
            while len(tz) < 3:
                tz += '0'
            delta = delta * (int(tz[0:2]) * 60 * 60 + int(tz[2:]) * 60)
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
        return UTCDateTime(dt) + (float(delta) + ms)

    def _get_timestamp(self):
        """
        Returns UTC timestamp in seconds.

        :rtype: float
        :return: Timestamp in seconds.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 123456)
        >>> dt.timestamp
        1222864235.123456
        """
        return self.timestamp

    def __float__(self):
        """
        Returns UTC timestamp in seconds.

        :rtype: float
        :return: Timestamp in seconds.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 123456)
        >>> float(dt)
        1222864235.123456
        """
        return self.timestamp

    def _get_datetime(self):
        """
        Returns a Python datetime object.

        :rtype: :class:`datetime.datetime`
        :return: Python datetime object.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.datetime
        datetime.datetime(2008, 10, 1, 12, 30, 35, 45020)
        """
        # datetime.utcfromtimestamp will cut off but not round
        # avoid through adding timedelta - also avoids the year 2038 problem
        return TIMESTAMP0 + datetime.timedelta(seconds=self.timestamp)

    datetime = property(_get_datetime)

    def _get_date(self):
        """
        Returns a Python date object..

        :rtype: :class:`datetime.date`
        :return: Python date object.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.date
        datetime.date(2008, 10, 1)
        """
        return self._get_datetime().date()

    date = property(_get_date)

    def _get_year(self):
        """
        Returns year of the current UTCDateTime object.

        :rtype: int
        :return: Returns year as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11)
        >>> dt.year
        2012
        """
        return self._get_datetime().year

    def _set_year(self, value):
        """
        Sets year of current UTCDateTime object.

        :param value: Year
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.year = 2010
        >>> dt
        UTCDateTime(2010, 2, 11, 10, 11, 12)
        """
        self._set(year=value)

    year = property(_get_year, _set_year)

    def _get_month(self):
        """
        Returns month as an integer (January is 1, December is 12).

        :rtype: int
        :return: Returns month as an integer, where January is 1 and December
            is 12.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11)
        >>> dt.month
        2
        """
        return self._get_datetime().month

    def _set_month(self, value):
        """
        Sets month of current UTCDateTime object.

        :param value: Month
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.month = 10
        >>> dt
        UTCDateTime(2012, 10, 11, 10, 11, 12)
        """
        self._set(month=value)

    month = property(_get_month, _set_month)

    def _get_day(self):
        """
        Returns day as an integer.

        :rtype: int
        :return: Returns day as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11)
        >>> dt.day
        11
        """
        return self._get_datetime().day

    def _set_day(self, value):
        """
        Sets day of current UTCDateTime object.

        :param value: Day
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.day = 20
        >>> dt
        UTCDateTime(2012, 2, 20, 10, 11, 12)
        """
        self._set(day=value)

    day = property(_get_day, _set_day)

    def _get_weekday(self):
        """
        Return the day of the week as an integer (Monday is 0, Sunday is 6).

        :rtype: int
        :return: Returns day of the week as an integer, where Monday is 0 and
            Sunday is 6.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.weekday
        2
        """
        return self._get_datetime().weekday()

    weekday = property(_get_weekday)

    def _get_time(self):
        """
        Returns a Python time object.

        :rtype: :class:`datetime.time`
        :return: Python time object.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.time
        datetime.time(12, 30, 35, 45020)
        """
        return self._get_datetime().time()

    time = property(_get_time)

    def _get_hour(self):
        """
        Returns hour as an integer.

        :rtype: int
        :return: Returns hour as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.hour
        10
        """
        return self._get_datetime().hour

    def _set_hour(self, value):
        """
        Sets hours of current UTCDateTime object.

        :param value: Hours
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.hour = 20
        >>> dt
        UTCDateTime(2012, 2, 11, 20, 11, 12)
        """
        self._set(hour=value)

    hour = property(_get_hour, _set_hour)

    def _get_minute(self):
        """
        Returns minute as an integer.

        :rtype: int
        :return: Returns minute as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.minute
        11
        """
        return self._get_datetime().minute

    def _set_minute(self, value):
        """
        Sets minutes of current UTCDateTime object.

        :param value: Minutes
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.minute = 20
        >>> dt
        UTCDateTime(2012, 2, 11, 10, 20, 12)
        """
        self._set(minute=value)

    minute = property(_get_minute, _set_minute)

    def _get_second(self):
        """
        Returns seconds as an integer.

        :rtype: int
        :return: Returns seconds as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.second
        12
        """
        return self._get_datetime().second

    def _set_second(self, value):
        """
        Sets seconds of current UTCDateTime object.

        :param value: Seconds
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.second = 20
        >>> dt
        UTCDateTime(2012, 2, 11, 10, 11, 20)
        """
        self.timestamp += value - self.second

    second = property(_get_second, _set_second)

    def _get_microsecond(self):
        """
        Returns microseconds as an integer.

        :rtype: int
        :return: Returns microseconds as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12, 345234)
        >>> dt.microsecond
        345234
        """
        return self._get_datetime().microsecond

    def _set_microsecond(self, value):
        """
        Sets microseconds of current UTCDateTime object.

        :param value: Microseconds
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12, 345234)
        >>> dt.microsecond = 999123
        >>> dt
        UTCDateTime(2012, 2, 11, 10, 11, 12, 999123)
        """
        self._set(microsecond=value)

    microsecond = property(_get_microsecond, _set_microsecond)

    def _get_julday(self):
        """
        Returns Julian day as an integer.

        :rtype: int
        :return: Julian day as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.julday
        275
        """
        return self.utctimetuple().tm_yday

    def _set_julday(self, value):
        """
        Sets Julian day of current UTCDateTime object.

        :param value: Julian day
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 12, 5, 12, 30, 35, 45020)
        >>> dt.julday = 275
        >>> dt
        UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        """
        self._set(julday=value)

    julday = property(_get_julday, _set_julday)

    def timetuple(self):
        """
        Return a time.struct_time such as returned by time.localtime().

        :rtype: time.struct_time
        """
        return self._get_datetime().timetuple()

    def utctimetuple(self):
        """
        Return a time.struct_time of current UTCDateTime object.

        :rtype: time.struct_time
        """
        return self._get_datetime().utctimetuple()

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
        if isinstance(value, datetime.timedelta):
            # see datetime.timedelta.total_seconds
            value = (value.microseconds + (value.seconds + value.days *
                     86400) * 1000000) / 1000000.0
        return UTCDateTime(self.timestamp + value)

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
        if isinstance(value, UTCDateTime):
            return round(self.timestamp - value.timestamp, self.__precision)
        elif isinstance(value, datetime.timedelta):
            # see datetime.timedelta.total_seconds
            value = (value.microseconds + (value.seconds + value.days *
                     86400) * 1000000) / 1000000.0
        return UTCDateTime(self.timestamp - value)

    def __str__(self):
        """
        Returns ISO8601 string representation from current UTCDateTime object.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> str(dt)
        '2008-10-01T12:30:35.045020Z'
        """
        return "%s%sZ" % (self.strftime('%Y-%m-%dT%H:%M:%S'),
                          (self.__ms_pattern % (abs(self.timestamp % 1)))[1:])

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __unicode__(self):
        """
        Returns ISO8601 unicode representation from current UTCDateTime object.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.__unicode__()
        '2008-10-01T12:30:35.045020Z'
        """
        return str(self.__str__())

    def __eq__(self, other):
        """
        Rich comparison operator '=='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 == t2
        True

        But the actual timestamp differ

        >>> t1.timestamp == t2.timestamp
        False

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 == t2
        False
        """
        try:
            return round(self.timestamp - float(other), self.__precision) == 0
        except (TypeError, ValueError):
            return False

    def __ne__(self, other):
        """
        Rich comparison operator '!='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 != t2
        False

        But the actual timestamp differ

        >>> t1.timestamp != t2.timestamp
        True

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 != t2
        True
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Rich comparison operator '<'.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 < t2
        False

        But the actual timestamp differ

        >>> t1.timestamp < t2.timestamp
        True

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 < t2
        True
        """
        try:
            return round(self.timestamp - float(other), self.__precision) < 0
        except (TypeError, ValueError):
            return False

    def __le__(self, other):
        """
        Rich comparison operator '<='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000099)
        >>> t2 = UTCDateTime(123.000000012)
        >>> t1 <= t2
        True

        But the actual timestamp differ

        >>> t1.timestamp <= t2.timestamp
        False

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 <= t2
        False
        """
        try:
            return round(self.timestamp - float(other), self.__precision) <= 0
        except (TypeError, ValueError):
            return False

    def __gt__(self, other):
        """
        Rich comparison operator '>'.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000099)
        >>> t2 = UTCDateTime(123.000000012)
        >>> t1 > t2
        False

        But the actual timestamp differ

        >>> t1.timestamp > t2.timestamp
        True

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 > t2
        True
        """
        try:
            return round(self.timestamp - float(other), self.__precision) > 0
        except (TypeError, ValueError):
            return False

    def __ge__(self, other):
        """
        Rich comparison operator '>='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 >= t2
        True

        But the actual timestamp differ

        >>> t1.timestamp >= t2.timestamp
        False

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 >= t2
        False
        """
        try:
            return round(self.timestamp - float(other), self.__precision) >= 0
        except (TypeError, ValueError):
            return False

    def __repr__(self):
        """
        Returns a representation of UTCDatetime object.
        """
        return 'UTCDateTime' + self._get_datetime().__repr__()[17:]

    def __abs__(self):
        """
        Returns absolute timestamp value of the current UTCDateTime object.
        """
        # needed for unittest.assertAlmostEqual tests on Linux
        return abs(self.timestamp)

    def __hash__(self):
        """
        An object is hashable if it has a hash value which never changes
        during its lifetime. As an UTCDateTime object may change over time,
        it's not hashable. Use the :meth:`~UTCDateTime.datetime()` method to
        generate a :class:`datetime.datetime` object for hashing. But be aware:
        once the UTCDateTime object changes, the hash is not valid anymore.
        """
        # explicitly flag it as unhashable
        return None

    def strftime(self, format):
        """
        Return a string representing the date and time, controlled by an
        explicit format string.

        :type format: str
        :param format: Format string.
        :return: Formatted string representing the date and time.

        Format codes referring to hours, minutes or seconds will see 0 values.
        See methods :meth:`~datetime.datetime.strftime()` and
        :meth:`~datetime.datetime.strptime()` for more information.
        """
        return self._get_datetime().strftime(format)

    def strptime(self, date_string, format):
        """
        Return a UTCDateTime corresponding to date_string, parsed according to
        given format.

        :type date_string: str
        :param date_string: Date and time string.
        :type format: str
        :param format: Format string.
        :return: :class:`~obspy.core.utcdatetime.UTCDateTime`

        See methods :meth:`~datetime.datetime.strftime()` and
        :meth:`~datetime.datetime.strptime()` for more information.
        """
        return UTCDateTime(datetime.datetime.strptime(date_string, format))

    def timetz(self):
        """
        Return time object with same hour, minute, second, microsecond, and
        tzinfo attributes. See also method :meth:`datetime.datetime.time()`.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.timetz()
        datetime.time(12, 30, 35, 45020)
        """
        return self._get_datetime().timetz()

    def utcoffset(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.utcoffset()
        """
        return self._get_datetime().utcoffset()

    def dst(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.dst()
        """
        return self._get_datetime().dst()

    def tzname(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.tzname()
        """
        return self._get_datetime().tzname()

    def ctime(self):
        """
        Return a string representing the date and time.

        .. rubric:: Example

        >>> UTCDateTime(2002, 12, 4, 20, 30, 40).ctime()
        'Wed Dec  4 20:30:40 2002'
        """
        return self._get_datetime().ctime()

    def isoweekday(self):
        """
        Return the day of the week as an integer (Monday is 1, Sunday is 7).

        :rtype: int
        :return: Returns day of the week as an integer, where Monday is 1 and
            Sunday is 7.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.isoweekday()
        3
        """
        return self._get_datetime().isoweekday()

    def isocalendar(self):
        """
        Returns a tuple containing (ISO year, ISO week number, ISO weekday).

        :rtype: tuple of ints
        :return: Returns a tuple containing ISO year, ISO week number and ISO
            weekday.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.isocalendar()
        (2008, 40, 3)
        """
        return self._get_datetime().isocalendar()

    def isoformat(self, sep="T"):
        """
        Return a string representing the date and time in ISO 8601 format.

        :rtype: str
        :return: String representing the date and time in ISO 8601 format like
            YYYY-MM-DDTHH:MM:SS.mmmmmm or, if microsecond is 0,
            YYYY-MM-DDTHH:MM:SS.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.isoformat()
        '2008-10-01T12:30:35.045020'

        >>> dt = UTCDateTime(2008, 10, 1)
        >>> dt.isoformat()
        '2008-10-01T00:00:00'
        """
        return self._get_datetime().isoformat(sep=native_str(sep))

    def format_fissures(self):
        """
        Returns string representation for the IRIS Fissures protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> print(dt.format_fissures())
        2008275T123035.0450Z
        """
        return "%04d%03dT%02d%02d%02d.%04dZ" % \
            (self.year, self.julday, self.hour, self.minute, self.second,
             self.microsecond // 100)

    def format_arclink(self):
        """
        Returns string representation for the ArcLink protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> print(dt.format_arclink())
        2008,10,1,12,30,35,45020
        """
        return "%d,%d,%d,%d,%d,%d,%d" % (self.year, self.month, self.day,
                                         self.hour, self.minute, self.second,
                                         self.microsecond)

    def format_seedlink(self):
        """
        Returns string representation for the SeedLink protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35.45020)
        >>> print(dt.format_seedlink())
        2008,10,1,12,30,35
        """
        # round seconds down to integer
        seconds = int(float(self.second) + float(self.microsecond) / 1.0e6)
        return "%d,%d,%d,%d,%d,%g" % (self.year, self.month, self.day,
                                      self.hour, self.minute, seconds)

    def format_seed(self, compact=False):
        """
        Returns string representation for a SEED volume.

        :type compact: bool, optional
        :param compact: Delivers a compact SEED date string if enabled. Default
            value is set to False.
        :rtype: string
        :return: Datetime string in the SEED format.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> print(dt.format_seed())
        2008,275,12:30:35.0450

        >>> dt = UTCDateTime(2008, 10, 1, 0, 30, 0, 0)
        >>> print(dt.format_seed(compact=True))
        2008,275,00:30
        """
        if not compact:
            if self.time == datetime.time(0):
                return "%04d,%03d" % (self.year, self.julday)
            return "%04d,%03d,%02d:%02d:%02d.%04d" % (self.year, self.julday,
                                                      self.hour, self.minute,
                                                      self.second,
                                                      self.microsecond // 100)
        temp = "%04d,%03d" % (self.year, self.julday)
        if self.time == datetime.time(0):
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

    def format_IRIS_web_service(self):
        """
        Returns string representation usable for the IRIS Web services.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 5, 27, 12, 30, 35, 45020)
        >>> print(dt.format_IRIS_web_service())
        2008-05-27T12:30:35.045
        """
        return "%04d-%02d-%02dT%02d:%02d:%02d.%03d" % \
            (self.year, self.month, self.day, self.hour, self.minute,
             self.second, self.microsecond // 1000)

    def _get_precision(self):
        """
        Returns precision of current UTCDateTime object.

        :return: int

        .. rubric:: Example

        >>> dt = UTCDateTime()
        >>> dt.precision
        6
        """
        return self.__precision

    def _set_precision(self, value=6):
        """
        Set precision of current UTCDateTime object.

        :type value: int, optional
        :param value: Precision value used by the rich comparison operators.
            Defaults to ``6``.

        .. rubric:: Example

        (1) Default precision

            >>> dt = UTCDateTime()
            >>> dt.precision
            6

        (2) Set precision during initialization of UTCDateTime object.

            >>> dt = UTCDateTime(precision=5)
            >>> dt.precision
            5

        (3) Set precision for an existing UTCDateTime object.

            >>> dt = UTCDateTime()
            >>> dt.precision = 12
            >>> dt.precision
            12
        """
        self.__precision = int(value)
        self.__ms_pattern = "%%0.%df" % (self.__precision)

    precision = property(_get_precision, _set_precision)

    def toordinal(self):
        """
        Return proleptic Gregorian ordinal. January 1 of year 1 is day 1.

        See :meth:`datetime.datetime.toordinal()`.

        :return: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 1, 1)
        >>> dt.toordinal()
        734503
        """
        return self._get_datetime().toordinal()

    @staticmethod
    def now():
        """
        Returns current UTC datetime.
        """
        return UTCDateTime()

    @staticmethod
    def utcnow():
        """
        Returns current UTC datetime.
        """
        return UTCDateTime()

    def _get_hours_after_midnight(self):
        """
        Calculate foating point hours after midnight.

        >>> t = UTCDateTime("2015-09-27T03:16:12.123456Z")
        >>> t._get_hours_after_midnight()
        3.270034293333333
        """
        timedelta = (
            self.datetime -
            self.datetime.replace(hour=0, minute=0, second=0, microsecond=0))
        return timedelta.total_seconds() / 3600.0


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
