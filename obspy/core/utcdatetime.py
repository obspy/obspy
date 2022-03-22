# -*- coding: utf-8 -*-
"""
Module containing a UTC-based datetime class.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import datetime
import calendar
import math
import operator
import re
import sys
import time
import warnings

import numpy as np
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning


# based on https://www.myintervals.com/blog/2009/05/20/iso-8601, w/ week 53 fix
_ISO8601_REGEX = re.compile(r"""
    ^
    ([\+-]?\d{4}(?!\d{2}\b))
    ((-?)
     ((0[1-9]|1[0-2])
      (\3([12]\d|0[1-9]|3[01]))?
      |W([0-4]\d|5[0-3])(-?[1-7])?
      |(00[1-9]|0[1-9]\d|[12]\d{2}|3([0-5]\d|6[0-6]))
     )
     ([T\s]
      ((([01]\d|2[0-3])((:?)[0-5]\d)?|24\:?00)([\.,]\d+(?!:))?)?
      (\17[0-5]\d([\.,]\d+)?)?
      ([zZ]|([\+-])([01]\d|2[0-3]):?([0-5]\d)?)?
     )?
    )?
    $
    """, re.VERBOSE)
# Regular expression used in the init function of the UTCDateTime objects which
# is called a lot. Thus pre-compile it.
_YEAR0REGEX = re.compile(r"^(\d{1,3}[-/,])(.*)$")

TIMESTAMP0 = datetime.datetime(1970, 1, 1, 0, 0)

# common attributes
YMDHMS = ('year', 'month', 'day', 'hour', 'minute', 'second')
YJHMS = ('year', 'julday', 'hour', 'minute', 'second')
YMDHMS_FORMAT = "%04d-%02d-%02dT%02d:%02d:%02d"


class UTCDateTime(object):
    """
    A UTC-based datetime object.

    This datetime class is based on the POSIX time, a system for describing
    instants in time, defined as the number of seconds elapsed since midnight
    Coordinated Universal Time (UTC) of Thursday, January 1, 1970. Internally,
    the POSIX time is represented in nanoseconds as an integer, which allows
    higher precision than the default Python :class:`datetime.datetime` class.
    It features the full `ISO8601:2004`_ specification and some additional
    string patterns during object initialization.

    :type args: int, float, str, :class:`datetime.datetime`, optional
    :param args: The creation of a new `UTCDateTime` object depends from the
        given input parameters. All possible options are summarized in the
        `Examples`_ section below.
    :type iso8601: bool or None, optional
    :param iso8601: Enforce/disable `ISO8601:2004`_ mode. Defaults to ``None``
        for auto detection. Works only with a string as first input argument.
    :type strict: bool, optional
    :param strict: If True, Conform to `ISO8601:2004`_ limits on positional
        and keyword arguments. If False, allow hour, minute, second, and
        microsecond values to exceed 23, 59, 59, and 1_000_000 respectively.
    :type precision: int, optional
    :param precision: Sets the precision used by the rich comparison operators.
        Defaults to ``6`` digits after the decimal point. See also `Precision`_
        section below.

    .. versionchanged:: 1.1.0
        UTCDateTime is no longer based on a single floating point value but
        rather an integer representing nanoseconds elapsed since midnight
        Coordinated Universal Time (UTC) of Thursday, January 1, 1970.
        An integer internal representation allows higher precision and more
        predictable behavior than a float representation.

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

    (2) Using a `ISO8601:2004`_ string. The detection may be enabled/disabled
        using the``iso8601`` parameter, the default is to attempt to
        auto-detect ISO8601 compliant strings.

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

            >>> UTCDateTime("2009360T")                    # compact no time
            UTCDateTime(2009, 12, 26, 0, 0)

        * Week date representation.

            >>> UTCDateTime("2009-W53-7T12:23:34.5")
            UTCDateTime(2010, 1, 3, 12, 23, 34, 500000)

            >>> UTCDateTime("2009W537T122334.5")           # compact
            UTCDateTime(2010, 1, 3, 12, 23, 34, 500000)

            >>> UTCDateTime("2009W011", iso8601=True)      # enforce ISO8601
            UTCDateTime(2008, 12, 29, 0, 0)

        * Specifying time zones.

            >>> UTCDateTime('2019-09-18T+06')  # time zone is UTC+6
            UTCDateTime(2019, 9, 17, 18, 0)

            >>> UTCDateTime('2019-09-18T02-02')  # time zone is UTC-2
            UTCDateTime(2019, 9, 18, 4, 0)

            >>> UTCDateTime('2019-09-18T18:23:10.22-01')  # time zone is UTC-1
            UTCDateTime(2019, 9, 18, 19, 23, 10, 220000)

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

        >>> UTCDateTime("1970/01/17 12:23:34")
        UTCDateTime(1970, 1, 17, 12, 23, 34)

    (4) Using multiple arguments in the following order: `year, month,
        day[, hour[, minute[, second[, microsecond]]]`. The year, month and day
        arguments are required.

        >>> UTCDateTime(1970, 1, 1)
        UTCDateTime(1970, 1, 1, 0, 0)

        >>> UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)
        UTCDateTime(1970, 1, 1, 12, 23, 34, 123456)

    (5) Using the following keyword arguments: `year, month, day, julday, hour,
        minute, second, microsecond`. Either the combination of year, month and
        day, or year and Julian day are required. This is the only input mode
        that supports using hour, minute, or second values above the natural
        limits of 24, 60, 60, respectively.

        >>> UTCDateTime(year=1970, month=1, day=1, minute=15, microsecond=20)
        UTCDateTime(1970, 1, 1, 0, 15, 0, 20)

        >>> UTCDateTime(year=2009, julday=234, hour=14, minute=13)
        UTCDateTime(2009, 8, 22, 14, 13)

    (6) Using a Python :class:`datetime.datetime` object.

        >>> dt = datetime.datetime(2009, 5, 24, 8, 28, 12, 5001)
        >>> UTCDateTime(dt)
        UTCDateTime(2009, 5, 24, 8, 28, 12, 5001)

    (7) Using strict=False the limits of hour, minute, and second become more
        flexible:

        >>> UTCDateTime(year=1970, month=1, day=1, hour=48, strict=False)
        UTCDateTime(1970, 1, 3, 0, 0)

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

    (1) using the ``precision`` keyword during object initialization
        (preferred):

        >>> dt = UTCDateTime(0, precision=4)
        >>> dt2 = UTCDateTime(0.00001, precision=4)
        >>> print(dt.precision)
        4
        >>> dt == dt2
        True

    (2) or by setting the class attribute ``DEFAULT_PRECISION`` to the desired
        precision to affect all new :class:`UTCDateTime` objects
        (not recommended):

        >>> UTCDateTime.DEFAULT_PRECISION = 4
        >>> dt = UTCDateTime(0)
        >>> dt2 = UTCDateTime(0.00001)
        >>> print(dt.precision)
        4
        >>> dt == dt2
        True

        Don't forget to reset ``DEFAULT_PRECISION`` if not needed anymore!

        >>> UTCDateTime.DEFAULT_PRECISION = 6

    .. _ISO8601:2004: https://en.wikipedia.org/wiki/ISO_8601
    """
    DEFAULT_PRECISION = 6
    _initialized = False
    _has_warned = False  # this is a temporary, it will be removed soon

    def __init__(self, *args, **kwargs):
        """
        Creates a new UTCDateTime object.
        """
        # set default precision
        self.precision = kwargs.pop('precision', self.DEFAULT_PRECISION)
        # set directly to nanoseconds if given
        ns = kwargs.pop('ns', None)
        strict = kwargs.pop('strict', True)
        if ns is not None:
            self._ns = ns
            return
        # iso8601 flag
        iso8601 = kwargs.pop('iso8601', None)
        # check parameter
        if len(args) == 0 and len(kwargs) == 0:
            # use current date/time if no argument is given
            self._from_timestamp(time.time())
            return
        elif len(args) == 1 and len(kwargs) == 0:
            value = args[0]
            if isinstance(value, UTCDateTime):
                # ugly workaround to be able to unpickle UTCDateTime objects
                # that were pickled on ObsPy <1.1
                try:
                    self._ns = value._ns
                except AttributeError:
                    # work around floating point accuracy/rounding issue on
                    # Py3.3, see
                    # https://travis-ci.org/obspy/obspy/jobs/208941376#L751
                    # timestamp is 1251073203.0399999618 so when converting to
                    # integer nanosecond based UTCDateTime this should be
                    # rounded to 1251073203040000 nanoseconds.. but on Py3.3 it
                    # ends up as 1251073203039999, so we manually set
                    # microseconds with correct rounding without artifacts from
                    # floating point precision. see #1664
                    timestamp_seconds = int(value.__dict__['timestamp'])
                    timestamp_microseconds = round(
                        (value.__dict__['timestamp'] % 1.0) * 1e6)
                    dt_ = datetime.datetime.utcfromtimestamp(timestamp_seconds)
                    dt_ = dt_.replace(microsecond=timestamp_microseconds)
                    self._from_datetime(dt_)
                return
            # check types
            # The string instance check is mainly needed to not convert
            # numpy strings as these can be converted to floats on
            # numpy >= 1.14.
            if not isinstance(value, (str, bytes)):
                try:
                    # got a timestamp
                    self._from_timestamp(value.__float__())
                    return
                except Exception:
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
                if not isinstance(value, str):
                    value = value.decode()
                # got a string instance
                value = value.strip()

                # Raising in the case where the leading string is less than 4
                # chars; linked to #2167
                if re.match(_YEAR0REGEX, value):
                    raise ValueError(
                        "'%s' does not start with a 4 digit year" % value)

                # check for ISO8601 date string
                if iso8601 is True or (iso8601 is None and
                                       re.match(_ISO8601_REGEX, value)):
                    try:
                        self._from_iso8601_string(value)
                        return
                    except Exception:
                        # raise here if iso8601 is enforced otherwise fallback
                        # to non iso8601 detection by continuing below
                        if iso8601:
                            raise

                # try to apply some standard patterns
                value = value.replace('T', ' ')
                value = value.replace('_', ' ')
                value = value.replace('-', ' ')
                value = value.replace(':', ' ')
                value = value.replace(',', ' ')
                value = value.replace('/', ' ')
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
                    value = ''.join(parts)
                    # fill missing elements with zeros
                    value += '0' * (14 - len(value))
                    pattern = "%Y%m%d%H%M%S"
                ms = 0
                if '.' in value:
                    parts = value.split('.')
                    value = parts[0].strip()
                    try:
                        ms = float('.' + parts[1].strip())
                    except Exception:
                        pass
                # all parts should be digits now - here we filter unknown
                # patterns and pass it directly to Python's  datetime.datetime
                if not ''.join(parts).isdigit():
                    dt = datetime.datetime(*args, **kwargs)
                    self._from_datetime(dt)
                    return
                dt = datetime.datetime.strptime(value, pattern)
                dt += datetime.timedelta(seconds=ms)
                self._from_datetime(dt)
                return
        # check for ordinal/julian date kwargs
        if 'julday' in kwargs:
            try:
                int(kwargs['julday'])
            except (ValueError, TypeError):
                msg = "Failed to convert 'julday' to int: {!s}".format(
                    kwargs['julday'])
                raise TypeError(msg)
            if 'year' in kwargs:
                # year given as kwargs
                year = kwargs['year']
            elif len(args) == 1:
                # year is first (and only) argument
                year = args[0]
            days_in_year = calendar.isleap(year) and 366 or 365
            if not (1 <= int(kwargs['julday']) <= days_in_year):
                msg = "'julday' out of bounds for year {!s}: {!s}".format(
                    year, kwargs['julday'])
                raise ValueError(msg)
            try:
                temp = "%4d%03d" % (int(year),
                                    int(kwargs['julday']))
                dt = datetime.datetime.strptime(temp, '%Y%j')
            except Exception:
                pass
            else:
                kwargs['month'] = dt.month
                kwargs['day'] = dt.day
                kwargs.pop('julday')

        # check if seconds are given as float value
        if len(args) == 6 and isinstance(args[5], float):
            _frac, _sec = math.modf(round(args[5], 6))
            kwargs['second'] = int(_sec)
            kwargs['microsecond'] = int(round(_frac * 1e6))
            args = args[0:5]

        try:  # If a value Error is raised try to allow overflow (see #2222)
            dt = datetime.datetime(*args, **kwargs)
        except ValueError:
            if not strict:
                self._handle_overflow(*args, **kwargs)
            else:
                raise
        else:
            self._from_datetime(dt)

    def _handle_overflow(self, year, month, day, hour=0, minute=0, second=0,
                         microsecond=0):
        """
        Handles setting date if an overflow of usual value limits is detected.
        """
        # Keep track of seconds due to hour, minute, second
        seconds = 0
        seconds += hour * 3600
        seconds += minute * 60
        seconds += second
        # Init UTCDateTime based on year, month, day, add seconds
        utc_base = UTCDateTime(year=year, month=month, day=day)
        # Add seconds and set nanoseconds on self
        self._ns = (utc_base + seconds + microsecond / 1000000).ns

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
            self._ns = UTCDateTime(year=year, julday=julday, hour=hour,
                                   minute=minute, second=second,
                                   microsecond=microsecond)._ns
        else:
            self._ns = UTCDateTime(year, month, day, hour, minute,
                                   second, microsecond)._ns

    def _get_ns(self):
        """
        Returns POSIX timestamp as integer nanoseconds.

        This is the internal representation of UTCDateTime objects.

        :rtype: int
        :returns: POSIX timestamp as integer nanoseconds
        """
        return self.__ns

    def _set_ns(self, value):
        """
        Set UTCDateTime object from POSIX timestamp as integer nanoseconds.

        :type value: int
        :param value: POSIX timestamp as integer nanoseconds
        """
        # allow setting numpy integer types..
        if isinstance(value, np.integer):
            value_ = int(value)
            # ..and be paranoid and check that it's still the same value after
            # type casting
            if value_ != value:
                msg = ('Numpy integer value ({!s}) changed during casting to '
                       'Python builtin integer ({!s}).').format(value, value_)
                raise ValueError(msg)
            value = value_
        if not isinstance(value, int):
            raise TypeError('nanoseconds must be set as int/long type')
        self.__ns = value
        # flag that this instance has been initialized; any changes will warn
        self._initialized = True

    _ns = property(_get_ns, _set_ns)
    ns = property(_get_ns, _set_ns)

    def _from_datetime(self, dt):
        """
        Use Python datetime object to set current time.

        :type dt: :class:`datetime.datetime`
        :param dt: Python datetime object.
        """
        self._ns = _datetime_to_ns(dt)

    def _from_timestamp(self, value):
        """
        Use given timestamp to set current time.

        :type value: int, float
        :param value: Timestamp in seconds.
        """
        self._ns = int(round(value * 10**9))

    def _from_iso8601_string(self, value):
        """
        Parses an ISO8601:2004 date time string.
        """
        # remove trailing 'Z'
        value = value.replace('Z', '')
        # split between date and time
        try:
            (date, time) = value.split("T")
        except Exception:
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
        dt += datetime.timedelta(seconds=float(delta) + ms)
        self._from_datetime(dt)

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
        return self._ns / 1e9

    timestamp = property(_get_timestamp)

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
        rounded_ns = round(self._ns, self.precision - 9)
        dt = datetime.timedelta(seconds=rounded_ns // 10**9,
                                microseconds=rounded_ns % 10**9 // 1000)
        try:
            return TIMESTAMP0 + dt
        except OverflowError:
            # for very large future / past dates
            return datetime.datetime.utcfromtimestamp(self.timestamp)

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
        return self.datetime.date()

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
        return self.datetime.year

    def _set_year(self, value):
        """
        Sets year of current UTCDateTime object.

        :param value: Year
        :type value: int
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
        return self.datetime.month

    def _set_month(self, value):
        """
        Sets month of current UTCDateTime object.

        :param value: Month
        :type value: int
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
        return self.datetime.day

    def _set_day(self, value):
        """
        Sets day of current UTCDateTime object.

        :param value: Day
        :type value: int
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
        return self.datetime.weekday()

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
        return self.datetime.time()

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
        return self.datetime.hour

    def _set_hour(self, value):
        """
        Sets hours of current UTCDateTime object.

        :param value: Hours
        :type value: int
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
        return self.datetime.minute

    def _set_minute(self, value):
        """
        Sets minutes of current UTCDateTime object.

        :param value: Minutes
        :type value: int
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
        return self.datetime.second

    def _set_second(self, value):
        """
        Sets seconds of current UTCDateTime object.

        :param value: Seconds
        :type value: int
        """
        self._set(second=value)

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
        ms = int(round(self._ns % 10**9, self.precision - 9) // 1000)
        return ms % 1000000

    def _set_microsecond(self, value):
        """
        Sets microseconds of current UTCDateTime object.

        :param value: Microseconds
        :type value: int
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
        """
        self._set(julday=value)

    julday = property(_get_julday, _set_julday)

    def timetuple(self):
        """
        Return a time.struct_time such as returned by time.localtime().

        :rtype: time.struct_time
        """
        return self.datetime.timetuple()

    def utctimetuple(self):
        """
        Return a time.struct_time of current UTCDateTime object.

        :rtype: time.struct_time
        """
        return self.datetime.utctimetuple()

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
                     86400) * 10**6) / 1e6
        elif isinstance(value, UTCDateTime):
            msg = ("unsupported operand type(s) for +: 'UTCDateTime' and "
                   "'UTCDateTime'")
            raise TypeError(msg)
        return UTCDateTime(ns=self._ns + int(round(value * 1e9)))

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
            return round((self._ns - value._ns) / 1e9, self.__precision)
        elif isinstance(value, datetime.timedelta):
            # see datetime.timedelta.total_seconds
            value = (value.microseconds + (value.seconds + value.days *
                     86400) * 10**6) / 1e6
        return UTCDateTime(ns=self._ns - int(round((value * 1e9))))

    def __str__(self):
        """
        Returns ISO8601 string representation from current UTCDateTime object.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> str(dt)
        '2008-10-01T12:30:35.045020Z'
        """
        dt = self.datetime
        time_str = YMDHMS_FORMAT % tuple(getattr(dt, x) for x in YMDHMS)

        if self.precision > 0:
            ns = round(self.ns, self.precision - 9)
            ns_str = ('%09d' % (ns % 10 ** 9))[:self.precision]
            time_str += ('.' + ns_str)
        return time_str + 'Z'

    def _repr_pretty_(self, p, cycle):  # @UnusedVariable
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

    def _operate(self, other, op_func):
        if isinstance(other, UTCDateTime):
            ndigits = min(self.precision, other.precision) - 9
            if self.precision != other.precision:
                msg = ('Comparing UTCDateTime objects of different precision'
                       ' is not defined will raise an Exception in a future'
                       ' version of obspy')
                warnings.warn(msg, ObsPyDeprecationWarning)
            a = round(self._ns, ndigits)
            b = round(other._ns, ndigits)
            return op_func(a, b)
        else:
            try:
                return self._operate(UTCDateTime(other), op_func)
            except TypeError:
                return False

    def __eq__(self, other):
        """
        Rich comparison operator '=='.

        .. rubric: Example

        Comparing two UTCDateTime objects will compare the nanoseconds integers
        rounded to a number of significant digits determined by the precision
        attribute.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 == t2
        True

        Defining a higher precision changes the behavior of the operator

        >>> t1 = UTCDateTime(123.000000012, precision=9)
        >>> t2 = UTCDateTime(123.000000099, precision=9)
        >>> t1 == t2
        False
        """
        return self._operate(other, operator.eq)

    def __ne__(self, other):
        """
        Rich comparison operator '!='.

        .. rubric: Example

        Comparing two UTCDateTime objects will compare the nanoseconds integers
        rounded to a number of significant digits determined by the precision
        attribute.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 == t2
        True

        Defining a higher precision changes the behavior of the operator

        >>> t1 = UTCDateTime(123.000000012, precision=9)
        >>> t2 = UTCDateTime(123.000000099, precision=9)
        >>> t1 == t2
        False
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Rich comparison operator '<'.

        .. rubric: Example

        Comparing two UTCDateTime objects will compare the nanoseconds integers
        rounded to a number of significant digits determined by the precision
        attribute.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 < t2
        False

        Defining a higher precision changes the behavior of the operator

        >>> t1 = UTCDateTime(123.000000012, precision=9)
        >>> t2 = UTCDateTime(123.000000099, precision=9)
        >>> t1 < t2
        True
        """
        return self._operate(other, operator.lt)

    def __le__(self, other):
        """
        Rich comparison operator '<='.

        .. rubric: Example

        Comparing two UTCDateTime objects will compare the nanoseconds integers
        rounded to a number of significant digits determined by the precision
        attribute.

        >>> t1 = UTCDateTime(123.000000099)
        >>> t2 = UTCDateTime(123.000000012)
        >>> t1 <= t2
        True

        Defining a higher precision changes the behavior of the operator

        >>> t1 = UTCDateTime(123.000000099, precision=9)
        >>> t2 = UTCDateTime(123.000000012, precision=9)
        >>> t1 <= t2
        False
        """
        return self._operate(other, operator.le)

    def __gt__(self, other):
        """
        Rich comparison operator '>'.

        .. rubric: Example

        Comparing two UTCDateTime objects will compare the nanoseconds integers
        rounded to a number of significant digits determined by the precision
        attribute.

        >>> t1 = UTCDateTime(123.000000099)
        >>> t2 = UTCDateTime(123.000000012)
        >>> t1 > t2
        False

        Defining a higher precision changes the behavior of the operator

        >>> t1 = UTCDateTime(123.000000099, precision=9)
        >>> t2 = UTCDateTime(123.000000012, precision=9)
        >>> t1 > t2
        True
        """
        return self._operate(other, operator.gt)

    def __ge__(self, other):
        """
        Rich comparison operator '>='.

        .. rubric: Example

        Comparing two UTCDateTime objects will compare the nanoseconds integers
        rounded to a number of significant digits determined by the precision
        attribute.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 >= t2
        True

        Defining a higher precision changes the behavior of the operator

        >>> t1 = UTCDateTime(123.000000012, precision=9)
        >>> t2 = UTCDateTime(123.000000099, precision=9)
        >>> t1 >= t2
        False
        """
        return self._operate(other, operator.ge)

    def __repr__(self):
        """
        Returns a representation of UTCDatetime object.
        """
        return 'UTCDateTime' + self.datetime.__repr__()[17:]

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

    def __setattr__(self, key, value):
        # raise a warning if overwriting previous ns (see #2072)
        if self._initialized and not self._has_warned:
            msg = ('Setting attributes on UTCDateTime instances will raise an'
                   ' Exception in a future version of Obspy.')
            warnings.warn(msg, ObsPyDeprecationWarning)
            # only issue the warning once per object
            self.__dict__['_has_warned'] = True
        super(UTCDateTime, self).__setattr__(key, value)

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
        # This is an attempt to get consistent behavior across platforms.
        # See https://bugs.python.org/issue32195
        # and https://bugs.python.org/issue13305
        # This is an issue of glibc implementation differing across platforms,
        # out of control of Python, but we still try to be consistent across
        # all platforms
        if sys.platform.startswith("linux"):
            format = format.replace("%Y", "%04Y")
        return self.datetime.strftime(format)

    @staticmethod
    def strptime(date_string, format):
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
        return self.datetime.timetz()

    def utcoffset(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.utcoffset()
        """
        return self.datetime.utcoffset()

    def dst(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.dst()
        """
        return self.datetime.dst()

    def tzname(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.tzname()
        """
        return self.datetime.tzname()

    def ctime(self):
        """
        Return a string representing the date and time.

        .. rubric:: Example

        >>> UTCDateTime(2002, 12, 4, 20, 30, 40).ctime()
        'Wed Dec  4 20:30:40 2002'
        """
        return self.datetime.ctime()

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
        return self.datetime.isoweekday()

    def isocalendar(self):
        """
        Returns a tuple containing (ISO year, ISO week number, ISO weekday).

        :rtype: tuple(int)
        :return: Returns a (named) tuple containing ISO year, ISO week number
            and ISO weekday. Depending on the used Python version it either
            returns a tuple (Py<3.9) or named tuple (Py>=3.9).

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> tuple(dt.isocalendar())
        (2008, 40, 3)
        """
        return self.datetime.isocalendar()

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
        return self.datetime.isoformat(sep=sep)

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
        :rtype: str
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
        temp += ",%02d" % (self.hour)
        if self.microsecond:
            return temp + ":%02d:%02d.%04d" % (self.minute, self.second,
                                               self.microsecond // 100)
        elif self.second:
            return temp + ":%02d:%02d" % (self.minute, self.second)
        elif self.minute:
            return temp + ":%02d" % (self.minute)
        return temp

    def format_iris_web_service(self):
        """
        Returns string representation usable for the IRIS Web services.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 5, 27, 12, 30, 35, 45020)
        >>> print(dt.format_iris_web_service())
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
        """
        if value > 9:
            msg = 'UTCDateTime precision above 9 is not supported, using 9'
            warnings.warn(msg)
            value = 9
        self.__precision = int(value)

    precision = property(_get_precision, _set_precision)

    def replace(self, **kwargs):
        """
        Return a new UTCDateTime object with one or more parameters replaced.

        Replace is useful for substituting parameters that depend on other
        parameters (eg hour depends on the current day for meaning). In order
        to replace independent parameters, such as timestamp, ns, or
        precision, simply create a new UTCDateTime instance.

        The following parameters are supported: year, month, day, julday,
        hour, minute second, microsecond. Additionally, the keyword 'strict'
        can be set to False to allow hour, minute, and second to exceed normal
        limits.

        .. rubric:: Example

        (1) Get time of the 15th day of the same month to which a timestamp
            belongs.

            >>> dt = UTCDateTime(999999999)
            >>> dt2 = dt.replace(day=15)
            >>> print(dt2)
            2001-09-15T01:46:39.000000Z


        (2) Determine the day of the week 2 months before Guy Fawkes day.

            >>> dt = UTCDateTime('1605-11-05')
            >>> dt.replace(month=9).weekday
            0
        """
        # check parameters, raise Value error if any are unsupported
        supported_args = set(YMDHMS) | set(YJHMS) | {'microsecond', 'strict'}
        if not set(kwargs).issubset(supported_args):
            unsupported_args = set(kwargs) - supported_args
            msg = ('%s are not supported arguments for replace, supported '
                   'arguments are %s') % (unsupported_args, supported_args)
            raise ValueError(msg)
        # ensure julday is used correctly if used
        if kwargs.get('julday') is not None:
            if 'month' in kwargs or 'day' in kwargs:
                msg = 'If julday is used month and day cannot be used.'
                raise ValueError(msg)
            time_paramters = YJHMS  # use julday

        else:
            time_paramters = YMDHMS  # use month and day
        # get a dict of time parameters to pass to UTCDateTime constructor
        new_dict = {x: getattr(self, x) for x in time_paramters}
        new_dict['microsecond'] = self.microsecond
        new_dict.update(kwargs)
        return UTCDateTime(**new_dict)

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
        return self.datetime.toordinal()

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

    @property
    def matplotlib_date(self):
        """
        Maplotlib date number representation.

        Useful for plotting on matplotlib time-based axes, like created by e.g.
        :meth:`obspy.core.stream.Stream.plot()`.

        >>> t = UTCDateTime("2009-08-24T00:20:07.700000Z")
        >>> t.matplotlib_date  # doctest: +SKIP
        14480.01397800926

        :rtype: float
        """
        from matplotlib.dates import date2num
        return date2num(self.datetime)


def _datetime_to_ns(dt):
    """
    Use Python datetime object to return equivalent nanoseconds.

    :type dt: :class:`datetime.datetime`
    :param dt: Python datetime object.
    :returns: nanoseconds as an int.
    """
    try:
        td = (dt - TIMESTAMP0)
    except TypeError:
        td = (dt.replace(tzinfo=None) - dt.utcoffset()) - TIMESTAMP0
    # see datetime.timedelta.total_seconds
    return (td.days * 86400 + td.seconds) * 10**9 + td.microseconds * 1000


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
