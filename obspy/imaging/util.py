# -*- coding: utf-8 -*-
"""
Waveform plotting utilities.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import re
from dateutil.rrule import MINUTELY, SECONDLY

from matplotlib.dates import (
    AutoDateLocator, AutoDateFormatter, DateFormatter, num2date)
from matplotlib.ticker import FuncFormatter

from obspy import UTCDateTime
from obspy.core.util import MATPLOTLIB_VERSION


def _seconds_to_days(sec):
    return sec / 3600.0 / 24.0


def decimal_seconds_format_x_decimals(decimals=None):
    """
    Function factory for format functions to format date ticklabels with
    given number of decimals to seconds (stripping trailing zeros).

    :type decimals: int
    :param decimals: Number of decimals for seconds after which to cut off
        format string. Decimals are left alone if not specified.
    """
    def func(x, pos=None):
        x = num2date(x)
        ret = x.strftime('%H:%M:%S.%f')
        ret = ret.rstrip("0")
        ret = ret.rstrip(".")
        _decimals = decimals
        # show more decimals in matplotlib info toolbar (for formatting of
        # mouse-over x coordinate, which is in-between picks and needs higher
        # accuracy)
        if pos is None and _decimals is not None:
            _decimals += 2
        if "." in ret and _decimals is not None:
            ret = ret[:ret.find(".") + _decimals + 1]
        return ret
    return func


def decimal_seconds_format_date_first_tick(x, pos=None):
    """
    This format function is used to format date ticklabels with decimal
    seconds but stripping trailing zeros.
    """
    x = num2date(x)
    if pos == 0:
        fmt = '%b %d %Y\n%H:%M:%S.%f'
    else:
        fmt = '%H:%M:%S.%f'
    ret = x.strftime(fmt)
    ret = ret.rstrip("0")
    ret = ret.rstrip(".")
    return ret


def format_hour_minute(x, pos=None):
    """
    Format tick like '%H:%M' but add date to first tick
    """
    t = num2date(x)
    if pos == 0:
        ret = t.strftime('%Y-%m-%dT%H:%M')
    else:
        ret = t.strftime('%H:%M')
    return ret


def format_hour_minute_second(x, pos=None):
    """
    Format tick like '%H:%M:%S' but add date to first tick
    """
    t = num2date(x)
    if pos == 0:
        ret = t.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        ret = t.strftime('%H:%M:%S')
    return ret


class ObsPyAutoDateFormatter(AutoDateFormatter):
    """
    Derived class to allow for more customized formatting with older matplotlib
    versions (older than 1.4.0, see matplotlib/matplotlib#2507).
    """
    def __init__(self, *args, **kwargs):
        # the root class of AutoDateFormatter (TickHelper) is an old style
        # class prior to matplotlib version 1.2
        if MATPLOTLIB_VERSION < [1, 2, 0]:
            AutoDateFormatter.__init__(self, *args, **kwargs)
        else:
            super(ObsPyAutoDateFormatter, self).__init__(*args, **kwargs)
        # Reset the scale to make it reproducible across matplotlib versions.
        self.scaled = {}
        self.scaled[1.0] = '%b %d %Y'
        self.scaled[30.0] = '%b %Y'
        self.scaled[365.0] = '%Y'
        self.scaled[1. / 24.] = FuncFormatter(format_hour_minute)
        self.scaled[1. / (24. * 60.)] = \
            FuncFormatter(format_hour_minute_second)
        self.scaled[_seconds_to_days(1)] = \
            FuncFormatter(format_hour_minute_second)
        self.scaled[_seconds_to_days(10)] = \
            FuncFormatter(decimal_seconds_format_x_decimals(1))
        # for some reason matplotlib is not using the following intermediate
        # decimal levels (probably some precision issue..) and falls back to
        # the lowest level immediately.
        self.scaled[_seconds_to_days(2e-1)] = \
            FuncFormatter(decimal_seconds_format_x_decimals(2))
        self.scaled[_seconds_to_days(2e-2)] = \
            FuncFormatter(decimal_seconds_format_x_decimals(3))
        self.scaled[_seconds_to_days(2e-3)] = \
            FuncFormatter(decimal_seconds_format_x_decimals(4))
        self.scaled[_seconds_to_days(2e-4)] = \
            FuncFormatter(decimal_seconds_format_x_decimals(5))

    def __call__(self, x, pos=None):
        # Always show full precision date string on info pane (pos=None)
        # because for some zoom levels the ticks might be ambiguous (e.g. hours
        # displayed, wrapped around days).
        if pos is None:
            return str(UTCDateTime(num2date(x)))

        scale = float(self._locator._get_unit())
        fmt = self.defaultfmt

        for k in sorted(self.scaled):
            if k >= scale:
                fmt = self.scaled[k]
                break

        if isinstance(fmt, (str, native_str)):
            self._formatter = DateFormatter(fmt, self._tz)
            return self._formatter(x, pos)
        elif hasattr(fmt, '__call__'):
            return fmt(x, pos)
        else:
            raise NotImplementedError()


def _id_key(id_):
    """
    Compare two trace IDs by network/station/location single character
    component codes according to sane ZNE/ZRT/LQT order. Any other characters
    are sorted afterwards alphabetically.

    >>> networks = ["A", "B", "AB"]
    >>> stations = ["X", "Y", "XY"]
    >>> locations = ["00", "01"]
    >>> channels = ["EHZ", "EHN", "EHE", "Z"]
    >>> trace_ids = []
    >>> for net in networks:
    ...     for sta in stations:
    ...         for loc in locations:
    ...             for cha in channels:
    ...                 trace_ids.append(".".join([net, sta, loc, cha]))
    >>> from random import shuffle
    >>> shuffle(trace_ids)
    >>> trace_ids = sorted(trace_ids, key=_id_key)
    >>> print(" ".join(trace_ids))  # doctest: +NORMALIZE_WHITESPACE
    A.X.00.Z A.X.00.EHZ A.X.00.EHN A.X.00.EHE A.X.01.Z A.X.01.EHZ A.X.01.EHN
    A.X.01.EHE A.XY.00.Z A.XY.00.EHZ A.XY.00.EHN A.XY.00.EHE A.XY.01.Z
    A.XY.01.EHZ A.XY.01.EHN A.XY.01.EHE A.Y.00.Z A.Y.00.EHZ A.Y.00.EHN
    A.Y.00.EHE A.Y.01.Z A.Y.01.EHZ A.Y.01.EHN A.Y.01.EHE AB.X.00.Z AB.X.00.EHZ
    AB.X.00.EHN AB.X.00.EHE AB.X.01.Z AB.X.01.EHZ AB.X.01.EHN AB.X.01.EHE
    AB.XY.00.Z AB.XY.00.EHZ AB.XY.00.EHN AB.XY.00.EHE AB.XY.01.Z AB.XY.01.EHZ
    AB.XY.01.EHN AB.XY.01.EHE AB.Y.00.Z AB.Y.00.EHZ AB.Y.00.EHN AB.Y.00.EHE
    AB.Y.01.Z AB.Y.01.EHZ AB.Y.01.EHN AB.Y.01.EHE B.X.00.Z B.X.00.EHZ
    B.X.00.EHN B.X.00.EHE B.X.01.Z B.X.01.EHZ B.X.01.EHN B.X.01.EHE B.XY.00.Z
    B.XY.00.EHZ B.XY.00.EHN B.XY.00.EHE B.XY.01.Z B.XY.01.EHZ B.XY.01.EHN
    B.XY.01.EHE B.Y.00.Z B.Y.00.EHZ B.Y.00.EHN B.Y.00.EHE B.Y.01.Z B.Y.01.EHZ
    B.Y.01.EHN B.Y.01.EHE
    """
    # remove processing info which was added previously
    id_ = re.sub(r'\[.*', '', id_)
    netstaloc, cha = id_.upper().rsplit(".", 1)
    key = netstaloc.split()
    # sort by network, station, location codes, then by..
    #  - length of channel code
    #  - last letter of channel code
    key.append(len(cha))
    if len(cha) != 0:
        key.append(_component_code_key(cha[-1]))
    return key


def _component_code_key(val):
    """
    Compare two single character component codes according to sane ZNE/ZRT/LQT
    order. Any other characters are sorted afterwards alphabetically.

    >>> from random import shuffle
    >>> from string import ascii_lowercase, ascii_uppercase
    >>> lowercase = list(ascii_lowercase)
    >>> uppercase = list(ascii_uppercase)
    >>> shuffle(lowercase)
    >>> shuffle(uppercase)
    >>> component_codes = lowercase + uppercase
    >>> component_codes = sorted(component_codes,
    ...                          key=_component_code_key)
    >>> print(component_codes)  # doctest: +NORMALIZE_WHITESPACE
    ['z', 'Z', 'n', 'N', 'e', 'E', 'r', 'R', 'l', 'L', 'q', 'Q', 't', 'T', 'a',
        'A', 'b', 'B', 'c', 'C', 'd', 'D', 'f', 'F', 'g', 'G', 'h', 'H', 'i',
        'I', 'j', 'J', 'k', 'K', 'm', 'M', 'o', 'O', 'p', 'P', 's', 'S', 'u',
        'U', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y']
    """
    order = "ZNERLQT"
    val = val.upper()
    try:
        # Return symbols that sort first and are invalid.
        return chr(order.index(val) + 32)
    except ValueError:
        return val


def _timestring(t):
    """
    Returns a full string representation of a
    :class:`~obspy.core.utcdatetime.UTCDateTime` object, stripping away
    trailing decimal-second zeros.

    >>> from obspy import UTCDateTime
    >>> print(_timestring(UTCDateTime("2012-04-05T12:12:12.123456Z")))
    2012-04-05T12:12:12.123456
    >>> print(_timestring(UTCDateTime("2012-04-05T12:12:12.120000Z")))
    2012-04-05T12:12:12.12
    >>> print(_timestring(UTCDateTime("2012-04-05T12:12:12.000000Z")))
    2012-04-05T12:12:12
    >>> print(_timestring(UTCDateTime("2012-04-05T12:12:00.000000Z")))
    2012-04-05T12:12:00
    >>> print(_timestring(UTCDateTime("2012-04-05T12:12:00.120000Z")))
    2012-04-05T12:12:00.12
    """
    return str(t).rstrip("Z0").rstrip(".")


def _set_xaxis_obspy_dates(ax, ticklabels_small=True):
    """
    Set Formatter/Locator of x-Axis to use ObsPyAutoDateFormatter and do some
    other tweaking.

    In contrast to normal matplotlib ``AutoDateFormatter`` e.g. shows full
    timestamp on first tick when zoomed in so far that matplotlib would only
    show hours or minutes on all ticks (making it impossible to tell the date
    from the axis labels) and also shows full timestamp in matplotlib figures
    info line (mouse-over info of current cursor position).

    :type ax: :class:`matplotlib.axes.Axes`
    :rtype: None
    """
    ax.xaxis_date()
    locator = AutoDateLocator(minticks=3, maxticks=6)
    locator.intervald[MINUTELY] = [1, 2, 5, 10, 15, 30]
    locator.intervald[SECONDLY] = [1, 2, 5, 10, 15, 30]
    ax.xaxis.set_major_formatter(ObsPyAutoDateFormatter(locator))
    ax.xaxis.set_major_locator(locator)
    if ticklabels_small:
        import matplotlib.pyplot as plt
        plt.setp(ax.get_xticklabels(), fontsize='small')


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
