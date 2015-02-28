# -*- coding: utf-8 -*-
"""
Waveform plotting utilities.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU General Public License (GPL)
    (http://www.gnu.org/licenses/gpl.txt)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

from matplotlib.dates import AutoDateFormatter, DateFormatter, num2date
from matplotlib.ticker import FuncFormatter


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


class ObsPyAutoDateFormatter(AutoDateFormatter):
    """
    Derived class to allow for more customized formatting with older matplotlib
    versions (see matplotlib/matplotlib#2507).
    """
    def __init__(self, *args, **kwargs):
        super(ObsPyAutoDateFormatter, self).__init__(*args, **kwargs)
        self.scaled[1. / 24.] = '%H:%M'
        self.scaled[1. / (24. * 60.)] = '%H:%M:%S'
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
