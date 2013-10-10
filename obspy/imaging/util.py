# -*- coding: utf-8 -*-
"""
Waveform plotting utilities.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU General Public License (GPL)
    (http://www.gnu.org/licenses/gpl.txt)
"""
from matplotlib.dates import num2date, AutoDateFormatter, DateFormatter


def decimal_seconds_format(x, pos=None):
    """
    This format function is used to format date ticklabels with decimal
    seconds but stripping trailing zeros.
    """
    x = num2date(x)
    ret = x.strftime('%H:%M:%S.%f')
    ret = ret.rstrip("0")
    ret = ret.rstrip(".")
    return ret


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
    def __call__(self, x, pos=None):
        scale = float(self._locator._get_unit())
        fmt = self.defaultfmt

        for k in sorted(self.scaled):
            if k >= scale:
                fmt = self.scaled[k]
                break

        if isinstance(fmt, basestring):
            self._formatter = DateFormatter(fmt, self._tz)
            return self._formatter(x, pos)
        elif hasattr(fmt, '__call__'):
            return fmt(x, pos)
        else:
            raise NotImplementedError()
