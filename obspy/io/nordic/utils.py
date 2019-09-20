# -*- coding: utf-8 -*-
"""
Utility functions for Nordic file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import warnings
from collections import defaultdict

from obspy.io.nordic import NordicParsingError
from obspy.geodetics.base import kilometers2degrees
from numpy import cos, radians


MAG_MAPPING = {"ML": "L", "MLv": "L", "mB": "B", "Ms": "s", "MS": "S",
               "MW": "W", "MbLg": "G", "Mc": "C"}
INV_MAG_MAPPING = {item: key for key, item in MAG_MAPPING.items()}
# List of currently implemented line-endings, which in Nordic mark what format
# info in that line will be.
ACCEPTED_TAGS = ('1', '6', '7', 'E', ' ', 'F', 'M', '3', 'H')


def _int_conv(string):
    """
    Convenience tool to convert from string to integer.

    If empty string return None rather than an error.

    >>> _int_conv('12')
    12
    >>> _int_conv('')

    """
    try:
        intstring = int(string)
    except Exception:
        intstring = None
    return intstring


def _float_conv(string):
    """
    Convenience tool to convert from string to float.

    If empty string return None rather than an error.

    >>> _float_conv('12')
    12.0
    >>> _float_conv('')
    >>> _float_conv('12.324')
    12.324
    """
    try:
        floatstring = float(string)
    except Exception:
        floatstring = None
    return floatstring


def _str_conv(number, rounded=False):
    """
    Convenience tool to convert a number, either float or int into a string.

    If the int or float is None, returns empty string.

    >>> print(_str_conv(12.3))
    12.3
    >>> print(_str_conv(12.34546, rounded=1))
    12.3
    >>> print(_str_conv(None))
    <BLANKLINE>
    >>> print(_str_conv(1123040))
    11.2e5
    """
    if not number:
        return str(' ')
    if not rounded and isinstance(number, (float, int)):
        if number < 100000:
            string = str(number)
        else:
            exponent = int('{0:.2E}'.format(number).split('E+')[-1]) - 1
            divisor = 10 ** exponent
            string = '{0:.1f}'.format(number / divisor) + 'e' + str(exponent)
    elif rounded and isinstance(number, (float, int)):
        if number < 100000:
            string = "{:.{precision}f}".format(number, precision=rounded)
        else:
            exponent = int('{0:.2E}'.format(number).split('E+')[-1]) - 1
            divisor = 10 ** exponent
            string = "{:.{precision}f}".format(
                number / divisor, precision=rounded) + 'e' + str(exponent)
    else:
        return str(number)
    return string


def _get_line_tags(f, report=True):
    """
    Associate lines with a known line-type
    :param f: File open in read
    :param report: Whether to report warnings about lines not implemented
    """
    f.seek(0)
    line = f.readline()
    if len(line.rstrip()) != 80:
        # Cannot be Nordic
        raise NordicParsingError(
            "Lines are not 80 characters long: not a nordic file")
    f.seek(0)
    tags = defaultdict(list)
    for i, line in enumerate(f):
        try:
            line_id = line.rstrip()[79]
        except IndexError:
            line_id = ' '
        if line_id in ACCEPTED_TAGS:
            tags[line_id].append((line, i))
        elif report:
            warnings.warn("Lines of type %s have not been implemented yet, "
                          "please submit a development request" % line_id)
    return tags


def _evmagtonor(mag_type):
    """
    Switch from obspy event magnitude types to seisan syntax.

    >>> print(_evmagtonor('mB'))  # doctest: +SKIP
    B
    >>> print(_evmagtonor('M'))  # doctest: +SKIP
    W
    >>> print(_evmagtonor('bob'))  # doctest: +SKIP
    <BLANKLINE>
    """
    if mag_type == 'M':
        warnings.warn('Converting generic magnitude to moment magnitude')
        return "W"
    mag = MAG_MAPPING.get(mag_type, '')
    if mag == '':
        warnings.warn(mag_type + ' is not convertible')
    return mag


def _nortoevmag(mag_type):
    """
    Switch from nordic type magnitude notation to obspy event magnitudes.

    >>> print(_nortoevmag('B'))  # doctest: +SKIP
    mB
    >>> print(_nortoevmag('bob'))  # doctest: +SKIP
    <BLANKLINE>
    """
    if mag_type.upper() == "L":
        return "ML"
    mag = INV_MAG_MAPPING.get(mag_type, '')
    if mag == '':
        warnings.warn(mag_type + ' is not convertible')
    return mag


def _km_to_deg_lat(kilometers):
    """
    Convenience tool for converting km to degrees latitude.

    """
    try:
        degrees = kilometers2degrees(kilometers)
    except Exception:
        degrees = None
    return degrees


def _km_to_deg_lon(kilometers, latitude):
    """
    Convenience tool for converting km to degrees longitude.

    latitude in degrees

    """
    try:
        degrees_lat = kilometers2degrees(kilometers)
    except Exception:
        return None
    degrees_lon = degrees_lat / cos(radians(latitude))

    return degrees_lon


if __name__ == "__main__":
    import doctest
    doctest.testmod()
