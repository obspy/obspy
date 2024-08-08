# -*- coding: utf-8 -*-
"""
Utility functions for Nordic file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import warnings
from collections import defaultdict

from obspy.io.nordic import NordicParsingError
from obspy.geodetics.base import kilometers2degrees
from numpy import cos, radians

# For a non-exhaustive list of magnitudes, see p. 11 in:
# https://doi.org/10.2312/GFZ.NMSOP-2_IS_3.2
MAG_MAPPING = {"ML": "L", "MLv": "L", "Ml": "l",
               "mB": "B", "mb": "b", "MbLg": "G",
               "Ms": "s", "MS": "S",
               "MW": "W", "Mw": "w", "Mc": "C",
               "MN": "N", "Mn": "n"}
INV_MAG_MAPPING = {item: key for key, item in MAG_MAPPING.items()}

# Event-type mapping:
EVENT_TYPE_MAPPING_FROM_SEISAN = {
    "E": "explosion",
    "P": "explosion",
    # not a great translation; V=volcanic event not allowed in obspy/QuakeML
    "V": "earthquake",
    "Q": "earthquake",
    " ": "earthquake",
    "L": "landslide",
    "X": "landslide",  # Depreceated Nordic code
    "S": "sonic boom",  # not a great translation for "acoustic signal"
    "I": "induced or triggered event",
    "O": "other event",
    "C": "ice quake",
    "G": "ice quake"}

EVENT_TYPE_CERTAINTY_MAPPING_FROM_SEISAN = {
    "E": 'known',
    "P": 'suspected',
    "V": 'known',
    "Q": 'known',
    " ": 'suspected',
    "L": 'known',
    "X": 'known',
    "S": 'known',
    "I": 'known',
    "O": 'known',
    "C": 'known',
    "G": 'known'}

EVENT_TYPE_MAPPING_TO_SEISAN = {
    "explosion": "E",
    "earthquake": "Q",
    "landslide": "L",
    "sonic boom": "S",
    "induced or triggered event": "I",
    "other event": "O",
    "ice quake": "C",
    "not reported": " "}

# Nordic format condenses type and certainty into one letter in some cases
EVENT_TYPE_AND_CERTAINTY_MAPPING_TO_SEISAN = {
    "known explosion": "E",
    "suspected explosion": "P",
    "known earthquake": "Q",
    "suspected earthquake": " "}

# List of currently implemented line-endings, which in Nordic mark what format
# info in that line will be.
ACCEPTED_TAGS = ('1', '6', '7', 'E', ' ', 'F', 'M', '3', 'H')

ACCEPTED_1CHAR_PHASE_TAGS = ['P', 'p', 'S', 's', 'L', 'G', 'R', 'H', 'T', 'x',
                             'r', 't', 'E']
ACCEPTED_2CHAR_PHASE_TAGS = ['I ', 'Ip', 'Is', 'Ir']
ACCEPTED_3CHAR_PHASE_TAGS = ['BAZ', 'END']

ACCEPTED_1CHAR_AMPLITUDE_PHASE_TAGS = ['A', 'V']
ACCEPTED_2CHAR_AMPLITUDE_PHASE_TAGS = ['AM']
ACCEPTED_3CHAR_AMPLITUDE_PHASE_TAGS = ['IAM', 'IVM', 'AML']


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
    if not number and number != 0:
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
    if mag_type == 'M' or mag_type is None:
        warnings.warn('Converting generic magnitude to moment magnitude')
        return "W"
    mag = MAG_MAPPING.get(mag_type, '')
    if mag == '':
        warnings.warn(mag_type + ' is not convertible')
        return ' '
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


def _nordic_iasp_phase_ok(phase):
    """
    Function to check whether a phase-string is a valid IASPEI-compatible
    phase in Seisan.
    :param phase: Phase string to check

    :returns: bool, whether phase string is valid in Seisan.
    """
    try:
        if phase[0] in ACCEPTED_1CHAR_PHASE_TAGS:
            return True
        if phase[0:2] in ACCEPTED_2CHAR_PHASE_TAGS:
            return True
        if phase[0:3] in ACCEPTED_3CHAR_PHASE_TAGS:
            return True
    except IndexError:
        pass
    return False


def _is_iasp_ampl_phase(phase):
    """
    Function to check whether a phase-string describes an IASPEI.conpatible
    amplitude phase in Seisan.

    :type phase: str
    :param phase: Phase string to be check

    :returns: bool, whether phase string is an amplitude phase
    """
    if phase is None:
        return False
    phase = phase.strip()
    try:
        if phase[0] in ACCEPTED_1CHAR_AMPLITUDE_PHASE_TAGS:
            return True
        if phase[0:2] in ACCEPTED_2CHAR_AMPLITUDE_PHASE_TAGS:
            return True
        if phase[0:3] in ACCEPTED_3CHAR_AMPLITUDE_PHASE_TAGS:
            return True
    except IndexError:
        pass
    return False


def _get_agency_id(item):
    """
    Function to return a properly formatted 3-character agency ID from the
    creation_info-property of an item.
    """
    agency_id = '   '
    if hasattr(item, 'creation_info'):
        if hasattr(item.creation_info, 'agency_id'):
            agency_id = item.creation_info.get('agency_id')
    if agency_id is None:
        agency_id = '   '
    agency_id = agency_id.rjust(3)[0:3]
    return agency_id


if __name__ == "__main__":
    import doctest
    doctest.testmod()
