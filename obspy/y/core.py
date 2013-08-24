# -*- coding: utf-8 -*-
"""
Y bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy import Stream
from struct import unpack


def isY(filename):
    """
    Checks whether a file is a Nanometrics Y file or not.

    :type filename: str
    :param filename: Name of the Nanometrics Y file to be checked.
    :rtype: bool
    :return: ``True`` if a Nanometrics Y file.

    .. rubric:: Example

    >>> isASC("/path/to/YAYT_BHZ_20021223.124800")  #doctest: +SKIP
    True
    """
    try:
        temp = open(filename, 'rb').read(4)
    except:
        return False
    try:
        # byte order format for this data. Uses letter “I” for Intel format
        # data (little endian) or letter “M” for Motorola (big endian) format
        if temp[0] == 'I':
            _endian = '<'
        elif temp[0] == 'M':
            _endian = '>'
        else:
            return False
        # check for magic number "31"
        magic = unpack('%sB' % _endian, temp[1])[0]
        if magic != 31:
            return False
        # The first tag in a Y-file must be the TAG_Y_FILE tag (tag type 0)
        tag_type = unpack('%sH' % _endian, temp[2:4])[0]
        if tag_type != 0:
            return False
    except:
        return False
    return True


def readY(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a Nanometrics Y file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: Nanometrics Y file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/YAYT_BHZ_20021223.124800")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    .TEST..BHN | 2009-10-01T12:46:01.000000Z - ... | 20.0 Hz, 801 samples
    """
    stream = Stream()
    return stream


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
