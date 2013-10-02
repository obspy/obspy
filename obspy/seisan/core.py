# -*- coding: utf-8 -*-
"""
SEISAN bindings to ObsPy core module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
import numpy as np


def isSEISAN(filename):
    """
    Checks whether a file is SEISAN or not.

    :type filename: str
    :param filename: Name of the audio SEISAN file to be checked.
    :rtype: bool
    :return: ``True`` if a SEISAN file.

    .. rubric:: Example

    >>> isSEISAN("/path/to/1996-06-03-1917-52S.TEST__002")  #doctest: +SKIP
    True
    """
    try:
        f = open(filename, 'rb')
    except:
        return False
    # read some data - contains at least 12 lines a 80 characters
    data = f.read(12 * 80)
    f.close()
    if _getVersion(data):
        return True
    return False


def _getVersion(data):
    """
    Extracts SEISAN version from given data chunk.

    Parameters
    ----------
    data : string
        Data chunk.

    Returns
    -------
    tuple, ([ '<' | '>' ], [ 32 | 64 ], [ 6 | 7 ])
        Byte order (little endian '<' or big endian '>'), architecture (32 or
        64) and SEISAN version (6 or 7).

    From the SEISAN documentation::

        When Fortran writes a files opened with "form=unformatted", additional
        data is added to the file to serve as record separators which have to
        be taken into account if the file is read from a C-program or if read
        binary from a Fortran program. Unfortunately, the number of and meaning
        of these additional characters are compiler dependent. On Sun, Linux,
        MaxOSX and PC from version 7.0 (using Digital Fortran), every write is
        preceded and terminated with 4 additional bytes giving the number of
        bytes in the write. On the PC, Seisan version 6.0 and earlier using
        Microsoft Fortran, the first 2 bytes in the file are the ASCII
        character "KP". Every write is preceded and terminated with one byte
        giving the number of bytes in the write. If the write contains more
        than 128 bytes, it is blocked in records of 128 bytes, each with the
        start and end byte which in this case is the number 128. Each record is
        thus 130 bytes long. All of these additional bytes are transparent to
        the user if the file is read as an unformatted file. However, since the
        structure is different on Sun, Linux, MacOSX and PC, a file written as
        unformatted on Sun, Linux or MacOSX cannot be read as unformatted on PC
        or vice versa.

        The files are very easy to write and read on the same computer but
        difficult to read if written on a different computer. To further
        complicate matters, the byte order is different on Sun and PC. With 64
        bit systems, 8 bytes is used to define number of bytes written. This
        type of file can also be read with SEISAN, but so far only data written
        on Linux have been tested for reading on all systems.

        From version 7.0,the Linux and PC file structures are exactly the same.
        On Sun the structure is the same except that the bytes are swapped.
        This is used by SEISAN to find out where the file was written. Since
        there is always 80 characters in the first write, character one in the
        Linux and PC file will be the character P (which is represented by 80)
        while on Sun character 4 is P.
    """
    # check size of data chunk
    if len(data) < 12 * 80:
        return False
    if data[0:2] == 'KP' and data[82] == 'P':
        return ("<", 32, 6)
    elif data[0:8] == '\x00\x00\x00\x00\x00\x00\x00P' and \
            data[88:96] == '\x00\x00\x00\x00\x00\x00\x00P':
        return (">", 64, 7)
    elif data[0:8] == 'P\x00\x00\x00\x00\x00\x00\x00' and \
            data[88:96] == '\x00\x00\x00\x00\x00\x00\x00P':
        return ("<", 64, 7)
    elif data[0:4] == '\x00\x00\x00P' and data[84:88] == '\x00\x00\x00P':
        return (">", 32, 7)
    elif data[0:4] == 'P\x00\x00\x00' and data[84:88] == 'P\x00\x00\x00':
        return ("<", 32, 7)
    return None


def readSEISAN(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a SEISAN file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: SEISAN file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/2001-01-13-1742-24S.KONO__004")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    4 Trace(s) in Stream:
    .KONO.0.B0Z | 2001-01-13T17:45:01.999000Z - ... | 20.0 Hz, 6000 samples
    .KONO.0.L0Z | 2001-01-13T17:42:24.924000Z - ... | 1.0 Hz, 3542 samples
    .KONO.0.L0N | 2001-01-13T17:42:24.924000Z - ... | 1.0 Hz, 3542 samples
    .KONO.0.L0E | 2001-01-13T17:42:24.924000Z - ... | 1.0 Hz, 3542 samples
    """
    def _readline(fh, length=80):
        data = fh.read(length + 8)
        end = length + 4
        start = 4
        return data[start:end]
    # read data chunk from given file
    fh = open(filename, 'rb')
    data = fh.read(80 * 12)
    # get version info from file
    (byteorder, arch, _version) = _getVersion(data)
    # fetch lines
    fh.seek(0)
    # start with event file header
    # line 1
    data = _readline(fh)
    number_of_channels = int(data[30:33])
    # calculate number of lines with channels
    number_of_lines = number_of_channels // 3 + (number_of_channels % 3 and 1)
    if number_of_lines < 10:
        number_of_lines = 10
    # line 2
    data = _readline(fh)
    # line 3
    for _i in xrange(0, number_of_lines):
        data = _readline(fh)
    # now parse each event file channel header + data
    stream = Stream()
    dlen = arch / 8
    dtype = byteorder + 'i' + str(dlen)
    stype = '=i' + str(dlen)
    for _i in xrange(number_of_channels):
        # get channel header
        temp = _readline(fh, 1040)
        # create Stats
        header = Stats()
        header['network'] = (temp[16] + temp[19]).strip()
        header['station'] = temp[0:5].strip()
        header['location'] = (temp[7] + temp[12]).strip()
        header['channel'] = (temp[5:7] + temp[8]).strip()
        header['sampling_rate'] = float(temp[36:43])
        header['npts'] = int(temp[43:50])
        # create start and end times
        year = int(temp[9:12]) + 1900
        month = int(temp[17:19])
        day = int(temp[20:22])
        hour = int(temp[23:25])
        mins = int(temp[26:28])
        secs = float(temp[29:35])
        header['starttime'] = UTCDateTime(year, month, day, hour, mins) + secs
        if headonly:
            # skip data
            fh.seek(dlen * (header['npts'] + 2), 1)
            stream.append(Trace(header=header))
        else:
            # fetch data
            data = np.fromfile(fh, dtype=dtype, count=header['npts'] + 2)
            # convert to system byte order
            data = np.require(data, stype)
            stream.append(Trace(data=data[2:], header=header))
    return stream


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
