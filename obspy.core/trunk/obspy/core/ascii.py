# -*- coding: utf-8 -*-
"""
Simple ASCII time series format

Each contiguous time series segment (no gaps or overlaps) is represented
with a header line followed by data samples in one of two styles: either
sample lists or time-sample pairs.  There are no restrictions on how the
segments are organized into files, a file might contain a single segment
or many, concatenated segments either for the same channel or many
different channels.

Header lines have the general form:

"TIMESERIES SourceName, # samples, # sps, Time, Format, Type, Units"

Header field descriptions:

SourceName:    "Net_Sta_Loc_Chan_Qual", no spaces, quality code optional
# samples:    Number of samples following header
# sps:        Sampling rate in samples per second
Time:        Time of first sample in ISO YYYY-MM-DDTHH:MM:SS.FFFFFF format
Format:        'SLIST' (sample list) or 'TSPAIR' (time-sample pair)
Type:        Sample type 'INTEGER', 'FLOAT' or 'ASCII'
Units:        Units of time-series, e.g. Counts, M/S, etc., should not contain spaces

Example header (no line wrapping):

TIMESERIES NL_HGN_00_BHZ_R, 11947 samples, 40 sps, 2003-05-29T02:13:22.043400, SLIST, INTEGER, Counts


Sample value format:

For the SLIST (sample list) format, samples are listed in 6 columns with 
the time-series incrementing from left to right and wrapping to the next 
line. The time of the first sample is the time listed in the header.

For the TSPAIR (time-sample pair) format, each sample is listed on a 
separate line with a specific time stamp in the same ISO format as used
in the header line.


Example SLIST format:

TIMESERIES NL_HGN_00_BHZ_R, 12 samples, 40 sps, 2003-05-29T02:13:22.043400, SLIST, INTEGER, Counts
      2787        2776        2774        2780        2783        2782
      2776        2766        2759        2760        2765        2767

Example TSPAIR format:

TIMESERIES NL_HGN_00_BHZ_R, 12 samples, 40 sps, 2003-05-29T02:13:22.043400, TSPAIR, INTEGER, Counts
2003-05-29T02:13:22.043400  2787
2003-05-29T02:13:22.068400  2776
2003-05-29T02:13:22.093400  2774
2003-05-29T02:13:22.118400  2780
2003-05-29T02:13:22.143400  2783
2003-05-29T02:13:22.168400  2782
2003-05-29T02:13:22.193400  2776
2003-05-29T02:13:22.218400  2766
2003-05-29T02:13:22.243400  2759
2003-05-29T02:13:22.268400  2760
2003-05-29T02:13:22.293400  2765
2003-05-29T02:13:22.318400  2767


:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from StringIO import StringIO
from obspy.core import Stream, Trace, UTCDateTime, Stats
import numpy as np


def isSLIST(filename):
    """
    Checks whether a file is ASCII SLIST format. Returns True or False.

    Parameters
    ----------

    filename : string
        Name of the ASCII SLIST file to be checked.
    """
    # first six chars should contain 'DELTA:'
    try:
        temp = open(filename, 'rt').readline()
    except:
        return False
    if not temp.startswith('TIMESERIES'):
        return False
    if not 'SLIST' in temp:
        return False
    return True


def readSLIST(filename, headonly=False):
    """
    Reads a ASCII SLIST file and returns an ObsPy Stream object.

    This function should NOT be called directly, it registers via the
    obspy :func:`~obspy.core.stream.read` function, call this instead.

    Parameters
    ----------
    filename : string
        ASC file to be read.
    headonly : bool, optional
        If set to True, read only the head. This is most useful for
        scanning available data in huge (temporary) data sets.

    Returns
    -------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.

    Example
    -------
    >>> from obspy.core import read # doctest: +SKIP
    >>> st = read("slist.ascii") # doctest: +SKIP
    """
    fh = open(filename, 'rt')
    # read file and split text into channels
    headers = {}
    key = None
    for line in fh.xreadlines():
        if line.isspace():
            # blank line
            continue
        elif line.startswith('TIMESERIES'):
            # new header line
            key = line
            headers[key] = StringIO()
        elif headonly:
            # skip data for option headonly
            continue
        elif key:
            # data entry - may be written in multiple columns
            headers[key].write(line.strip() + ' ')
    fh.close()
    # create ObsPy stream object
    stream = Stream()
    for header, data in headers.iteritems():
        # create Stats
        stats = Stats()
        parts = header.replace(',', '').split()
        temp = parts[1].split('_')
        stats.network = temp[0]
        stats.station = temp[1]
        stats.location = temp[2]
        stats.channel = temp[3]
        stats.sampling_rate = parts[4]
        # XXX: quality missing yet
        stats.starttime = UTCDateTime(parts[6])
        stats.npts = parts[2]
        if headonly:
            # skip data
            stream.append(Trace(header=stats))
        else:
            # parse data
            data.seek(0)
            if parts[8] == 'INTEGER':
                data = np.loadtxt(data, dtype='int')
            elif parts[8] == 'FLOAT':
                data = np.loadtxt(data, dtype='float32')
            else:
                raise NotImplementedError
            stream.append(Trace(data=data, header=stats))
    return stream
