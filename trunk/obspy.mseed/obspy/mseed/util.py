# -*- coding: utf-8 -*-
"""
Mini-SEED specific utilities.
"""

from headers import HPTMODULUS, clibmseed, FRAME, SAMPLESIZES
from msstruct import _MSStruct
from obspy.core import UTCDateTime
from obspy.core.util import scoreatpercentile
from struct import unpack
import ctypes as C
import numpy as np
import os


def _unpackSteim1(data_string, npts, swapflag=0, verbose=0):
    """
    Unpack steim1 compressed data given as string.

    :param data_string: data as string
    :param npts: number of data points
    :param swapflag: Swap bytes, defaults to 0
    :return: Return data as numpy.ndarray of dtype int32
    """
    dbuf = data_string
    datasize = len(dbuf)
    samplecnt = npts
    datasamples = np.empty(npts, dtype='int32')
    diffbuff = np.empty(npts, dtype='int32')
    x0 = C.c_int32()
    xn = C.c_int32()
    nsamples = clibmseed.msr_unpack_steim1(\
            C.cast(dbuf, C.POINTER(FRAME)), datasize,
            samplecnt, samplecnt, datasamples, diffbuff,
            C.byref(x0), C.byref(xn), swapflag, verbose)
    if nsamples != npts:
        raise Exception("Error in unpack_steim1")
    return datasamples


def _unpackSteim2(data_string, npts, swapflag=0, verbose=0):
    """
    Unpack steim2 compressed data given as string.

    :param data_string: data as string
    :param npts: number of data points
    :param swapflag: Swap bytes, defaults to 0
    :return: Return data as numpy.ndarray of dtype int32
    """
    dbuf = data_string
    datasize = len(dbuf)
    samplecnt = npts
    datasamples = np.empty(npts, dtype='int32')
    diffbuff = np.empty(npts, dtype='int32')
    x0 = C.c_int32()
    xn = C.c_int32()
    nsamples = clibmseed.msr_unpack_steim2(\
            C.cast(dbuf, C.POINTER(FRAME)), datasize,
            samplecnt, samplecnt, datasamples, diffbuff,
            C.byref(x0), C.byref(xn), swapflag, verbose)
    if nsamples != npts:
        raise Exception("Error in unpack_steim2")
    return datasamples


def _readQuality(file, filepos, chain, tq, dq):
    """
    Reads all quality informations from a file and writes it to tq and dq.
    """
    # Seek to correct position.
    file.seek(filepos, 0)
    # Skip non data records.
    data = file.read(39)
    if data[6] == 'D':
        # Read data quality byte.
        data_quality_flags = data[38]
        # Unpack the binary data.
        data_quality_flags = unpack('B', data_quality_flags)[0]
        # Add the value of each bit to the quality_count.
        for _j in xrange(8):
            if (data_quality_flags & (1 << _j)) != 0:
                dq[_j] += 1
    try:
        # Get timing quality in blockette 1001.
        tq.append(float(chain.Blkt1001.contents.timing_qual))
    except:
        pass


def _ctypesArray2NumpyArray(buffer, buffer_elements, sampletype):
    """
    Takes a Ctypes array and its length and type and returns it as a
    NumPy array.

    This works by reference and no data is copied.

    :param buffer: Ctypes c_void_p pointer to buffer.
    :param buffer_elements: length of the whole buffer
    :param sampletype: type of sample, on of "a", "i", "f", "d"
    """
    # Allocate NumPy array to move memory to
    numpy_array = np.empty(buffer_elements, dtype=sampletype)
    datptr = numpy_array.ctypes.get_data()
    # Manually copy the contents of the C allocated memory area to
    # the address of the previously created NumPy array
    C.memmove(datptr, buffer, buffer_elements * SAMPLESIZES[sampletype])
    return numpy_array


def _convertDatetimeToMSTime(dt):
    """
    Takes obspy.util.UTCDateTime object and returns an epoch time in ms.

    :param dt: obspy.util.UTCDateTime object.
    """
    return int(dt.timestamp * HPTMODULUS)


def _convertMSTimeToDatetime(timestring):
    """
    Takes Mini-SEED timestamp and returns a obspy.util.UTCDateTime object.

    :param timestamp: Mini-SEED timestring (Epoch time string in ms).
    """
    return UTCDateTime(timestring / HPTMODULUS)


def getFileformatInformation(filename):
    """
    Reads the first record and returns information about the Mini-SEED file.

    :type filename: str
    :param filename: MiniSEED file name.
    :return: Dictionary containing record length (``reclen``), ``encoding`` and
        ``byteorder`` of the first record of given Mini-SEED file.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile  # needed to get the \
absolute path of test file
    >>> filename = getExampleFile("test.mseed")
    >>> getFileformatInformation(filename)  # doctest: +NORMALIZE_WHITESPACE
    {'reclen': 4096, 'encoding': 11, 'byteorder': 1}
    """
    # Create _MSStruct instance to read the file.
    ms = _MSStruct(filename)
    chain = ms.msr.contents
    # Read all interesting attributes.
    attribs = ['byteorder', 'encoding', 'reclen']
    info = {}
    for attr in attribs:
        info[attr] = getattr(chain, attr)
    # Will delete C pointers and structures.
    del ms
    return info


def _getMSFileInfo(f, real_name):
    """
    Takes a Mini-SEED filename as an argument and returns a dictionary
    with some basic information about the file. Also suiteable for Full
    SEED.

    :param f: File pointer of opened file in binary format
    :param real_name: Realname of the file, needed for calculating size
    """
    # get size of file
    info = {'filesize': os.path.getsize(real_name)}
    pos = f.tell()
    f.seek(0)
    rec_buffer = f.read(512)
    info['record_length'] = clibmseed.ms_detect(rec_buffer, 512)
    # Calculate Number of Records
    info['number_of_records'] = long(info['filesize'] // \
                                     info['record_length'])
    info['excess_bytes'] = info['filesize'] % info['record_length']
    f.seek(pos)
    return info


def getStartAndEndTime(filename):
    """
    Returns the start- and endtime of a MiniSEED file.

    :type filename: str
    :param filename: MiniSEED file name.
    :return: tuple (start time of first record, end time of last record)

    This method returns the start- and endtime of a MiniSEED file as a tuple
    containing two datetime objects. It only reads the first and the last
    record. Thus it will only work correctly for files containing only one
    trace with all records in the correct order.

    The returned endtime is the time of the last datasample and not the
    time that the last sample covers.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile  # needed to get the \
absolute path of test file
    >>> filename = getExampleFile("BW.BGLD.__.EHE.D.2008.001.first_10_records")
    >>> getStartAndEndTime(filename)  # doctest: +NORMALIZE_WHITESPACE
    (UTCDateTime(2007, 12, 31, 23, 59, 59, 915000), \
UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))
    """
    # Get the starttime
    ms = _MSStruct(filename)
    starttime = ms.getStart()
    # Get the endtime
    ms.offset = ms.filePosFromRecNum(record_number=-1)
    endtime = ms.getEnd()
    del ms  # for valgrind
    return starttime, endtime


def getTimingQuality(filename, first_record=True, rl_autodetection=-1):
    """
    Reads timing quality and returns statistics about it.

    :type filename: str
    :param filename: MiniSEED file name.
    :param first_record: Determines whether all records are assumed to
        either have a timing quality in Blockette 1001 or not depending on
        whether the first records has one. If True and the first records
        does not have a timing quality it will not parse the whole file. If
        False is will parse the whole file anyway and search for a timing
        quality in each record. Defaults to True.
    :param rl_autodetection: Determines the auto-detection of the record
        lengths in the file. If 0 only the length of the first record is
        detected automatically. All subsequent records are then assumed
        to have the same record length. If -1 the length of each record
        is automatically detected. Defaults to -1.
    :return: Dictionary of quality statistics.

    This method will read the timing quality in Blockette 1001 for each
    record in the file if available and return the following statistics:
    Minima, maxima, average, median and upper and lower quantile.

    It is probably pretty safe to set the first_record parameter to True
    because the timing quality is a vendor specific value and thus it will
    probably be set for each record or for none.

    The method to calculate the quantiles uses a integer round outwards
    policy: lower quantiles are rounded down (probability < 0.5), and upper
    quantiles (probability > 0.5) are rounded up.
    This gives no more than the requested probability in the tails, and at
    least the requested probability in the central area.
    The median is calculating by either taking the middle value or, with an
    even numbers of values, the average between the two middle values.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile  # needed to get the \
absolute path of test file
    >>> filename = getExampleFile("timingquality.mseed")
    >>> getTimingQuality(filename)  # doctest: +NORMALIZE_WHITESPACE
    {'min': 0.0, 'max': 100.0, 'average': 50.0, 'median': 50.0, \
'upper_quantile': 75.0, 'lower_quantile': 25.0}
    """
    # Get some information about the file.
    fp = open(filename, 'rb')
    fileinfo = _getMSFileInfo(fp, filename)
    fp.close()
    ms = _MSStruct(filename, init_msrmsf=False)
    # Create Timing Quality list.
    data = []
    # Loop over each record
    for _i in xrange(fileinfo['number_of_records']):
        # Loop over every record.
        ms.read(rl_autodetection, 0, 0, 0)
        # Enclose in try-except block because not all records need to
        # have Blockette 1001.
        try:
            # Append timing quality to list.
            tq = ms.msr.contents.Blkt1001.contents.timing_qual
            data.append(float(tq))
        except:
            if first_record:
                break
    # Deallocate for debugging with valgrind
    del ms
    # Length of the list.
    n = len(data)
    data = sorted(data)
    # Create new dictionary.
    result = {}
    # If no data was collected just return an empty list.
    if n == 0:
        return result
    # Calculate some statistical values.
    result['min'] = min(data)
    result['max'] = max(data)
    result['average'] = sum(data) / n
    data = sorted(data)
    result['median'] = scoreatpercentile(data, 50, issorted=False)
    result['lower_quantile'] = scoreatpercentile(data, 25, issorted=False)
    result['upper_quantile'] = scoreatpercentile(data, 75, issorted=False)
    return result


def getDataQualityFlagsCount(filename):
    """
    Counts all data quality flags of the given Mini-SEED file.

    :type filename: str
    :param filename: MiniSEED file name.
    :return: List of all flag counts.

    This method will count all set data quality flag bits in the fixed section
    of the data header in a Mini-SEED file and returns the total count for each
    flag type. This will only work correctly if each record in the file has the
    same record length.

    .. rubric:: Data quality flags

    ========  =================================================
    Bit       Description
    ========  =================================================
    [Bit 0]   Amplifier saturation detected (station dependent)
    [Bit 1]   Digitizer clipping detected
    [Bit 2]   Spikes detected
    [Bit 3]   Glitches detected
    [Bit 4]   Missing/padded data present
    [Bit 5]   Telemetry synchronization error
    [Bit 6]   A digital filter may be charging
    [Bit 7]   Time tag is questionable
    ========  =================================================

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile  # needed to get the \
absolute path of test file
    >>> filename = getExampleFile("qualityflags.mseed")
    >>> getDataQualityFlagsCount(filename)
    [9, 8, 7, 6, 5, 4, 3, 2]
    """
    # Open the file.
    mseedfile = open(filename, 'rb')
    # Get record length of the file.
    info = _getMSFileInfo(mseedfile, filename)
    # This will increase by one for each set quality bit.
    quality_count = [0, 0, 0, 0, 0, 0, 0, 0]
    record_length = info['record_length']
    # Loop over all records.
    for _i in xrange(info['number_of_records']):
        # Skip non data records.
        data = mseedfile.read(39)
        if data[6] != 'D':
            continue
        # Read data quality byte.
        data_quality_flags = data[38]
        # Jump to next byte.
        mseedfile.seek(record_length - 39, 1)
        # Unpack the binary data.
        data_quality_flags = unpack('B', data_quality_flags)[0]
        # Add the value of each bit to the quality_count.
        for _j in xrange(8):
            if (data_quality_flags & (1 << _j)) != 0:
                quality_count[_j] += 1
    mseedfile.close()
    return quality_count


def _convertMSRToDict(m):
    h = {}
    attributes = ('network', 'station', 'location', 'channel',
                  'dataquality', 'starttime', 'samprate',
                  'samplecnt', 'numsamples', 'sampletype')
    # loop over attributes
    for _i in attributes:
        h[_i] = getattr(m, _i)
    return h


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
