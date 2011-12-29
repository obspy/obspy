# -*- coding: utf-8 -*-
"""
Mini-SEED specific utilities.
"""
from __future__ import with_statement

from headers import HPTMODULUS, clibmseed, FRAME, SAMPLESIZES, ENDIAN
from obspy.core import UTCDateTime
from obspy.core.util import scoreatpercentile
from struct import unpack
import ctypes as C
import numpy as np


def getStartAndEndTime(file_or_file_object):
    """
    Returns the start- and endtime of a MiniSEED file or file-like object.

    :type file_or_file_object: basestring or open file-like object.
    :param file_or_file_object: MiniSEED file name or open file-like object
    containing a MiniSEED record.
    :return: tuple (start time of first record, end time of last record)

    This method will return the start time of the first record and the end time
    of the last record. Keep in mind that it will not return the correct result
    if the records in the MiniSEED file do not have a chronological ordering.

    The returned endtime is the time of the last data sample and not the
    time that the last sample covers.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile
    >>> filename = getExampleFile("BW.BGLD.__.EHE.D.2008.001.first_10_records")
    >>> getStartAndEndTime(filename)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))

    It also works with an open file pointer. The file pointer itself will not
    be changed.
    >>> f = open(filename, 'rb')
    >>> getStartAndEndTime(f)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))

    And also with a MiniSEED file stored in a StringIO.
    >>> from StringIO import StringIO
    >>> file_object = StringIO(f.read())
    >>> f.seek(0, 0)
    >>> getStartAndEndTime(file_object)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))
    >>> file_object.close()

    If the file pointer does not point to the first record, the start time will
    refer to the record it points to.
    >>> f.seek(512, 1)
    >>> getStartAndEndTime(f)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2008, 1, 1, 0, 0, 1, 975000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))

    If the file pointer does not point to the first record, the start time will
    refer to the record it points to.
    >>> file_object = StringIO(f.read())
    >>> getStartAndEndTime(file_object)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2008, 1, 1, 0, 0, 1, 975000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))

    >>> f.close()
    """
    # Get the starttime of the first record.
    info = getRecordInformation(file_or_file_object)
    starttime = info['starttime']
    # Get the endtime of the last record.
    info = getRecordInformation(file_or_file_object,
               (info['number_of_records'] - 1) * info['record_length'])
    endtime = info['endtime']
    return starttime, endtime


def getTimingQualityAndDataQualityFlagsCount(file_or_file_object):
    """
    Counts all data quality flags of the given Mini-SEED file and returns
    statistics about the timing quality if applicable.

    :type file_or_file_object: basestring or open file-like object.
    :param file_or_file_object: MiniSEED file name or open file-like object
        containing a MiniSEED record.

    :return: Dictionary with information about the timing quality and the data
        quality flags.

    This method will count all set data quality flag bits in the fixed section
    of the data header in a Mini-SEED file and returns the total count for each
    flag type.
    If the file has a Blockette 1001 statistics about the timing quality will
    also be returned. See the doctests for more information.

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

    >>> from obspy.core.util import getExampleFile
    >>> filename = getExampleFile("qualityflags.mseed")
    >>> getTimingQualityAndDataQualityFlagsCount(filename)
    {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]}

    Also works with file pointers and StringIOs.
    >>> f = open(filename, 'rb')
    >>> getTimingQualityAndDataQualityFlagsCount(f)
    {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]}

    >>> from StringIO import StringIO
    >>> file_object = StringIO(f.read())
    >>> f.close()
    >>> getTimingQualityAndDataQualityFlagsCount(file_object)
    {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]}

    If the file pointer or StringIO position does not correspond to the first
    record the omitted records will be skipped.
    >>> file_object.seek(1024, 1)
    >>> getTimingQualityAndDataQualityFlagsCount(file_object)
    {'data_quality_flags': [8, 8, 7, 6, 5, 4, 3, 2]}
    >>> file_object.close()


    Reading a file with Blockette 1001 will return timing quality statistics.
    The data quality flags will always exists because they are part of the
    fixed MiniSEED header and therefore need to be in every MiniSEED file.
    >>> filename = getExampleFile("timingquality.mseed")
    >>> getTimingQualityAndDataQualityFlagsCount(filename) \
        # doctest: +NORMALIZE_WHITESPACE
    {'timing_quality_upper_quantile': 75.0,
    'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0], 'timing_quality_min': 0.0,
    'timing_quality_lower_quantile': 25.0, 'timing_quality_average': 50.0,
    'timing_quality_median': 50.0, 'timing_quality_max': 100.0}

    Also works with file pointers and StringIOs.
    >>> f = open(filename, 'rb')
    >>> getTimingQualityAndDataQualityFlagsCount(f) \
        # doctest: +NORMALIZE_WHITESPACE
    {'timing_quality_upper_quantile': 75.0,
    'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0], 'timing_quality_min': 0.0,
    'timing_quality_lower_quantile': 25.0, 'timing_quality_average': 50.0,
    'timing_quality_median': 50.0, 'timing_quality_max': 100.0}

    >>> file_object = StringIO(f.read())
    >>> f.close()
    >>> getTimingQualityAndDataQualityFlagsCount(file_object) \
        # doctest: +NORMALIZE_WHITESPACE
    {'timing_quality_upper_quantile': 75.0,
    'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0], 'timing_quality_min': 0.0,
    'timing_quality_lower_quantile': 25.0, 'timing_quality_average': 50.0,
    'timing_quality_median': 50.0, 'timing_quality_max': 100.0}
    >>> file_object.close()
    """
    # Read the first record to get a starting point and.
    info = getRecordInformation(file_or_file_object)
    # Keep track of the extracted information.
    quality_count = [0, 0, 0, 0, 0, 0, 0, 0]
    timing_quality = []
    offset = 0

    # Loop over each record. A valid record needs to have a record length of at
    # least 256 bytes.
    while offset <= (info['filesize'] - 256):
        this_info = getRecordInformation(file_or_file_object, offset)
        # Add the timing quality.
        if 'timing_quality' in this_info:
            timing_quality.append(float(this_info['timing_quality']))
        # Add the value of each bit to the quality_count.
        for _i in xrange(8):
            if (this_info['data_quality_flags'] & (1 << _i)) != 0:
                quality_count[_i] += 1
        offset += this_info['record_length']

    # Collect the results in a dictionary.
    result = {'data_quality_flags': quality_count}

    # Parse of the timing quality list.
    count = len(timing_quality)
    timing_quality = sorted(timing_quality)
    # If no timing_quality was collected just return an empty dictionary.
    if count == 0:
        return result
    # Otherwise calculate some statistical values from the timing quality.
    result['timing_quality_min'] = min(timing_quality)
    result['timing_quality_max'] = max(timing_quality)
    result['timing_quality_average'] = sum(timing_quality) / count
    result['timing_quality_median'] = scoreatpercentile(timing_quality, 50,
                                                        issorted=False)
    result['timing_quality_lower_quantile'] = scoreatpercentile(timing_quality,
                                                  25, issorted=False)
    result['timing_quality_upper_quantile'] = scoreatpercentile(timing_quality,
                                                  75, issorted=False)
    return result


def getRecordInformation(file_or_file_object, offset=0):
    """
    Wrapper around _getRecordInformation to be able to read files and file-like
    objects.
    """
    if isinstance(file_or_file_object, basestring):
        with open(file_or_file_object, 'rb') as f:
            info = _getRecordInformation(f, offset=offset)
    else:
        info =  _getRecordInformation(file_or_file_object, offset=offset)
    return info


def _getRecordInformation(file_object, offset=0):
    """
    Takes the MiniSEED record stored in file_object at the current position and
    returns some information about it.

    If offset is given, the MiniSEED record is assumed to start at current
    position + offset in file_object.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile
    >>> filename = getExampleFile("test.mseed")
    >>> getRecordInformation(filename)  # doctest: +NORMALIZE_WHITESPACE
    {'record_length': 4096, 'data_quality_flags': 0, 'samp_rate': 40.0,
    'byteorder': '>', 'encoding': 11, 'activity_flags': 0, 'excess_bytes': 0,
    'filesize': 8192, 'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 43400),
    'npts': 5980, 'endtime': UTCDateTime(2003, 5, 29, 2, 15, 51, 518400),
    'number_of_records': 2L, 'io_and_clock_flags': 0}
    """
    initial_position = file_object.tell()
    record_start = initial_position

    info = {}

    # Apply the offset.
    file_object.seek(offset, 1)
    record_start += offset

    # Get the size of the buffer.
    file_object.seek(0, 2)
    info['filesize'] = file_object.tell() - record_start
    file_object.seek(record_start, 0)

    # Figure out the byteorder.
    file_object.seek(record_start + 20, 0)
    # Get the year.
    year = unpack('>H', file_object.read(2))[0]
    if year >= 1900 and year <= 2050:
        endian = '>'
    else:
        endian = '<'

    # Seek back and read more information.
    file_object.seek(record_start + 20, 0)
    # Capital letters indicate unsigned quantities.
    values = unpack('%sHHBBBxHHhhBBBxlxxH' % endian, file_object.read(28))
    starttime = UTCDateTime(\
            year=values[0], julday=values[1], hour=values[2], minute=values[3],
            second=values[4], microsecond=values[5] * 100)
    npts = values[6]
    info['npts'] = npts
    samp_rate_factor = values[7]
    samp_rate_mult = values[8]
    info['activity_flags'] = values[9]
    # Bit 1 of the activity flags.
    time_correction_applied = bool(info['activity_flags'] & 2)
    info['io_and_clock_flags'] = values[10]
    info['data_quality_flags'] = values[11]
    time_correction = values[12]
    blkt_offset = values[13]

    # Correct the starttime if applicable.
    if (time_correction_applied is False) and time_correction:
        # Time correction is in units of 0.0001 seconds.
        starttime += time_correction * 0.0001

    # Calculate the sample rate according to the SEED manual.
    if (samp_rate_factor > 0) and (samp_rate_mult) > 0:
        samp_rate = float(samp_rate_factor * samp_rate_mult)
    elif (samp_rate_factor > 0) and (samp_rate_mult) < 0:
        samp_rate = -1.0 * float(samp_rate_factor) / float(samp_rate_mult)
    elif (samp_rate_factor < 0) and (samp_rate_mult) > 0:
        samp_rate = -1.0 * float(samp_rate_mult) / float(samp_rate_factor)
    elif (samp_rate_factor < 0) and (samp_rate_mult) < 0:
        samp_rate = -1.0 / float(samp_rate_factor * samp_rate_mult)

    info['samp_rate'] = samp_rate

    # Traverse the blockettes and parse Blockettes 500, 1000 and/or 1001 if
    # any of those is found.
    while blkt_offset:
        file_object.seek(record_start + blkt_offset, 0)
        blkt_type, blkt_offset = unpack('%sHH' % endian, file_object.read(4))
        # Parse in order of likeliness.
        if blkt_type == 1000:
            encoding, word_order, record_length = unpack('%sBBB' % endian,
                                                  file_object.read(3))
            if ENDIAN[word_order] != endian:
                msg = 'Inconsistent word order.'
                raise Exception(msg)
            info['encoding'] = encoding
            info['record_length'] = 2 ** record_length
        elif blkt_type == 1001:
            info['timing_quality'], mu_sec = unpack('%sBb' % endian,
                                                    file_object.read(2))
            starttime += float(mu_sec) / 1E6
        elif blkt_type == 500:
            file_object.seek(14, 1)
            mu_sec = unpack('%sb' % endian, file_object.read(1))[0]
            starttime += float(mu_sec) / 1E6

    info['starttime'] = starttime
    # Endtime is the time of the last sample.
    info['endtime'] = starttime + (npts - 1) / samp_rate
    info['byteorder'] = endian

    info['number_of_records'] = long(info['filesize'] // \
                                     info['record_length'])
    info['excess_bytes'] = info['filesize'] % info['record_length']

    # Reset file pointer.
    file_object.seek(initial_position, 0)
    return info


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


def _convertMSRToDict(m):
    """
    Internal method used for setting header attributes.
    """
    h = {}
    attributes = ('network', 'station', 'location', 'channel',
                  'dataquality', 'starttime', 'samprate',
                  'samplecnt', 'numsamples', 'sampletype')
    # loop over attributes
    for _i in attributes:
        h[_i] = getattr(m, _i)
    return h


def _convertDatetimeToMSTime(dt):
    """
    Takes a obspy.util.UTCDateTime object and returns an epoch time in ms.

    :param dt: obspy.util.UTCDateTime object.
    """
    return int(dt.timestamp * HPTMODULUS)


def _convertMSTimeToDatetime(timestring):
    """
    Takes a Mini-SEED timestamp and returns a obspy.util.UTCDateTime object.

    :param timestamp: Mini-SEED timestring (Epoch time string in ms).
    """
    return UTCDateTime(timestring / HPTMODULUS)


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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
