# -*- coding: utf-8 -*-
"""
Mini-SEED specific utilities.
"""
from headers import HPTMODULUS, clibmseed, FRAME, SAMPLESIZES, ENDIAN
from obspy import UTCDateTime
from obspy.core.util import scoreatpercentile
from struct import unpack
import sys
import ctypes as C
import numpy as np
import warnings


def getStartAndEndTime(file_or_file_object):
    """
    Returns the start- and endtime of a Mini-SEED file or file-like object.

    :type file_or_file_object: basestring or open file-like object.
    :param file_or_file_object: Mini-SEED file name or open file-like object
        containing a Mini-SEED record.
    :return: tuple (start time of first record, end time of last record)

    This method will return the start time of the first record and the end time
    of the last record. Keep in mind that it will not return the correct result
    if the records in the Mini-SEED file do not have a chronological ordering.

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

    And also with a Mini-SEED file stored in a StringIO.

    >>> from StringIO import StringIO
    >>> file_object = StringIO(f.read())
    >>> getStartAndEndTime(file_object)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2007, 12, 31, 23, 59, 59, 915000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))
    >>> file_object.close()

    If the file pointer does not point to the first record, the start time will
    refer to the record it points to.

    >>> f.seek(512)
    >>> getStartAndEndTime(f)  # doctest: +NORMALIZE_WHITESPACE
        (UTCDateTime(2008, 1, 1, 0, 0, 1, 975000),
        UTCDateTime(2008, 1, 1, 0, 0, 20, 510000))

    The same is valid for a file-like object.

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
    info = getRecordInformation(
        file_or_file_object,
        (info['number_of_records'] - 1) * info['record_length'])
    endtime = info['endtime']
    return starttime, endtime


def getTimingAndDataQuality(file_or_file_object):
    """
    Counts all data quality flags of the given Mini-SEED file and returns
    statistics about the timing quality if applicable.

    :type file_or_file_object: basestring or open file-like object.
    :param file_or_file_object: Mini-SEED file name or open file-like object
        containing a Mini-SEED record.

    :return: Dictionary with information about the timing quality and the data
        quality flags.

    .. rubric:: Data quality

    This method will count all set data quality flag bits in the fixed section
    of the data header in a Mini-SEED file and returns the total count for each
    flag type.

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

    .. rubric:: Timing quality

    If the file has a Blockette 1001 statistics about the timing quality will
    also be returned. See the doctests for more information.

    This method will read the timing quality in Blockette 1001 for each
    record in the file if available and return the following statistics:
    Minima, maxima, average, median and upper and lower quantile.
    Quantiles are calculated using a integer round outwards policy: lower
    quantiles are rounded down (probability < 0.5), and upper quantiles
    (probability > 0.5) are rounded up.
    This gives no more than the requested probability in the tails, and at
    least the requested probability in the central area.
    The median is calculating by either taking the middle value or, with an
    even numbers of values, the average between the two middle values.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile
    >>> filename = getExampleFile("qualityflags.mseed")
    >>> getTimingAndDataQuality(filename)
    {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]}

    Also works with file pointers and StringIOs.

    >>> f = open(filename, 'rb')
    >>> getTimingAndDataQuality(f)
    {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]}

    >>> from StringIO import StringIO
    >>> file_object = StringIO(f.read())
    >>> f.close()
    >>> getTimingAndDataQuality(file_object)
    {'data_quality_flags': [9, 8, 7, 6, 5, 4, 3, 2]}

    If the file pointer or StringIO position does not correspond to the first
    record the omitted records will be skipped.

    >>> file_object.seek(1024, 1)
    >>> getTimingAndDataQuality(file_object)
    {'data_quality_flags': [8, 8, 7, 6, 5, 4, 3, 2]}
    >>> file_object.close()

    Reading a file with Blockette 1001 will return timing quality statistics.
    The data quality flags will always exists because they are part of the
    fixed Mini-SEED header and therefore need to be in every Mini-SEED file.

    >>> filename = getExampleFile("timingquality.mseed")
    >>> getTimingAndDataQuality(filename)  # doctest: +NORMALIZE_WHITESPACE
    {'timing_quality_upper_quantile': 75.0,
    'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0], 'timing_quality_min': 0.0,
    'timing_quality_lower_quantile': 25.0, 'timing_quality_average': 50.0,
    'timing_quality_median': 50.0, 'timing_quality_max': 100.0}

    Also works with file pointers and StringIOs.

    >>> f = open(filename, 'rb')
    >>> getTimingAndDataQuality(f)  # doctest: +NORMALIZE_WHITESPACE
    {'timing_quality_upper_quantile': 75.0,
    'data_quality_flags': [0, 0, 0, 0, 0, 0, 0, 0], 'timing_quality_min': 0.0,
    'timing_quality_lower_quantile': 25.0, 'timing_quality_average': 50.0,
    'timing_quality_median': 50.0, 'timing_quality_max': 100.0}

    >>> file_object = StringIO(f.read())
    >>> f.close()
    >>> getTimingAndDataQuality(file_object)  # doctest: +NORMALIZE_WHITESPACE
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
    result['timing_quality_median'] = \
        scoreatpercentile(timing_quality, 50, issorted=False)
    result['timing_quality_lower_quantile'] = \
        scoreatpercentile(timing_quality, 25, issorted=False)
    result['timing_quality_upper_quantile'] = \
        scoreatpercentile(timing_quality, 75, issorted=False)
    return result


def getRecordInformation(file_or_file_object, offset=0, endian=None):
    """
    Returns record information about given files and file-like object.

    :param endian: If given, the byteorder will be enforced. Can be either "<"
        or ">". If None, it will be determined automatically.
        Defaults to None.

    .. rubric:: Example

    >>> from obspy.core.util import getExampleFile
    >>> filename = getExampleFile("test.mseed")
    >>> getRecordInformation(filename)  # doctest: +NORMALIZE_WHITESPACE
    {'record_length': 4096, 'data_quality_flags': 0, 'activity_flags': 0,
     'byteorder': '>', 'encoding': 11, 'samp_rate': 40.0, 'excess_bytes': 0L,
     'filesize': 8192L,
     'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 43400), 'npts': 5980,
     'endtime': UTCDateTime(2003, 5, 29, 2, 15, 51, 518400),
     'number_of_records': 2L, 'io_and_clock_flags': 0}
    """
    if isinstance(file_or_file_object, basestring):
        with open(file_or_file_object, 'rb') as f:
            info = _getRecordInformation(f, offset=offset, endian=endian)
    else:
        info = _getRecordInformation(file_or_file_object, offset=offset,
                                     endian=endian)
    return info


def _getRecordInformation(file_object, offset=0, endian=None):
    """
    Searches the first Mini-SEED record stored in file_object at the current
    position and returns some information about it.

    If offset is given, the Mini-SEED record is assumed to start at current
    position + offset in file_object.

    :param endian: If given, the byteorder will be enforced. Can be either "<"
        or ">". If None, it will be determined automatically.
        Defaults to None.
    """
    initial_position = file_object.tell()
    record_start = initial_position
    samp_rate = None

    info = {}

    # Apply the offset.
    file_object.seek(offset, 1)
    record_start += offset

    # Get the size of the buffer.
    file_object.seek(0, 2)
    info['filesize'] = long(file_object.tell() - record_start)
    file_object.seek(record_start, 0)

    # check current position
    if info['filesize'] % 256 != 0:
        # if a multiple of minimal record length 256
        record_start = 0
    elif file_object.read(8)[6] not in ['D', 'R', 'Q', 'M']:
        # if valid data record start at all starting with D, R, Q or M
        record_start = 0
    file_object.seek(record_start, 0)

    # check if full SEED or Mini-SEED
    if file_object.read(8)[6] == 'V':
        # found a full SEED record - seek first Mini-SEED record
        # search blockette 005, 008 or 010 which contain the record length
        blockette_id = file_object.read(3)
        while blockette_id not in ['010', '008', '005']:
            if not blockette_id.startswith('0'):
                msg = "SEED Volume Index Control Headers: blockette 0xx" + \
                      " expected, got %s"
                raise Exception(msg % blockette_id)
            # get length and jump to end of current blockette
            blockette_len = int(file_object.read(4))
            file_object.seek(blockette_len - 7, 1)
            # read next blockette id
            blockette_id = file_object.read(3)
        # Skip the next bytes containing length of the blockette and version
        file_object.seek(8, 1)
        # get record length
        rec_len = pow(2, int(file_object.read(2)))
        # reset file pointer
        file_object.seek(record_start, 0)
        # cycle through file using record length until first data record found
        while file_object.read(7)[6] not in ['D', 'R', 'Q', 'M']:
            record_start += rec_len
            file_object.seek(record_start, 0)

    # Use the date to figure out the byteorder.
    file_object.seek(record_start + 20, 0)
    # Capital letters indicate unsigned quantities.
    data = file_object.read(28)
    if endian is None:
        try:
            endian = ">"
            values = unpack('%sHHBBBxHHhhBBBxlxxH' % endian, data)
            starttime = UTCDateTime(
                year=values[0], julday=values[1],
                hour=values[2], minute=values[3], second=values[4],
                microsecond=values[5] * 100)
        except:
            endian = "<"
            values = unpack('%sHHBBBxHHhhBBBxlxxH' % endian, data)
            starttime = UTCDateTime(
                year=values[0], julday=values[1],
                hour=values[2], minute=values[3], second=values[4],
                microsecond=values[5] * 100)
    else:
        values = unpack('%sHHBBBxHHhhBBBxlxxH' % endian, data)
        try:
            starttime = UTCDateTime(
                year=values[0], julday=values[1],
                hour=values[2], minute=values[3], second=values[4],
                microsecond=values[5] * 100)
        except:
            msg = ("Invalid starttime found. The passed byteorder is likely "
                   "wrong.")
            raise ValueError(msg)
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

    # Traverse the blockettes and parse Blockettes 100, 500, 1000 and/or 1001
    # if any of those is found.
    while blkt_offset:
        file_object.seek(record_start + blkt_offset, 0)
        blkt_type, blkt_offset = unpack('%sHH' % endian, file_object.read(4))
        # Parse in order of likeliness.
        if blkt_type == 1000:
            encoding, word_order, record_length = \
                unpack('%sBBB' % endian, file_object.read(3))
            if ENDIAN[word_order] != endian:
                msg = 'Inconsistent word order.'
                warnings.warn(msg, UserWarning)
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
        elif blkt_type == 100:
            samp_rate = unpack('%sf' % endian, file_object.read(4))[0]

    # If samprate not set via blockette 100 calculate the sample rate according
    # to the SEED manual.
    if not samp_rate:
        if (samp_rate_factor > 0) and (samp_rate_mult) > 0:
            samp_rate = float(samp_rate_factor * samp_rate_mult)
        elif (samp_rate_factor > 0) and (samp_rate_mult) < 0:
            samp_rate = -1.0 * float(samp_rate_factor) / float(samp_rate_mult)
        elif (samp_rate_factor < 0) and (samp_rate_mult) > 0:
            samp_rate = -1.0 * float(samp_rate_mult) / float(samp_rate_factor)
        elif (samp_rate_factor < 0) and (samp_rate_mult) < 0:
            samp_rate = -1.0 / float(samp_rate_factor * samp_rate_mult)
        else:
            # if everything is unset or 0 set sample rate to 1
            samp_rate = 1

    info['samp_rate'] = samp_rate

    info['starttime'] = starttime
    # Endtime is the time of the last sample.
    info['endtime'] = starttime + (npts - 1) / samp_rate
    info['byteorder'] = endian

    info['number_of_records'] = long(info['filesize'] //
                                     info['record_length'])
    info['excess_bytes'] = long(info['filesize'] % info['record_length'])

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
    nsamples = clibmseed.msr_unpack_steim1(
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
    nsamples = clibmseed.msr_unpack_steim2(
        C.cast(dbuf, C.POINTER(FRAME)), datasize,
        samplecnt, samplecnt, datasamples, diffbuff,
        C.byref(x0), C.byref(xn), swapflag, verbose)
    if nsamples != npts:
        raise Exception("Error in unpack_steim2")
    return datasamples


def shiftTimeOfFile(input_file, output_file, timeshift):
    """
    Takes a MiniSEED file and shifts the time of every record by the given
    amount.

    The same could be achieved by reading the MiniSEED file with ObsPy,
    modifying the starttime and writing it again. The problem with this
    approach is that some record specific flags and special blockettes might
    not be conserved. This function directly operates on the file and simply
    changes some header fields, not touching the rest, thus preserving it.

    Will only work correctly if all records have the same record length which
    usually should be the case.

    All times are in 0.0001 seconds, that is in 1/10000 seconds. NOT ms but one
    order of magnitude smaller! This is due to the way time corrections are
    stored in the MiniSEED format.

    :type input_file: str
    :param input_file: The input filename.
    :type output_file: str
    :param output_file: The output filename.
    :type timeshift: int
    :param timeshift: The time-shift to be applied in 0.0001, e.g. 1E-4
        seconds. Use an integer number.

    Please do NOT use identical input and output files because if something
    goes wrong, your data WILL be corrupted/destroyed. Also always check the
    resulting output file.

    .. rubric:: Technical details

    The function will loop over every record and change the "Time correction"
    field in the fixed section of the MiniSEED data header by the specified
    amount. Unfortunately a further flag (bit 1 in the "Activity flags" field)
    determines whether or not the time correction has already been applied to
    the record start time. If it has not, all is fine and changing the "Time
    correction" field is enough. Otherwise the actual time also needs to be
    changed.

    One further detail: If bit 1 in the "Activity flags" field is 1 (True) and
    the "Time correction" field is 0, then the bit will be set to 0 and only
    the time correction will be changed thus avoiding the need to change the
    record start time which is preferable.
    """
    timeshift = int(timeshift)
    # A timeshift of zero makes no sense.
    if timeshift == 0:
        msg = "The timeshift must to be not equal to 0."
        raise ValueError(msg)

    # Get the necessary information from the file.
    info = getRecordInformation(input_file)
    record_length = info["record_length"]

    byteorder = info["byteorder"]
    sys_byteorder = "<" if (sys.byteorder == "little") else ">"
    doSwap = False if (byteorder == sys_byteorder) else True

    # This is in this scenario somewhat easier to use than StringIO because one
    # can directly modify the data array.
    data = np.fromfile(input_file, dtype="uint8")
    array_length = len(data)
    record_offset = 0
    # Loop over every record.
    while True:
        remaining_bytes = array_length - record_offset
        if remaining_bytes < 48:
            if remaining_bytes > 0:
                msg = "%i excessive byte(s) in the file. " % remaining_bytes
                msg += "They will be appended to the output file."
                warnings.warn(msg)
            break
        # Use a slice for the current record.
        current_record = data[record_offset: record_offset + record_length]
        record_offset += record_length

        activity_flags = current_record[36]
        is_time_correction_applied = bool(activity_flags & 2)

        current_time_shift = current_record[40:44]
        current_time_shift.dtype = np.int32
        if doSwap:
            current_time_shift = current_time_shift.byteswap(False)
        current_time_shift = current_time_shift[0]

        # If the time correction has been applied, but there is no actual
        # time correction, then simply set the time correction applied
        # field to false and process normally.
        # This should rarely be the case.
        if current_time_shift == 0 and is_time_correction_applied:
            # This sets bit 2 of the activity flags to 0.
            current_record[36] = current_record[36] & (~2)
            is_time_correction_applied = False
        # This is the case if the time correction has been applied. This
        # requires some more work by changing both, the actual time and the
        # time correction field.
        elif is_time_correction_applied:
            msg = "The timeshift can only be applied by actually changing the "
            msg += "time. This is experimental. Please make sure the output "
            msg += "file is correct."
            warnings.warn(msg)
            # The whole process is not particularly fast or optimized but
            # instead intends to avoid errors.
            # Get the time variables.
            time = current_record[20:30]
            year = time[0:2]
            julday = time[2:4]
            hour = time[4:5]
            minute = time[5:6]
            second = time[6:7]
            msecs = time[8:10]
            # Change dtype of multibyte values.
            year.dtype = np.uint16
            julday.dtype = np.uint16
            msecs.dtype = np.uint16
            if doSwap:
                year = year.byteswap(False)
                julday = julday.byteswap(False)
                msecs = msecs.byteswap(False)
            dtime = UTCDateTime(year=year[0], julday=julday[0], hour=hour[0],
                                minute=minute[0], second=second[0],
                                microsecond=msecs[0] * 100)
            dtime += (float(timeshift) / 10000)
            year[0] = dtime.year
            julday[0] = dtime.julday
            hour[0] = dtime.hour
            minute[0] = dtime.minute
            second[0] = dtime.second
            msecs[0] = dtime.microsecond / 100
            # Swap again.
            if doSwap:
                year = year.byteswap(False)
                julday = julday.byteswap(False)
                msecs = msecs.byteswap(False)
            # Change dtypes back.
            year.dtype = np.uint8
            julday.dtype = np.uint8
            msecs.dtype = np.uint8
            # Write to current record.
            time[0:2] = year[:]
            time[2:4] = julday[:]
            time[4] = hour[:]
            time[5] = minute[:]
            time[6] = second[:]
            time[8:10] = msecs[:]
            current_record[20:30] = time[:]

        # Now modify the time correction flag.
        current_time_shift += timeshift
        current_time_shift = np.array([current_time_shift], np.int32)
        if doSwap:
            current_time_shift = current_time_shift.byteswap(False)
        current_time_shift.dtype = np.uint8
        current_record[40:44] = current_time_shift[:]

    # Write to the output file.
    data.tofile(output_file)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
