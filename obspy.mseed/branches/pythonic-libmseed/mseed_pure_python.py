from conversion_tables import STEIM1_frame_nibbles
from obspy.core import UTCDateTime
from struct import unpack
import time

def _getMSStarttime(self, open_file):
    """
    Returns the starttime of the given MiniSEED record and returns a 
    UTCDateTime object.
    
    Due to various possible time correction it is complicated to get the
    actual start time of a MiniSEED file. This method hopefully handles
    all possible cases.
    
    It evaluates fields 8, 12 and 16 of the fixed header section and
    additionally blockettes 500 and 1001 which both contain (the same?)
    microsecond correction. If both blockettes are available only blockette
    1001 is used. Not sure if this is the correct way to handle it but
    I could not find anything in the SEED manual nor an example file.
    
    Please see the SEED manual for additional information.
    
    @param open_file: Open file or StringIO. The pointer has to be set
        at the beginning of the record of interst. When the method is
        done with the calulations it will reset the file pointer to the
        original state.
    """
    # Save the originial state of the file pointer.
    file_pointer_start = open_file.tell()
    # Jump to the beginning of field 8 and read the rest of the fixed
    # header section.
    open_file.seek(20, 1)
    # Unpack the starttime, field 12, 16, 17 and 18.
    unpacked_tuple = unpack('>HHBBBxHxxxxxxBxxxiHH', open_file.read(28))
    # Use field 17 to calculate how long all blockettes are and read them.
    blockettes = open_file.read(unpacked_tuple[-2] - 48)
    # Reset the file_pointer
    open_file.seek(file_pointer_start, 0)
    time_correction = 0
    # Check if bit 1 of field 12 has not been set.
    if unpacked_tuple[6] & 2 == 0:
        # If it has not been set the time correction of field 16 still
        # needs to be applied. The units are in 0.0001 seconds.
        time_correction += unpacked_tuple[7] * 100
    # Loop through the blockettes to find blockettes 500 and 1001.
    offset = 0
    blkt_500 = 0
    blkt_1001 = 0
    while True:
        # Get blockette number.
        cur_blockette = unpack('>H', blockettes[offset : offset + 2])
        if cur_blockette == 1001:
            blkt_1001 = unpack('>H', blockettes[5])
            if unpack('>H', blockettes[offset + 2 : offset + 4]) == 0:
                break
        if cur_blockette == 500:
            blkt_500 = unpack('>H', blockettes[19])
            if unpack('>H', blockettes[offset + 2 : offset + 4]) == 0:
                break
        next_blockette = unpack('>H',
                                blockettes[offset + 2 : offset + 4])[0]
        # Leave the loop if no further blockettes follow.
        if next_blockette == 0:
            break
        # New offset.
        offset = next_blockette - 48
    # Adjust the starrtime. Blockette 1001 will override blkt_500.
    additional_correction = 0
    if blkt_500:
        additional_correction = blkt_500
    if blkt_1001:
        additional_correction = blkt_1001
    # Return a UTCDateTime object with the applied corrections.
    starttime = UTCDateTime(year=unpacked_tuple[0],
                    julday=unpacked_tuple[1], hour=unpacked_tuple[2],
                    minute=unpacked_tuple[3], second=unpacked_tuple[4],
                    microsecond=unpacked_tuple[5] * 100)
    # Due to weird bug a difference between positive and negative offsets
    # is needed.
    total_correction = time_correction + additional_correction
    if total_correction < 0:
        starttime = starttime - abs(total_correction) / 1e6
    else:
        starttime = starttime + total_correction / 1e6
    return starttime


def getTimingQuality(self, filename, first_record=True):
    """
    Reads timing quality and returns a dictionary containing statistics
    about it.
    
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
    
    @param filename: MiniSEED file to be parsed.
    @param first_record: Determines whether all records are assumed to 
        either have a timing quality in Blockette 1001 or not depending on
        whether the first records has one. If True and the first records
        does not have a timing quality it will not parse the whole file. If
        False is will parse the whole file anyway and search for a timing
        quality in each record. Defaults to True.
    """
    # Create Timing Quality list.
    data = []
    # Open file.
    mseed_file = open(filename, 'rb')
    filesize = os.path.getsize(filename)
    # Loop over all records. After each loop the file pointer is supposed
    # to be at the beginning of the next record.
    while True:
        starting_pointer = mseed_file.tell()
        # Unpack field 17 and 18 of the fixed section of the data header.
        mseed_file.seek(44, 1)
        (beginning_of_data, first_blockette) = unpack('>HH',
                                                      mseed_file.read(4))
        # Jump to the first blockette.
        mseed_file.seek(first_blockette - 48, 1)
        # Read all blockettes.
        blockettes = mseed_file.read(beginning_of_data - first_blockette)
        # Loop over all blockettes and find number 1000 and 1001.
        offset = 0
        record_length = None
        timing_quality = None
        blockettes_length = len(blockettes)
        while True:
            # Double check to avoid infinite loop.
            if offset >= blockettes_length:
                break
            (blkt_number, next_blkt) = unpack('>HH',
                                        blockettes[offset : offset + 4])
            if blkt_number == 1000:
                record_length = 2 ** unpack('>B',
                                            blockettes[offset + 6])[0]
            elif blkt_number == 1001:
                timing_quality = unpack('>B',
                                        blockettes[offset + 4])[0]
            # Leave loop if no more blockettes follow.
            if next_blkt == 0:
                break
            # New offset.
            offset = next_blkt - first_blockette
        # If no Blockette 1000 could be found raise warning.
        if not record_length:
            msg = 'No blockette 1000 found to determine record length'
            raise Exception(msg)
        end_pointer = starting_pointer + record_length
        # Set the file pointer to the beginning of the next record.
        mseed_file.seek(end_pointer)
        # Leave the loop if first record is set and no timing quality
        # could be found.
        if first_record and timing_quality == None:
            break
        if timing_quality != None:
            data.append(timing_quality)
        # Leave the loop when all records have been processed.
        if end_pointer >= filesize:
            break
    # Create new dictionary.
    result = {}
    # Length of the list.
    n = len(data)
    data = sorted(data)
    # If no data was collected just return an empty list.
    if n == 0:
        return result
    # Calculate some statistical values.
    result['min'] = min(data)
    result['max'] = max(data)
    result['average'] = sum(data) / n
    data = sorted(data)
    result['median'] = scoreatpercentile(data, 50, sort=False)
    result['lower_quantile'] = scoreatpercentile(data, 25, sort=False)
    result['upper_quantile'] = scoreatpercentile(data, 75, sort=False)
    return result

def _calculateSamplingRate(self, samp_rate_factor, samp_rate_multiplier):
    """
    Calculates the actual sampling rate of the record.
    
    This is needed for manual readimg of MiniSEED headers. See the SEED
    Manual page 100 for details.
    
    @param samp_rate_factor: Field 10 of the fixed header in MiniSEED.
    @param samp_rate_multiplier: Field 11 of the fixed header in MiniSEED.
    """
    # Case 1
    if samp_rate_factor > 0 and samp_rate_multiplier > 0:
        return samp_rate_factor * float(samp_rate_multiplier)
    # Case 2
    elif samp_rate_factor > 0 and samp_rate_multiplier < 0:
        # Using float is needed to avoid integer division.
        return - 1 * samp_rate_factor / float(samp_rate_multiplier)
    # Case 3
    elif samp_rate_factor < 0 and samp_rate_multiplier > 0:
        return - 1 * samp_rate_multiplier / float(samp_rate_factor)
    # Case 4
    elif samp_rate_factor < 0 and samp_rate_multiplier < 0:
        return float(1) / (samp_rate_multiplier * samp_rate_factor)
    else:
        msg = 'The sampling rate of the record could not be determined.'
        raise Exception(msg)

def _getMSFileInfo(self, filename, real_name=None):
    """
    Takes a MiniSEED filename or an open file/StringIO as an argument and
    returns a dictionary with some basic information about the file.
    
    The information returned is: filesize, record_length,
    number_of_records and excess_bytes (bytes at the end not used by any
    record).
    
    If filename is an open file/StringIO the file pointer will not be
    changed by this method.
    
    @param filename: MiniSEED file string or an already open file.
    @param real_name: If filename is an open file you need to support the
        filesystem name of it so that the method is able to determine the
        file size. Use None if filename is a file string. Defaults to None.
    """
    info = {}
    # Filename is a true filename.
    if isinstance(filename, basestring) and not real_name:
        info['filesize'] = os.path.getsize(filename)
        #Open file and get record length using libmseed.
        mseed_file = open(filename, 'rb')
        starting_pointer = None
    # Filename is an open file and real_name is a string that refers to
    # a file.
    elif (isinstance(filename, file) or isinstance(filename, StringIO)) \
            and isinstance(real_name, basestring):
        # Save file pointer to restore it later on.
        starting_pointer = filename.tell()
        mseed_file = filename
        info['filesize'] = os.path.getsize(real_name)
    # Otherwise raise error.
    else:
        msg = 'filename either needs to be a string with a filename ' + \
              'or a file/StringIO object. If its a filename real_' + \
              'name needs to be None, otherwise a string with a filename.'
        raise TypeError(msg)
    # Read all blockettes.
    mseed_file.seek(44)
    unpacked_tuple = unpack('>HH', mseed_file.read(4))
    blockettes_offset = unpacked_tuple[1] - 48
    mseed_file.seek(blockettes_offset, 1)
    blockettes = mseed_file.read(unpacked_tuple[0] - unpacked_tuple[1])
    # Loop over blockettes until Blockette 1000 is found.
    offset = 0
    while True:
        two_fields = unpack('>HH', blockettes[offset:offset + 4])
        if two_fields[0] == 1000:
            info['record_length'] = 2 ** unpack('>B', blockettes[6])[0]
            break
        else:
            # Only true when no blockette 1000 is present.
            if two_fields[1] <= 0:
                msg = 'Record length could not be determined due to ' + \
                      'missing blockette 1000'
                raise Exception(msg)
            offset = two_fields[1] - blockettes_offset
        # To avoid an infinite loop the total offset is checked.
        if offset >= len(blockettes):
            msg = 'Record length could not be determined due to ' + \
                      'missing blockette 1000'
            raise Exception(msg)
    # Number of total records.
    info['number_of_records'] = int(info['filesize'] / \
                                    info['record_length'])
    # Excess bytes that do not belong to a record.
    info['excess_bytes'] = info['filesize'] % info['record_length']
    if starting_pointer:
        mseed_file.seek(starting_pointer)
    else:
        mseed_file.close()
    return info

def cutMSFileByRecords(self, filename, starttime=None, endtime=None):
    """
    Cuts a MiniSEED file by cutting at records.
    
    The method takes a MiniSEED file and tries to match it as good as
    possible to the supplied time range. It will simply cut off records
    that are not within the time range. The record that covers the
    start time will be the first record and the one that covers the 
    end time will be the last record.
    
    This method will only work correctly for files containing only traces
    from one single source. All traces have to be in chronological order.
    Also all records in the file need to have the same length.
    
    It will return an empty string if the file does not cover the desired
    range.
    
    @return: Byte string containing the cut file.
    
    @param filename: File string of the MiniSEED file to be cut.
    @param starttime: L{obspy.core.UTCDateTime} object.
    @param endtime: L{obspy.core.UTCDateTime} object.
    """
    # Read the start and end time of the file.
    (start, end) = self.getStartAndEndTime(filename)
    # Set the start time.
    if not starttime or starttime <= start:
        starttime = start
    elif starttime >= end:
        return ''
    # Set the end time.
    if not endtime or endtime >= end:
        endtime = end
    elif endtime <= start:
        return ''
    fh = open(filename, 'rb')
    # Guess the most likely records that cover start- and end time.
    info = self._getMSFileInfo(fh, filename)
    nr = info['number_of_records']
    start_record = int((starttime - start) / (end - start) * nr)
    end_record = int((endtime - start) / (end - start) * nr) + 1
    # Loop until the correct start_record is found
    delta = 0
    while True:
        # check boundaries
        if start_record < 0:
            start_record = 0
            break
        elif start_record > nr - 1:
            start_record = nr - 1
            break
        fh.seek(start_record * info['record_length'])
        stime = self._getMSStarttime(fh)
        # Calculate last covered record.
        fh.seek(30, 1)
        (npts, sr_factor, sr_multiplier) = unpack('>Hhh', fh.read(6))
        # Calculate sample rate.
        sample_rate = self._calculateSamplingRate(sr_factor, sr_multiplier)
        # Calculate time of the first sample of new record
        etime = stime + ((npts - 1) / sample_rate)
        # Leave loop if correct record is found or change record number
        # otherwise. 
        if starttime >= stime and starttime <= etime:
            break
        elif delta == -1 and starttime > etime:
            break
        elif delta == 1 and starttime < stime:
            start_record += 1
            break
        elif starttime < stime:
            delta = -1
        else:
            delta = 1
        start_record += delta
    # Loop until the correct end_record is found
    delta = 0
    while True:
        # check boundaries
        if end_record < 0:
            end_record = 0
            break
        elif end_record > nr - 1:
            end_record = nr - 1
            break
        fh.seek(end_record * info['record_length'])
        stime = self._getMSStarttime(fh)
        # Calculate last covered record.
        fh.seek(30, 1)
        (npts, sr_factor, sr_multiplier) = unpack('>Hhh', fh.read(6))
        # Calculate sample rate.
        sample_rate = self._calculateSamplingRate(sr_factor, sr_multiplier)
        # The time of the last covered sample is now:
        etime = stime + ((npts - 1) / sample_rate)
        # Leave loop if correct record is found or change record number
        # otherwise.
        if endtime >= stime and endtime <= etime:
            break
        elif delta == -1 and endtime > etime:
            end_record += 1
            break
        elif delta == 1 and endtime < stime:
            break
        elif endtime < stime:
            delta = -1
        else:
            delta = 1
        end_record += delta
    # Open the file and read the cut file.
    record_length = info['record_length']
    # Jump to starting location.
    fh.seek(record_length * start_record, 0)
    # Read until end_location.
    data = fh.read(record_length * (end_record - start_record + 1))
    fh.close()
    # Return the cut file string.
    return data

def getStartAndEndTime(self, filename):
    """
    Returns the start- and end time of a MiniSEED file as a tuple
    containing two datetime objects.
    
    This method only reads the first and the last record. Thus it will only
    work correctly for files containing only one trace with all records
    in the correct order and all records necessarily need to have the same
    record length.
    
    The returned end time is the time of the last datasample and not the
    time that the last sample covers.
    
    It is written in pure Python to resolve some memory issues present
    with creating file pointers and passing them to the libmseed.
    
    @param filename: MiniSEED file string.
    """
    # Open the file of interest ad jump to the beginning of the timing
    # information in the file.
    mseed_file = open(filename, 'rb')
    # Get some general information of the file.
    info = self._getMSFileInfo(mseed_file, filename)
    # Get the start time.
    starttime = self._getMSStarttime(mseed_file)
    # Jump to the last record.
    mseed_file.seek(info['filesize'] - info['excess_bytes'] - \
                    info['record_length'])
    # Starttime of the last record.
    last_record_starttime = self._getMSStarttime(mseed_file)
    # Get the number of samples, the sample rate factor and the sample
    # rate multiplier.
    mseed_file.seek(30, 1)
    (npts, sample_rate_factor, sample_rate_multiplier) = \
        unpack('>Hhh', mseed_file.read(6))
    # Calculate sample rate.
    sample_rate = self._calculateSamplingRate(sample_rate_factor, \
                                              sample_rate_multiplier)
    # The time of the last covered sample is now:
    endtime = last_record_starttime + ((npts - 1) / sample_rate)
    return(starttime, endtime)


def getFirstRecordHeaderInfo(self, filename):
    """
    Takes a MiniSEED file and returns header of the first record.
    
    Returns a dictionary containing some header information from the first
    record of the MiniSEED file only. It returns the location, network,
    station and channel information.
    
    @param filename: MiniSEED file string.
    """
    # open file and jump to beginning of the data of interest.
    mseed_file = open(filename, 'rb')
    mseed_file.seek(8)
    # Unpack the information using big endian byte order.
    unpacked_tuple = unpack('>cccccccccccc', mseed_file.read(12))
    # Close the file.
    mseed_file.close()
    # Return a dictionary containing the necessary information.
    return \
        {'station' : ''.join([_i for _i in unpacked_tuple[0:5]]).strip(),
         'location' : ''.join([_i for _i in unpacked_tuple[5:7]]).strip(),
         'channel' :''.join([_i for _i in unpacked_tuple[7:10]]).strip(),
         'network' : ''.join([_i for _i in unpacked_tuple[10:12]]).strip()}


def get_samplingRate(samp_rate_factor, samp_rate_multiplier):
    """
    Calculates the actual sampling rate of the record. See the SEED Manual
    page 100 for details.
    
    @param samp_rate_factor: Field 10 of the fixed header in Mini-SEED.
    @param samp_rate_multiplier: Field 11 of the fixed header in Mini-SEED.
    """
    # Case 1
    if samp_rate_factor > 0 and samp_rate_multiplier > 0:
        return samp_rate_factor * samp_rate_multiplier
    # Case 2
    elif samp_rate_factor > 0 and samp_rate_multiplier < 0:
        return - 1 * samp_rate_factor / samp_rate_multiplier
    # Case 3
    elif samp_rate_factor < 0 and samp_rate_multiplier > 0:
        return - 1 * samp_rate_multiplier / samp_rate_factor
    # Case 4
    elif samp_rate_factor < 0 and samp_rate_multiplier < 0:
        return 1 / (samp_rate_multiplier * samp_rate_factor)
    else:
        msg = 'The sampling rate could not be determined.'
        raise Exception(msg)

def read_MSEED(file_string):
    """
    Reads a STEIM1 encoded Mini-SEED file as fast a possible using only Python.
    
    This will only work with STEIM1 encoded files that behave 'nice' as it is
    intended to test how fast in can work using pure Python.
    Only works for big-endian byte-order files.
    
    @param file_string: Open file or StreamIO object.
    """
    mseed_structure = []
    # Loop over the file.
    while True:
        # Starting point for the file pointer needed for absolute positioning
        # used in the Mini-SEED file format.
        starting_pointer = file_string.tell()
        # Read fixed header.
        fixed_header = file_string.read(48)
        # Check if the header as 48 bytes. Otherwise end the loop.
        if len(fixed_header) < 48:
            break
        # Unpack the whole header at once. Fields not needed in this case are
        # marked with a pad byte for speed issues.
        unpacked_tuple = unpack('>xxxxxxxxccccccccccccHHBBBBHHHHxxxxxxxxHH',
                            fixed_header)
        # Create list containing network, location, station, channel,
        # starttime, sampling_rate and number of samples
        temp_list = [''.join([_i for _i in unpacked_tuple[10:12]]),
                     ''.join([_i for _i in unpacked_tuple[5:7]]),
                     ''.join([_i for _i in unpacked_tuple[0:5]]),
                     ''.join([_i for _i in unpacked_tuple[7:10]]),
                     UTCDateTime(unpacked_tuple[12], 1, 1,
                     unpacked_tuple[14], unpacked_tuple[15], unpacked_tuple[16],
                     unpacked_tuple[17] * 100) + (unpacked_tuple[12] - 1) * \
                     24 * 60 * 60, get_samplingRate(unpacked_tuple[20],
                     unpacked_tuple[21]), unpacked_tuple[19]]
        # Loop through the blockettes until blockette 1000 is found. The file
        # pointer is always supposed to be at the beginning of the next
        # blockette.
        while True:
            if unpack('>H', file_string.read(2))[0] == 1000:
                # Read encoding.
                file_string.seek(2, 1)
                encoding = unpack('>B', file_string.read(1))[0]
                break
            else:
                file_string.seek(starting_pointer + \
                                  unpack('>H', file_string.read(2))[0] , 0)
        # Test for STEIM1 encoding.
        if encoding == 10:
            # Call the STEIM1 unpacking routine with the file pointer at the
            # beginning of the packed data. The second argument is the number
            # of packed samples.
            file_string.seek(starting_pointer + unpacked_tuple[-2])
            temp_list.append(unpack_Steim1(file_string, temp_list[-1]))
        else:
            msg = 'Currently only supports STEIM1 encoded files.'
            raise NotImplementedError(msg)
        if len(mseed_structure) == 0:
            mseed_structure.append(temp_list)
        else:
            cur_structure = mseed_structure[-1]
            if cur_structure[0:4] == temp_list[0:4] and temp_list[4] == \
                cur_structure[4] + cur_structure[6] * 1 / \
                cur_structure[5]:
                cur_structure.extend(temp_list[-1])
                cur_structure[6] += temp_list[6]
    import pdb;pdb.set_trace()

def unpack_Steim1(file_string, npts):
    """
    Unpacks STEIM1 packed data and returns a numpy ndarray.
    
    This tries to be as fast as possible and therefore the code is sometimes
    redundant and not necessarly pretty.
    
    The big problem is to get it all into a numpy array.
    """
    sample_count = 0
    # Create empty list.
    unpacked_data = []
    # The first word contains encoding information about the next 15 words.
    conversions_tuples = [STEIM1_frame_nibbles[_j] for _j in \
           unpack('>BBBB', file_string.read(4))]
    conversions = []
    for _j in conversions_tuples:
        conversions.extend(_j)
    # Remove the first three items from the conversions.
    del conversions[0:3]
    # The second word of the first frame always is the start value. This
    # seems to be undocumented.
    unpacked_data.append(unpack('>i', file_string.read(4))[0])
    sample_count += 1
    # The third word is the last value of the last frame and is used for
    # consistency checks. The position of this number also seems to be
    # undocumented.
    last_value = unpack('>i', file_string.read(4))[0]
    # Check if all other conversions are 1 (8-bit differences). As this
    # mostly is the case it is worth the tradeoff for the three extra
    # checks.
    if not 0 in conversions and not 10 in conversions and not 11 in \
        conversions:
        # The next word is also a special case. The first difference is
        # meaningless
        for _i in unpack('>xbbb', file_string.read(4)):
            unpacked_data.append(unpacked_data[-1] + _i)
        sample_count += 3
        # Loop over all remaining words.
        for _i in range(12):
            for _j in unpack('>bbbb', file_string.read(4)):
                unpacked_data.append(unpacked_data[-1] + _j)
            sample_count += 4
    else:
        # The next word is also a case. The first difference is meaningless.
        if conversions[0] == 1:
            for _i in unpack('>xbbb', file_string.read(4)):
                unpacked_data.append(unpacked_data[-1] + _i)
            sample_count += 3
        elif conversions[0] == 10:
            unpacked_data.append(unpacked_data[-1] + unpack('>xxh', \
                                                    file_string.read(4))[0])
            sample_count += 1
        elif conversions[0] == 11:
            # If the first one is meaningless and its the only data value skip
            # it.
            file.string.seek(4, 1)
        # Delete the just used conversion to avoid later confusion.
        del conversions[0]
        # Loop over all remaining words.
        for _i in xrange(12):
            # Four 8-bit differences.
            if conversions[_i] == 1:
                for _j in unpack('>bbbb', file_string.read(4)):
                    unpacked_data.append(unpacked_data[-1] + _j)
                sample_count += 4
            # Two 16-bit differences.
            elif conversions[_i] == 10:
                for _j in unpack('>hh', file_string.read(4)):
                    unpacked_data.append(unpacked_data[-1] + _j)
                sample_count += 2
            # One 32-bit difference.
            elif conversions[_i] == 11:
                unpacked_data.append(unpacked_data[-1] + \
                                     unpack('>bbbb', file_string.read(4))[0])
                sample_count += 1
    # Loop over all remaining frames.
    while True:
        if sample_count >= npts:
            break
        conversions_tuples = [STEIM1_frame_nibbles[_j] for _j in \
           unpack('>BBBB', file_string.read(4))]
        conversions = []
        for _j in conversions_tuples:
            conversions.extend(_j)
        # Remove the first items from the conversions.
        del conversions[0]
        # Check if all other conversions are 1 (8-bit differences). As this
        # mostly is the case it is worth the tradeoff for the three extra
        # checks.
        if not 0 in conversions and not 10 in conversions and not 11 in \
            conversions:
            # Loop over all remaining words.
            for _i in xrange(15):
                for _j in unpack('>bbbb', file_string.read(4)):
                    unpacked_data.append(unpacked_data[-1] + _j)
                sample_count += 4
        else:
            for _i in xrange(15):
                # Four 8-bit differences.
                if conversions[_i] == 1:
                    for _j in unpack('>bbbb', file_string.read(4)):
                        unpacked_data.append(unpacked_data[-1] + _j)
                    sample_count += 4
                # Two 16-bit differences.
                elif conversions[_i] == 10:
                    for _j in unpack('>hh', file_string.read(4)):
                        unpacked_data.append(unpacked_data[-1] + _j)
                    sample_count += 2
                # One 32-bit difference.
                elif conversions[_i] == 11:
                    unpacked_data.append(unpacked_data[-1] + \
                                    unpack('>bbbb', file_string.read(4))[0])
                    sample_count += 1
    # Integrity Check
    if unpacked_data[-1] != last_value:
        msg = 'The last data value is not correct! Got %s, expected %s' % \
                (unpacked_data[-1], last_value)
        print file_string.tell()
        raise Warning(msg)
    return unpacked_data

#def readSingleRecord(file_string):
#    """
#    """
#    # Unpack the fixed section of the data header.
#    unpacked_tuple = unpack('>' + 6 * 'c' + 's' + 'x' + 12 * 'c' + 2 * 'H' + \
#                            4 * 'B' + 4 * 'H' + 4 * 'B' + 'l' + 2 * 'H',
#                            file_string.read(48))
#    file_string.seek(-48, 1)
#    fixed_header = {}
#    fixed_header['sequence_number'] = ''.join(_i for _i in unpacked_tuple[0:6])
#    fixed_header['data_header'] = unpacked_tuple[6]
#    fixed_header['station'] = ''.join(_i for _i in \
#                                      unpacked_tuple[7:12]).strip()
#    fixed_header['location'] = ''.join(_i for _i in \
#                                       unpacked_tuple[12:14]).strip()
#    fixed_header['channel'] = ''.join(_i for _i in \
#                                      unpacked_tuple[14:17]).strip()
#    fixed_header['network'] = ''.join(_i for _i in \
#                                      unpacked_tuple[17:19]).strip()
#    # Structure of starttime:
#    # (YEAR, DAY_OF_YEAR, HOURS_OF_DAY, MINUTES_OF_DAY, SECONDS_OF_DAY,
#    #  UNUSED, .0001 seconds)
#    # The current implementation is not the best as it does not account for any
#    # leap seconds that might occur.
#    fixed_header['start_time'] = UTCDateTime(unpacked_tuple[19], 1, 1,
#        unpacked_tuple[21], unpacked_tuple[22], unpacked_tuple[23],
#        unpacked_tuple[25] * 100) + (unpacked_tuple[20] - 1) * 24 * 60 * 60
#    fixed_header['number_of_samples'] = unpacked_tuple[26]
#    fixed_header['sample_rate_factor'] = unpacked_tuple[27]
#    fixed_header['sample_rate_multiplier'] = unpacked_tuple[28]
#    fixed_header['activity_flags'] = unpacked_tuple[29]
#    fixed_header['io_and_clock_flags'] = unpacked_tuple[30]
#    fixed_header['data_quality_flags'] = unpacked_tuple[31]
#    fixed_header['number_of_blockettes_that_follow'] = unpacked_tuple[32]
#    fixed_header['time_correction'] = unpacked_tuple[33]
#    fixed_header['beginning_of_data'] = unpacked_tuple[34]
#    fixed_header['first_blockette'] = unpacked_tuple[35]
#    file_string.seek(fixed_header['beginning_of_data'], 1)
#    # Still packed data.
#    data = file_string.read(512 - \
#                                fixed_header['beginning_of_data'])
#
#    #Manually encoding STEIM1
#    return unpackSteim1(data, fixed_header['number_of_samples'])






#
#
#        # loop over each word
#        for _j in xrange(16):
#            # A conversion of 1 means that the word stores four 8-bit integer
#            # differences.
#            if conversions[_j] == 1:
#                difference_list.extend(unpack('>bbbb', data[frame_start + _j \
#                                    * 4 : frame_start + _j * 4 + 4]))
#            # The first word always contains no data.
#            elif conversions[_j] == 0:
#                if no_data_count == 0:
#                    no_data_count += 1
#                    continue
#                # The second word that contains no data is the start_value
#                elif no_data_count == 1:
#                    no_data_count += 1
#                    first_value = unpack('>i', data[frame_start + _j * 4 : \
#                                    frame_start + _j * 4 + 4])[0]
#                # The third word is the last value for data integrity checks.
#                elif no_data_count == 2:
#                    no_data_count += 1
#                    last_value = unpack('>i', data[frame_start + _j * 4 : \
#                                    frame_start + _j * 4 + 4])[0]
#                else:
#                    continue
#            # 10 means two 16-bit integers.
#            elif conversions[_j] == 10:
#                difference_list.extend(unpack('>hh', data[frame_start + _j \
#                                    * 4 : frame_start + _j * 4 + 4]))
#            # 11 means one 32-bit integer.
#            elif conversions[_j] == 11:
#                difference_list.extend(unpack('>i', data[frame_start + _j \
#                                    * 4 : frame_start + _j * 4 + 4]))
#    # Create the final data list.
#    real_data = [first_value]
#    del difference_list[0]
#    for _j in difference_list:
#        real_data.append(real_data[-1] + _j)
#    # Validate the list via the last sample
#    if real_data[-1] == last_value:
#        #print 'CORRECT!'
#        pass
#    return real_data

#def readFile():
#    file_string = open('tests/data/BW.BGLD..EHE.D.2008.001', 'rb')
#    file_size = os.path.getsize('tests/data/BW.BGLD..EHE.D.2008.001')
#    data = []
#    a = time.time()
#    for _i in xrange(file_size / 512 / 10):
#        file_string.seek(_i * 512, 0)
#        data.extend(readSingleRecord(file_string))
#    b = time.time()
#    print 'Time taken:', b - a

#    a = time.time()
#    data = N.array(data)
#    b = time.time()
#    print 'Time taken for array conversion:', b - a
#
#
#    #print data
#    print len(data)
#
#    import obspy
#    a = time.time()
#    dd = read('tests/data/BW.BGLD..EHE.D.2008.001')
#    b = time.time()
#    print 'Compare Time:', b - a
#    print len(dd[0].data)

#import cProfile
#cProfile.run('readFile()')

def read_DayFile():
    file_string = open('tests/data/BW.BGLD..EHE.D.2008.001', 'rb')
    a = time.time()
    read_MSEED(file_string)
    b = time.time()
    print 'Time taken:', b - a

if __name__ == '__main__':
    read_DayFile()

