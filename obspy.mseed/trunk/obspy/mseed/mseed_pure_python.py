from conversion_tables import STEIM1_frame_nibbles
from obspy.core import UTCDateTime
from struct import unpack
import time


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
#    dd = obspy.read('tests/data/BW.BGLD..EHE.D.2008.001')
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

