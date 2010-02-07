from obspy.core import read, Stream, Trace, UTCDateTime
from glob import iglob
import numpy as np
from numpy.ma import is_masked

folder = '/Users/lion/Documents/workspace/TestFiles/archive/RJOB/EHE.D/BW.*'

for file in iglob(folder):
    print '\n'
    print 'Processing %s' % file
    # Read file.
    st = read(file)
    # Check for gaps.
    if len(st) > 1:
        print file, 'has more than one trace.'
        st.merge()
    # Make a wild guess and assume, that the middle of the file is the correct
    # day.
    day = st[0].stats.starttime + (st[0].stats.endtime -
                                   st[0].stats.starttime)/2
    day_starttime = UTCDateTime(day.getDate())
    day_endtime = day_starttime + 24*60*60
    # Get beginning and end samples.
    samples_before_start = None
    samples_after_end = None
    if day_starttime > st[0].stats.starttime:
        samples_before_start = st.slice(st[0].stats.starttime,
                                        day_starttime)[0].data.ptp()
    if st[0].stats.endtime > day_endtime:
        samples_after_end = st.slice(day_endtime, st[0].stats.endtime)[0].data.ptp()
    st = st.slice(day_starttime, day_endtime)
    stats = st[0].stats

    samples_per_spot = int(24*60*60 / 1000.0 * stats.sampling_rate)

    # Get the data.
    data = st[0].data

    # Figure out how many empty 'spots' there are at the beginning of the file.
    empty_spots = 0
    lost_samples = 0
    first_spot = None
    if day_starttime < stats.starttime:
        empty_spots = int(((stats.starttime - day_starttime) *
                          stats.sampling_rate) // \
                          samples_per_spot) + 1
        # Samples that kinda lost in the last empty spot.
        lost_samples = (stats.starttime - day_starttime) % samples_per_spot
        lost_samples = int(samples_per_spot / lost_samples)
        first_spot = data[0:lost_samples].ptp()
    start = lost_samples + 1
    end = start + (len(data) - start) // samples_per_spot * samples_per_spot
    spots = (end - start) / samples_per_spot
    free_samples = int((end-start) % samples_per_spot)
    if free_samples:
        end_samples = data[:-free_samples].ptp()
    
    # Reshape the data.
    data = data[start:end]
    data = data.reshape(spots, len(data)/spots)

    # Final array.
    result = np.zeros(1000, dtype = 'float32')
    # Get the difference between two points for each pixel.
    diff = data.ptp(axis = 1)
    # Convert to float 32 to save the data as float which is useful for
    # unpacking of the Mini-SEED data.
    diff = diff.astype('float32')

    result[empty_spots: empty_spots + spots] = diff
    # Merge the seperately treated data values.
    if samples_before_start and samples_before_start > result[0]:
        result[0] = samples_before_start
    if samples_after_end and samples_after_end > result[-1]:
        result[-1] = samples_after_end
    if lost_samples and first_spot > result[empty_spots]:
        result[empty_spots] = first_spot
    if free_samples and end_samples > result[empty_spots + spots]:
        result[empty_spots + spots] = end_samples


    # Check for masked values.
    if is_masked(result):
        result.fill_value = 0.0
        result = result.filled()

    # Create new stream.
    tr = Trace(data = result)
    stats = st[0].stats
    stats.starttime = day_starttime
    # Hard code the sampling rate for safe handling of leap seconds and
    # floating point inaccuracies. This will result in 1000 samples with the
    # last sample exactly one delta step away from the beginning of the next
    # day.
    stats.sampling_rate = 0.0115740740741
    stats.npts = 1000
    tr.stats = stats
    out_stream = Stream(traces = [tr])
    # Write the stream.
    out_stream.write (file + '_index.mseed', format = 'MSEED')

