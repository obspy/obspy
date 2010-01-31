from obspy.core import read, Stream, Trace
from glob import iglob

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
    # Get the data.
    data = st[0].data
    # Reshape to 1000.
    border = int(len(data)//1000) * 1000
    data = data[0: border]
    rest = st[0].data[border:]
    # Reshape the data.
    data = data.reshape((1000, len(data)/1000))
    # Get the difference between two points for each pixel.
    diff = data.ptp(axis = 1)
    # Convert to float 32 to save the data as float which is useful for
    # unpacking of the Mini-SEED data.
    diff = diff.astype('float32')
    # Read the rest too.
    if rest.ptp() > diff[-1]:
        diff[-1] = rest.ptp()
    # Create new stream.
    tr = Trace(data = diff)
    stats = st[0].stats
    stats.sampling_rate = 1.0/((stats.endtime - stats.starttime)/999.0)
    stats.npts = 1000
    tr.stats = stats
    out_stream = Stream(traces = [tr])
    # Write the stream.
    out_stream.write (file + '_index.mseed', format = 'MSEED')

