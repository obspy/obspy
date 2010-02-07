from obspy.core import read, Stream, Trace
from glob import iglob
import numpy as np
from numpy.ma import is_masked

folder =\
'/Users/lion/Documents/workspace/TestFiles/archive/RJOB/EHE.D/output/*_index.mseed'

st = read(folder)
#XXX: This fix is just for wrong index files. Remove the next time around.
for trace in st:
    trace.stats.sampling_rate = 1000.0/(24*60*60)

st.merge()

# Set masked arrays to zero.
if is_masked(st[0].data):
    st[0].data.fill_value = 0.0
    st[0].data = st[0].data.filled()

st.write('BW.RJOB..EHE.2009.index', format = 'MSEED')
