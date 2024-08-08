from __future__ import print_function

import numpy as np
from obspy import UTCDateTime, read, Trace, Stream


weather = """
00.0000 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000000
00.0002 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000001
00.0005 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000002
00.0008 0.0 ??? 4.7 97.7 1015.4 0.0 010308 000003
00.0011 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000004
00.0013 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000005
00.0016 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000006
00.0019 0.0 ??? 4.7 97.7 1015.0 0.0 010308 000007
"""

# Convert to NumPy character array
data = np.fromstring(weather, dtype='|S1')

# Fill header attributes
stats = {'network': 'BW', 'station': 'RJOB', 'location': '',
         'channel': 'WLZ', 'npts': len(data), 'sampling_rate': 0.1,
         'mseed': {'dataquality': 'D'}}
# set current time
stats['starttime'] = UTCDateTime()
st = Stream([Trace(data=data, header=stats)])
# write as ASCII file (encoding=0)
st.write("weather.mseed", format='MSEED', encoding=0, reclen=256)

# Show that it worked, convert NumPy character array back to string
st1 = read("weather.mseed")
print(st1[0].data.tobytes())
