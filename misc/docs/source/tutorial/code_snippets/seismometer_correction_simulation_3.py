from obspy.fdsn import Client
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

# MW 7.1 Darfield earthquake, New Zealand
t1 = UTCDateTime("2010-09-3T16:30:00.000")
t2 = UTCDateTime("2010-09-3T17:00:00.000")

# Fetch waveform from IRIS FDSN web service into a ObsPy stream object
fdsn_client = Client('IRIS')
st = fdsn_client.get_waveforms(network='NZ', station='BFZ', location='10',
                               channel='HHZ', starttime=t1, endtime=t2)

# Fetch RESP information from IRIS FDSN web service
inv = fdsn_client.get_stations(network='NZ', station='BFZ', location='10',
                               channel='HHZ', starttime=t1, endtime=t2,
                               # specify level to include response information
                               level='response')

# make a copy to keep our original data
st_orig = st.copy()

# attach response data to available traces at trace.stats.response
st.attach_response(inv)

# define a filter band to prevent amplifying noise during the deconvolution
pre_filt = (0.005, 0.006, 30.0, 35.0)

# remove response from all traces that include stats.response
st.remove_response(output='DISP',  # Units for output (also: VEL or ACC)
                   pre_filt=pre_filt)

# plot original and simulated data
tr = st[0]
tr_orig = st_orig[0]
time = np.arange(tr.stats.npts) / tr.stats.sampling_rate

plt.subplot(211)
plt.plot(time, tr_orig.data, 'k')
plt.ylabel('STS-2 [counts]')
plt.subplot(212)
plt.plot(time, tr.data, 'k')
plt.ylabel('Displacement [m]')
plt.xlabel('Time [s]')
plt.show()
