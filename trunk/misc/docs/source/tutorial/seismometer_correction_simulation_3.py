from obspy.iris import Client
from obspy.core import UTCDateTime, read
import matplotlib.pyplot as plt
import numpy as np

rawf = 'BFZ.HHZ.mseed'
respf = 'RESP.NZ.BFZ.10.HHZ'

# MW 7.1 Darfield earthquake, New Zealand
t1 = UTCDateTime("2010-09-3T16:30:00.000")
t2 = UTCDateTime("2010-09-3T17:00:00.000")

# Download and save waveform
client = Client()
client.saveWaveform(rawf, 'NZ', 'BFZ', '10', 'HHZ', t1, t2)

# Download and save instrument response file
client.saveResponse(respf, 'NZ', 'BFZ', '10', 'HHZ',t1, t2, format="RESP")

# read in your raw data
st = read(rawf)

# make a copy to keep our original data
st_orig = st.copy()

# define a filter band to 
# prevent amplifying noise
# during the deconvolution
fl1 = 0.005
fl2 = 0.006
fl3 = 30.
fl4 = 35.

# this can be the date of your raw data or
# any date for which the SEED Resp-file is valid
date = t1

seedresp = {'filename':respf, # RESP filename
            'date':date,
            'units':'VEL' #Units to return response in ('DIS', 'VEL' or ACC)
            }

# Remove instrument response using the information from the given RESP file
st.simulate(paz_remove=None,remove_sensitivity=False,pre_filt=(fl1,fl2,fl3,fl4),
            seedresp=seedresp)

# plot original and simulated data
tr = st[0]
tr_orig = st_orig[0]
time = np.arange(tr.stats.npts) / tr.stats.sampling_rate

plt.subplot(211)
plt.plot(time, tr_orig.data, 'k')
plt.ylabel('STS-2 [counts]')
plt.subplot(212)
plt.plot(time, tr.data, 'k')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [s]')
plt.show()
