from obspy.iris import Client
from obspy.core import UTCDateTime
from obspy.core.util import NamedTemporaryFile
import matplotlib.pyplot as plt
import numpy as np

# MW 7.1 Darfield earthquake, New Zealand
t1 = UTCDateTime("2010-09-3T16:30:00.000")
t2 = UTCDateTime("2010-09-3T17:00:00.000")

# Fetch waveform from IRIS web service into a ObsPy stream object
client = Client()
st = client.getWaveform('NZ', 'BFZ', '10', 'HHZ', t1, t2)

# Download and save instrument response file into a temporary file
with NamedTemporaryFile() as tf:
    respf = tf.name
    client.saveResponse(respf, 'NZ', 'BFZ', '10', 'HHZ', t1, t2, format="RESP")

    # make a copy to keep our original data
    st_orig = st.copy()

    # define a filter band to prevent amplifying noise during the deconvolution
    pre_filt = (0.005, 0.006, 30.0, 35.0)

    # this can be the date of your raw data or any date for which the
    # SEED RESP-file is valid
    date = t1

    seedresp = {'filename': respf,  # RESP filename
                # when using Trace/Stream.simulate() the "date" parameter can
                # also be omitted, and the starttime of the trace is then used.
                'date': date,
                # Units to return response in ('DIS', 'VEL' or ACC)
                'units': 'DIS'
                }

    # Remove instrument response using the information from the given RESP file
    st.simulate(paz_remove=None, pre_filt=pre_filt, seedresp=seedresp)

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
