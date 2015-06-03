import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.signal.invsim import corn_freq_2_paz


paz_sts2 = {
    'poles': [-0.037004 + 0.037016j, -0.037004 - 0.037016j, -251.33 + 0j,
              - 131.04 - 467.29j, -131.04 + 467.29j],
    'zeros': [0j, 0j],
    'gain': 60077000.0,
    'sensitivity': 2516778400.0}
paz_1hz = corn_freq_2_paz(1.0, damp=0.707)  # 1Hz instrument
paz_1hz['sensitivity'] = 1.0

st = obspy.read()
# make a copy to keep our original data
st_orig = st.copy()

# Simulate instrument given poles, zeros and gain of
# the original and desired instrument
st.simulate(paz_remove=paz_sts2, paz_simulate=paz_1hz)


tr = st[0]
tr_orig = st_orig[0]

t = np.arange(tr.stats.npts) / tr.stats.sampling_rate

plt.subplot(211)
plt.plot(t, tr_orig.data, 'k')
plt.ylabel('STS-2 [counts]')
plt.subplot(212)
plt.plot(t, tr.data, 'k')
plt.ylabel('1Hz Instrument [m/s]')
plt.xlabel('Time [s]')
plt.show()
