import numpy as np
import matplotlib.pyplot as plt
from obspy.core import read
from obspy.signal.tf_misfit import cwt

st = read()
tr = st[0]
npts = tr.stats.npts
dt = tr.stats.delta
t = np.linspace(0, dt * npts, npts)
f_min = 1
f_max = 50

scalogram = cwt(tr.data, dt, 8, f_min, f_max)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.abs(scalogram)[-1::-1], extent=[t[0], t[-1], f_min, f_max],
          aspect='auto', interpolation="nearest")
ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
ax.set_ylabel("Frequency [Hz]")
ax.set_yscale('log')
plt.show()
