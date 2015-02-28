import numpy as np
import matplotlib.pyplot as plt

from obspy.arclink import Client
from obspy.core import UTCDateTime
from obspy.signal import cornFreq2Paz, seisSim


# Retrieve data via ArcLink
# please provide a valid email address for the keyword user
client = Client(user="test@obspy.de")
t = UTCDateTime("2009-08-24 00:20:03")
st = client.getWaveform('BW', 'RJOB', '', 'EHZ', t, t + 30)
paz = client.getPAZ('BW', 'RJOB', '', 'EHZ', t)

# 1Hz instrument
one_hertz = cornFreq2Paz(1.0)
# Correct for frequency response of the instrument
res = seisSim(st[0].data.astype('float32'),
              st[0].stats.sampling_rate,
              paz,
              inst_sim=one_hertz)
# Correct for overall sensitivity
res = res / paz['sensitivity']

# Plot the seismograms
sec = np.arange(len(res)) / st[0].stats.sampling_rate
plt.subplot(211)
plt.plot(sec, st[0].data, 'k')
plt.title("%s %s" % (st[0].stats.station, t))
plt.ylabel('STS-2')
plt.subplot(212)
plt.plot(sec, res, 'k')
plt.xlabel('Time [s]')
plt.ylabel('1Hz CornerFrequency')
plt.show()
