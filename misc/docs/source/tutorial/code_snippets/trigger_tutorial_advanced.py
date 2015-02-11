import matplotlib.pyplot as plt

from obspy.arclink import Client
from obspy.core import UTCDateTime
from obspy.signal.trigger import recSTALTA, triggerOnset


# Retrieve waveforms via ArcLink
client = Client(host="erde.geophysik.uni-muenchen.de", port=18001,
                user="test@obspy.de")
t = UTCDateTime("2009-08-24 00:19:45")
st = client.getWaveform('BW', 'RTSH', '', 'EHZ', t, t + 50)

# For convenience
tr = st[0]  # only one trace in mseed volume
df = tr.stats.sampling_rate

# Characteristic function and trigger onsets
cft = recSTALTA(tr.data, int(2.5 * df), int(10. * df))
on_of = triggerOnset(cft, 3.5, 0.5)

# Plotting the results
ax = plt.subplot(211)
plt.plot(tr.data, 'k')
ymin, ymax = ax.get_ylim()
plt.vlines(on_of[:, 0], ymin, ymax, color='r', linewidth=2)
plt.vlines(on_of[:, 1], ymin, ymax, color='b', linewidth=2)
plt.subplot(212, sharex=ax)
plt.plot(cft, 'k')
plt.hlines([3.5, 0.5], 0, len(cft), color=['r', 'b'], linestyle='--')
plt.axis('tight')
plt.show()
