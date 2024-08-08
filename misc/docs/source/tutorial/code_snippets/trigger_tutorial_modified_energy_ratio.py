import matplotlib.pyplot as plt

import obspy
from obspy.signal.trigger import plot_trigger, modified_energy_ratio


trace = obspy.read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = modified_energy_ratio(trace.data, int(4 * df))
fig, axes = plot_trigger(trace, cft, cft.max(), 0.1, show=False)
axes[1].set_yscale('log')
axes[1].set_ylim(ymin=0.01)
plt.show()
