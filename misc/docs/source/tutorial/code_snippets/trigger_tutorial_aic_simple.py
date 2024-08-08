"""
Credits: mbagagli, Danylo Ulianych
"""
import matplotlib.pyplot as plt

from obspy.core import read, UTCDateTime
from obspy.signal.trigger import aic_simple
from obspy.signal.trigger import plot_trace

trace = read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate
trace_s = trace.slice(UTCDateTime('1970-01-01T01:00:31.6'),
                      UTCDateTime('1970-01-01T01:00:34.3'))
aic_f = aic_simple(trace_s.data)
trigger_onset_idx = aic_f.argmin()
trigger_onset_start_sec = trigger_onset_idx / df
print("Trigger onset in seconds:", trigger_onset_start_sec)
print(UTCDateTime('1970-01-01T01:00:31.6') + trigger_onset_start_sec)
fig, axes = plot_trace(trace_s, aic_f)
axes[0].vlines(trigger_onset_start_sec, *axes[0].get_ylim(), color='r', lw=2,
               label="Trigger On")
axes[0].legend()
axes[1].set_ylabel('AIC')
axes[1].scatter(trigger_onset_start_sec, aic_f.min(), color='r')
plt.show()
