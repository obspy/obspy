from obspy.core import read
from obspy.imaging.waveform import plot_trigger
from obspy.signal.trigger import *

trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = carlStaTrig(trace.data, int(5 * df), int(10 * df), 0.8, 0.8)
plot_trigger(trace, cft, 20.0, -20.0)
