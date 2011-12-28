from obspy.core import read
from obspy.imaging.waveform import plot_trigger
from obspy.signal.trigger import zDetect

trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = zDetect(trace.data, int(10. * df))
plot_trigger(trace, cft, -0.4, -0.3)
