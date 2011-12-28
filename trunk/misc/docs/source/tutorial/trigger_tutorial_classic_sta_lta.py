from obspy.core import read
from obspy.imaging.waveform import plot_trigger
from obspy.signal.trigger import classicSTALTA

trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = classicSTALTA(trace.data, int(5. * df), int(10. * df))
plot_trigger(trace, cft, 1.5, 0.5)
