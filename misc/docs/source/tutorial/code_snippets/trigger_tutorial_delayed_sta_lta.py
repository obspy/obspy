from obspy.core import read
from obspy.signal.trigger import delayedSTALTA, plotTrigger


trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = delayedSTALTA(trace.data, int(5 * df), int(10 * df))
plotTrigger(trace, cft, 5, 10)
