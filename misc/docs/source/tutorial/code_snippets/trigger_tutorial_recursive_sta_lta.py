from obspy.core import read
from obspy.signal.trigger import plotTrigger, recSTALTA


trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = recSTALTA(trace.data, int(5 * df), int(10 * df))
plotTrigger(trace, cft, 1.2, 0.5)
