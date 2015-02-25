from obspy.core import read
from obspy.signal.trigger import carlSTATrig, plotTrigger


trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = carlSTATrig(trace.data, int(5 * df), int(10 * df), 0.8, 0.8)
plotTrigger(trace, cft, 20.0, -20.0)
