from obspy.core import read
from obspy.signal.trigger import plotTrigger, zDetect


trace = read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = zDetect(trace.data, int(10. * df))
plotTrigger(trace, cft, -0.4, -0.3)
