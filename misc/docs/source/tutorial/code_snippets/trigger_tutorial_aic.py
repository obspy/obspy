from obspy.core import read, UTCDateTime
from obspy.signal.trigger import aic
trace = read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate
trace_s = trace.slice(UTCDateTime('1970-01-01T01:00:31.6'),
                      UTCDateTime('1970-01-01T01:00:34.3'))
p_idx, aic_f = aic(trace_s.data)
print(p_idx / df)
print(UTCDateTime('1970-01-01T01:00:31.6') + (p_idx / df))
