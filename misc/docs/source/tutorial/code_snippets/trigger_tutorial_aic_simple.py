"""
Credits: mbagagli
"""

from obspy.core import read, UTCDateTime
from obspy.signal.trigger import aic_simple


trace = read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate
trace_s = trace.slice(UTCDateTime('1970-01-01T01:00:31.6'),
                      UTCDateTime('1970-01-01T01:00:34.3'))
aic_f = aic_simple(trace_s.data)
p_idx = aic_f.argmin()
print(p_idx / df)
print(UTCDateTime('1970-01-01T01:00:31.6') + (p_idx / df))
