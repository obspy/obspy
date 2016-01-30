import obspy
from obspy.signal.trigger import plot_trigger, recursive_STALTA


trace = obspy.read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = recursive_STALTA(trace.data, int(5 * df), int(10 * df))
plot_trigger(trace, cft, 1.2, 0.5)
