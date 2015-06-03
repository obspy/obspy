import obspy
from obspy.signal.trigger import delayed_STALTA, plot_trigger


trace = obspy.read("http://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = delayed_STALTA(trace.data, int(5 * df), int(10 * df))
plot_trigger(trace, cft, 5, 10)
