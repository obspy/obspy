import obspy
from obspy.signal.trigger import carl_sta_trig, plot_trigger


trace = obspy.read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = carl_sta_trig(trace.data, int(5 * df), int(10 * df), 0.8, 0.8)
plot_trigger(trace, cft, 20.0, -20.0)
