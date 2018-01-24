import obspy
from obspy.signal.trigger import classic_sta_lta, plot_trigger


trace = obspy.read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = classic_sta_lta(trace.data, int(5. * df), int(10. * df))
plot_trigger(trace, cft, 1.5, 0.5)
