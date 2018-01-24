import obspy
from obspy.signal.trigger import plot_trigger, z_detect


trace = obspy.read("https://examples.obspy.org/ev0_6.a01.gse2")[0]
df = trace.stats.sampling_rate

cft = z_detect(trace.data, int(10. * df))
plot_trigger(trace, cft, -0.4, -0.3)
