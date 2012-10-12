from obspy.core import read
from obspy.core.util.geodetics import gps2DistAzimuth
from math import log10

st = read("../data/LKBD_WA_CUT.MSEED")

tr_n = st.select(component="N")[0]
ampl_n = max(abs(tr_n.data))
tr_e = st.select(component="E")[0]
ampl_e = max(abs(tr_e.data))
ampl = max(ampl_n, ampl_e)

sta_lat = 46.38703
sta_lon = 7.62714
event_lat = 46.218
event_lon = 7.706

epi_dist, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
epi_dist = epi_dist / 1000

a = 0.018
b = 2.17
ml = log10(ampl * 1000) + a * epi_dist + b
print ml
