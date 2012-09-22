from obspy.core import UTCDateTime
from obspy.core.util.geodetics import gps2DistAzimuth
from obspy.arclink import Client
from math import log10
from numpy import median

paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
          'poles': [-6.2832-4.7124j, -6.2832+4.7124j]}

client = Client(user="sed-workshop@obspy.org")
t = UTCDateTime("2012-04-03T02:45:03")

stations = client.getStations(t, t + 300, "CH")
mags = []

for station in stations:
    station = station['code']
    try:
        st = client.getWaveform("CH", station, "", "[EH]H[ZNE]", t - 300, t + 300, metadata=True)
        assert(len(st) == 3)
    except:
        print station, "---"
        continue

    st.simulate(paz_remove="self", paz_simulate=paz_wa, water_level=10)
    st.trim(t, t + 50)

    tr_n = st.select(component="N")[0]
    ampl_n = max(abs(tr_n.data))
    tr_e = st.select(component="E")[0]
    ampl_e = max(abs(tr_e.data))
    ampl = max(ampl_n, ampl_e)

    sta_lat = st[0].stats.coordinates.latitude
    sta_lon = st[0].stats.coordinates.longitude
    event_lat = 46.218
    event_lon = 7.706

    epi_dist, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
    epi_dist = epi_dist / 1000

    if epi_dist < 60:
        a = 0.018
        b = 2.17
    else:
        a = 0.0038
        b = 3.02
    ml = log10(ampl * 1000) + a * epi_dist + b
    print station, ml
    mags.append(ml)

net_mag = median(mags)
print "Network magnitude:", net_mag
