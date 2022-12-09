from __future__ import print_function

from math import log10

import numpy as np

from obspy.clients.arclink import Client
from obspy import Stream, UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import coincidence_trigger


client = Client(user="sed-workshop@obspy.org")

t = UTCDateTime("2012-04-03T01:00:00")
t2 = t + 4 * 3600

stations = ["AIGLE", "SENIN", "DIX", "LAUCH", "MMK", "SIMPL"]
st = Stream()

for station in stations:
    try:
        tmp = client.get_waveforms("CH", station, "", "[EH]HZ", t, t2,
                                   metadata=True)
    except Exception:
        print(station, "---")
        continue
    st += tmp

st.taper()
st.filter("bandpass", freqmin=1, freqmax=20)
triglist = coincidence_trigger("recstalta", 10, 2, st, 4, sta=0.5, lta=10)
print(len(triglist), "events triggered.")

for trig in triglist:
    closest_sta = trig['stations'][0]
    tr = st.select(station=closest_sta)[0]
    trig['latitude'] = tr.stats.coordinates.latitude
    trig['longitude'] = tr.stats.coordinates.longitude

paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
          'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

for trig in triglist:
    t = trig['time']
    print("#" * 80)
    print("Trigger time:", t)
    mags = []

    stations = client.get_stations(t, t + 300, "CH")

    for station in stations:
        station = station['code']
        try:
            st = client.get_waveforms("CH", station, "", "[EH]H[ZNE]", t - 300,
                                      t + 300, metadata=True)
            assert len(st) == 3
        except Exception:
            print(station, "---")
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
        event_lat = trig['latitude']
        event_lon = trig['longitude']

        epi_dist, az, baz = gps2dist_azimuth(event_lat, event_lon, sta_lat,
                                             sta_lon)
        epi_dist = epi_dist / 1000

        if epi_dist < 60:
            a = 0.018
            b = 2.17
        else:
            a = 0.0038
            b = 3.02
        ml = log10(ampl * 1000) + a * epi_dist + b
        print(station, ml)
        mags.append(ml)

    net_mag = np.median(mags)
    print("Network magnitude:", net_mag)
