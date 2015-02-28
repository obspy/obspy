from obspy.core.stream import Stream, read
from obspy.core.util import gps2DistAzimuth


host = 'http://examples.obspy.org/'
# Files (fmt: SAC)
files = ['TOK.2011.328.21.10.54.OKR01.HHN.inv',
         'TOK.2011.328.21.10.54.OKR02.HHN.inv',
         'TOK.2011.328.21.10.54.OKR03.HHN.inv',
         'TOK.2011.328.21.10.54.OKR04.HHN.inv',
         'TOK.2011.328.21.10.54.OKR05.HHN.inv',
         'TOK.2011.328.21.10.54.OKR06.HHN.inv',
         'TOK.2011.328.21.10.54.OKR07.HHN.inv',
         'TOK.2011.328.21.10.54.OKR08.HHN.inv',
         'TOK.2011.328.21.10.54.OKR09.HHN.inv',
         'TOK.2011.328.21.10.54.OKR10.HHN.inv']
# Earthquakes' epicenter
eq_lat = 35.565
eq_lon = -96.792

# Reading the waveforms
st = Stream()
for waveform in files:
    st += read(host + waveform)

# Calculating distance from SAC headers lat/lon
# (trace.stats.sac.stla and trace.stats.sac.stlo)
for tr in st:
    tr.stats.distance = gps2DistAzimuth(tr.stats.sac.stla, tr.stats.sac.stlo,
                                        eq_lat, eq_lon)[0]
    # Setting Network name for plot title
    tr.stats.network = 'TOK'

st.filter('bandpass', freqmin=0.1, freqmax=10)
# Plot
st.plot(type='section', plot_dx=20e3, recordlength=100,
        time_down=True, linewidth=.25, grid_linewidth=.25)
