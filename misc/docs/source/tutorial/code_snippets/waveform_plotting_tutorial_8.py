from obspy import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from obspy import read, Stream
from obspy.geodetics import gps2dist_azimuth


host = 'https://examples.obspy.org/'
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

# Calculating distance from SAC headers lat/lon and add custom TimeMarks
# (trace.stats.sac.stla and trace.stats.sac.stlo)
for tr in st:
    tr.stats.distance = gps2dist_azimuth(tr.stats.sac.stla, tr.stats.sac.stlo,
                                         eq_lat, eq_lon)[0]
    # Setting Network name for plot title
    tr.stats.network = 'TOK'

    # Time Marks creation:
    tr.stats.timemarks = [
               (UTCDateTime("2011-11-24T21:11:07.0"),
                {'color': 'sandybrown', 'markeredgewidth': 3}),
       ]

    if tr.stats.station == "OKR01":
        tr.stats.timemarks.append(
            (UTCDateTime("2011-11-24T21:11:09.39"),
             {'markeredgewidth': 2, 'markersize': 40}))
    elif tr.stats.station == "OKR02":
        tr.stats.timemarks.append(
            (UTCDateTime("2011-11-24T21:11:10.48"),
             {'markeredgewidth': 2, 'markersize': 40}))
    elif tr.stats.station == "OKR03":
        tr.stats.timemarks.append(
            (UTCDateTime("2011-11-24T21:11:12.5"),
             {'markeredgewidth': 2, 'markersize': 40}))
#
st.filter('bandpass', freqmin=0.1, freqmax=10)


# ========================= Section
# Do the section plot..
# If no customization is done after the section plot command, figure
# initialization can be left out and also option ".., show=False, fig=fig)" can
# be omitted, and figure is shown automatically

fig1 = plt.figure()
st.plot(type='section', plot_dx=20e3, recordlength=100, offset_max=60000,
        time_down=True, linewidth=.25, grid_linewidth=.25, fig=fig1,
        plot_time_marks=True)

# Plot customization: Add station labels to offset axis
ax = fig1.axes[0]
transform = blended_transform_factory(ax.transData, ax.transAxes)
for tr in st:
    ax.text(tr.stats.distance / 1e3, 1.0, tr.stats.station, rotation=270,
            va="bottom", ha="center", transform=transform, zorder=10)

# ========================= Standard (normal/relative)
fig2 = plt.figure()
st[0:3].plot(type='normal', show=False, plot_time_marks=True, fig=fig2)

fig3 = plt.figure()
st[0:3].plot(type='relative', show=False, plot_time_marks=True, fig=fig3,
             reftime=UTCDateTime("2011-11-24T21:11:9.39")-5.0)

plt.show()
