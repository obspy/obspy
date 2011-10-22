from obspy.core import UTCDateTime
from obspy.signal import cornFreq2Paz
from obspy.signal.array_analysis import sonic
import pickle
import urllib

# Load data
st = pickle.load(urllib.urlopen("http://examples.obspy.org/agfa.dump"))

# Instrument correction to 1Hz corner frequency
paz1hz = cornFreq2Paz(1.0, damp=0.707)
st.simulate(paz_remove='self', paz_simulate=paz1hz)

# Execute sonic
kwargs = dict(
    # slowness grid: X min, X max, Y min, Y max, Slow Step
    sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
    # sliding window properties
    win_len=1.0, win_frac=0.05,
    # frequency properties
    frqlow=1.0, frqhigh=8.0, prewhiten=0,
    # restrict output
    semb_thres=-1e9, vel_thres=-1e9, verbose=True, timestamp='mlabhour',
    stime=UTCDateTime("20080217110515"), etime=UTCDateTime("20080217110545")
)
out = sonic(st, **kwargs)

# Plot
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

cmap = cm.hot_r
pi = np.pi

#
# make output human readable, adjust backazimuth to values between 0 and 360
t, rel_power, abs_power, baz, slow = out.T
baz[baz < 0.0] += 360

# choose number of fractions in plot (desirably 360 degree/N is an integer!)
N = 30
abins = np.arange(N + 1) * 360. / N
sbins = np.linspace(0, 3, N + 1)

# sum rel power in bins given by abins and sbins
hist, baz_edges, sl_edges = np.histogram2d(baz, slow,
        bins=[abins, sbins], weights=rel_power)

# transform to gradient
baz_edges = baz_edges / 180 * np.pi

# add polar and colorbar axes
fig = plt.figure(figsize=(8, 8))
cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)

dh = abs(sl_edges[1] - sl_edges[0])
dw = abs(baz_edges[1] - baz_edges[0])

# circle through backazimuth
for i, row in enumerate(hist):
    bars = ax.bar(left=(pi / 2 - (i + 1) * dw) * np.ones(N),
                  height=dh * np.ones(N),
                  width=dw, bottom=dh * np.arange(N),
                  color=cmap(row / hist.max()))

ax.set_xticks([pi / 2, 0, 3. / 2 * pi, pi])
ax.set_xticklabels(['N', 'E', 'S', 'W'])

# set slowness limits
ax.set_ylim(0, 3)
ColorbarBase(cax, cmap=cmap,
             norm=Normalize(vmin=hist.min(), vmax=hist.max()))

plt.show()
