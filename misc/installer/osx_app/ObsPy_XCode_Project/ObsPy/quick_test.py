import matplotlib.pyplot as plt
import obspy
import numpy as np
import mlpy
from mpl_toolkits.basemap import Basemap

tr = obspy.read()[0]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

ax1.plot(tr.data, 'k')
ax1.set_title("Waveform Plot")

omega0 = 8
wavelet_fct = "morlet"
scales = mlpy.wavelet.autoscales(N=len(tr.data), dt=tr.stats.delta, dj=0.05,
                                 wf=wavelet_fct, p=omega0)
spec = mlpy.wavelet.cwt(tr.data, dt=tr.stats.delta, scales=scales,
                        wf=wavelet_fct, p=omega0)
# approximate scales through frequencies
freq = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales[1:])

t = np.arange(tr.stats.npts) / tr.stats.sampling_rate

img = ax2.imshow(np.abs(spec), extent=[t[0], t[-1], freq[-1], freq[0]],
                 aspect='auto', interpolation="nearest")
ax2.set_yscale('log')
ax2.set_title("mlpy Wavelet Transform")


# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.
map = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='c', ax=ax3)
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.fillcontinents(color='coral',lake_color='aqua')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='aqua')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
# make up some data on a regular lat/lon grid.
nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
lons = (delta*np.indices((nlats,nlons))[1,:,:])
wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
# compute native map projection coordinates of lat/lon grid.
x, y = map(lons*180./np.pi, lats*180./np.pi)
# contour data over the map.
cs = map.contour(x,y,wave+mean,15,linewidths=1.5)
ax3.set_title('Basemap plot')

from pandas import Series, date_range

ts = Series(np.random.randn(1000), index=date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot(ax=ax4)
ax4.set_title("Pandas Plot")

plt.tight_layout()
plt.show()