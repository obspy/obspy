import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import Beach

m = Basemap(projection='moll', lon_0=142.36929, lat_0=38.3215,
            resolution='c')

m.drawcoastlines()
m.fillcontinents()
m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,420.,60.))
m.drawmapboundary()

x, y = m(142.36929, 38.3215)
focmecs = [0.136, -0.591, 0.455, -0.396, 0.046, -0.615]

ax = plt.gca()
b = Beach(focmecs, xy=(x, y), width=1e6, linewidth = 0.1, alpha=0.85)
b.set_zorder(10)
ax.add_collection(b)
plt.show()
