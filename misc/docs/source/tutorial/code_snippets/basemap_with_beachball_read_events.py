import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from obspy import read_events
from obspy.imaging.beachball import beach


event = read_events(
    'https://earthquake.usgs.gov/archive/product/moment-tensor/'
    'us_20005ysu_mww/us/1470868224040/quakeml.xml', format='QUAKEML')[0]
origin = event.preferred_origin() or event.origins[0]
focmec = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
tensor = focmec.moment_tensor.tensor
moment_list = [tensor.m_rr, tensor.m_tt, tensor.m_pp,
               tensor.m_rt, tensor.m_rp, tensor.m_tp]

m = Basemap(projection='cyl', lon_0=origin.longitude, lat_0=origin.latitude,
            resolution='c')

m.drawcoastlines()
m.fillcontinents()
m.drawparallels(np.arange(-90., 120., 30.))
m.drawmeridians(np.arange(0., 420., 60.))
m.drawmapboundary()

x, y = m(origin.longitude, origin.latitude)

ax = plt.gca()
b = beach(moment_list, xy=(x, y), width=20, linewidth=1, alpha=0.85)
b.set_zorder(10)
ax.add_collection(b)
plt.show()
