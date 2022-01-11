import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from obspy import read_events
from obspy.imaging.beachball import beach


event = read_events(
    'https://earthquake.usgs.gov/product/moment-tensor/'
    'us_20005ysu_mww/us/1470868224040/quakeml.xml', format='QUAKEML')[0]
origin = event.preferred_origin() or event.origins[0]
focmec = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
tensor = focmec.moment_tensor.tensor
moment_list = [tensor.m_rr, tensor.m_tt, tensor.m_pp,
               tensor.m_rt, tensor.m_rp, tensor.m_tp]

projection = ccrs.PlateCarree(central_longitude=0.0)
x, y = projection.transform_point(x=origin.longitude, y=origin.latitude,
                                  src_crs=ccrs.Geodetic())

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection=projection)
ax.set_extent((-180, 180, -90, 90))
ax.coastlines()
ax.gridlines()

b = beach(moment_list, xy=(x, y), width=20, linewidth=1, alpha=0.85, zorder=10)
b.set_zorder(10)
ax.add_collection(b)

fig.show()
