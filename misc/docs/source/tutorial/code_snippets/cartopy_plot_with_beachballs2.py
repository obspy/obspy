import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from obspy.imaging.beachball import beach


projection = ccrs.PlateCarree(central_longitude=142.0)

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection=projection)
ax.set_extent((-180, 180, -90, 90))
ax.coastlines()
ax.gridlines()

x, y = projection.transform_point(x=142.36929, y=38.3215,
                                  src_crs=ccrs.Geodetic())
focmecs = [0.136, -0.591, 0.455, -0.396, 0.046, -0.615]

ax = plt.gca()
b = beach(focmecs, xy=(x, y), width=10, linewidth=1, alpha=0.85)
b.set_zorder(10)
ax.add_collection(b)

plt.show()
