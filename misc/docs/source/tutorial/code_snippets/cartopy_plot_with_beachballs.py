import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import gzip
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from obspy.imaging.beachball import beach


# read in topo data (on a regular lat/lon grid)
# (SRTM data from: http://srtm.csi.cgiar.org/)
with gzip.open("srtm_1240-1300E_4740-4750N.asc.gz") as fp:
    srtm = np.loadtxt(fp, skiprows=8)

# origin of data grid as stated in SRTM data file header
# create arrays with all lon/lat values from min to max and
lats = np.linspace(47.8333, 47.6666, srtm.shape[0])
lons = np.linspace(12.6666, 13.0000, srtm.shape[1])

# Prepare figure and Axis object with a proper projection and extent
projection = ccrs.PlateCarree()
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection=projection)
ax.set_extent((12.75, 12.95, 47.69, 47.81))

# create grids and compute map projection coordinates for lon/lat grid
grid = projection.transform_points(ccrs.Geodetic(),
                                   *np.meshgrid(lons, lats),  # unpacked x, y
                                   srtm)  # z from topo data

# Make contour plot
ax.contour(*grid.T, transform=projection, levels=30)

# Draw country borders with red
ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='red')

# Draw a lon/lat grid and format the labels
gl = ax.gridlines(crs=projection, draw_labels=True)

gl.xlocator = mticker.FixedLocator([12.75, 12.8, 12.85, 12.9, 12.95])
gl.ylocator = mticker.FixedLocator([47.7, 47.75, 47.8])
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.ylabel_style = {'size': 13, 'color': 'gray'}
gl.xlabel_style = {'weight': 'bold'}


# Plot station positions and names into the map
# again we have to compute the projection of our lon/lat values
lats = np.array([47.761659, 47.7405, 47.755100, 47.737167])
lons = np.array([12.864466, 12.8671, 12.849660, 12.795714])
names = [" RMOA", " RNON", " RTSH", " RJOB"]
x, y, _ = projection.transform_points(ccrs.Geodetic(), lons, lats).T
ax.scatter(x, y, 200, color="r", marker="v", edgecolor="k", zorder=3)
for i in range(len(names)):
    plt.text(x[i], y[i], names[i], va="top", family="monospace", weight="bold")

# Add beachballs for two events
lats = np.array([47.751602, 47.75577])
lons = np.array([12.866492, 12.893850])
points = projection.transform_points(ccrs.Geodetic(), lons, lats)

# Two focal mechanisms for beachball routine, specified as [strike, dip, rake]
focmecs = [[80, 50, 80], [85, 30, 90]]
for i in range(len(focmecs)):
    b = beach(focmecs[i], xy=(points[i][0], points[i][1]),
              width=0.008, linewidth=1, zorder=10)
    ax.add_collection(b)

plt.show()
