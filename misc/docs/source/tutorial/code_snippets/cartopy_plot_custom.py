import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from obspy import read_inventory, read_events

# Set up a custom projection
projection = ccrs.AlbersEqualArea(
    central_longitude=35,
    central_latitude=50,
    standard_parallels=(40, 42)
)

# Set up a figure
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection=projection)
ax.set_extent((-15., 75., 15., 80.))

# Draw standard features
ax.gridlines()
ax.coastlines()
ax.stock_img()
ax.add_feature(cfeature.BORDERS)

ax.set_title("Albers Equal Area Projection")

# Now, let's plot some data on the map
inv = read_inventory()
inv.plot(fig=fig, show=False)
cat = read_events()
cat.plot(fig=fig, show=False, title="", colorbar=False)

plt.show()
