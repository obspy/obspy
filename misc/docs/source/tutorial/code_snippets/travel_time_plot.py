from obspy.taup import traveltime_plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = traveltime_plot(source_depth=10, ax=ax, fig=fig)
plt.show()
