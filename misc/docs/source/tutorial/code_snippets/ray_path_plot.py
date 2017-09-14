from obspy.taup.tau import raypath_plot
import matplotlib.pyplot as plt

ax = plt.subplot(111, polar=True)
fig = ax.figure
ax = raypath_plot(source_depth=100, ax=ax, fig=fig)
plt.show()
