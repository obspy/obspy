from obspy.taup.tau import plot_ray_paths
import matplotlib.pyplot as plt

ax = plt.subplot(111, polar=True)
fig = ax.figure
ax = plot_ray_paths(source_depth=100, ax=ax, fig=fig, phase_list=['P', 'PKP'],
                    npoints=25)
