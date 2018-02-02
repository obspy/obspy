from obspy.taup.tau import plot_ray_paths
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw=dict(polar=True))
ax = plot_ray_paths(source_depth=100, ax=ax, fig=fig, phase_list=['P', 'PKP'],
                    npoints=25)
