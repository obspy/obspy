=================================
Travel Time and Ray Path Plotting
=================================

Travel Time Plot
----------------

The following lines show how to use the convenience wrapper function
:func:`~obspy.taup.tau.plot_travel_times` to plot the travel times for a
given distance range and selected phases, calculated with the
``iasp91`` velocity model.

.. plot:: tutorial/code_snippets/plot_travel_times.py
   :include-source:

Cartesian Ray Paths
-------------------

The following lines show how to plot the ray paths for a given
distance, and phase(s). The ray paths are calculated with the ``iasp91``
velocity model, and plotted on a Cartesian map, using the
:func:`~obspy.taup.tau.Arrivals.plot_rays` method of the class
:class:`obspy.taup.tau.Arrivals`.

.. plot:: tutorial/code_snippets/travel_time_cartesian_raypath.py
   :include-source:

Spherical Ray Paths
-------------------

The following lines show how to plot the ray paths for a given
distance, and phase(s). The ray paths are calculated with the
``iasp91`` velocity model, and plotted on a spherical map, using the
:func:`~obspy.taup.tau.Arrivals.plot_rays` method of the class
:class:`obspy.taup.tau.Arrivals`.

.. plot:: tutorial/code_snippets/travel_time_spherical_raypath.py
   :include-source:

Ray Paths for Multiple Distances
--------------------------------

The following lines plot the ray paths for several epicentral
distances, and phases. The rays are calculated with the ``iasp91``
velocity model, and the plot is made using the convenience wrapper
function :func:`~obspy.taup.tau.plot_ray_paths`.

.. plot:: tutorial/code_snippets/plot_ray_paths.py
   :include-source:

For examples with rays for a single epicentral distance, try the
:func:`~obspy.taup.tau.Arrivals.plot_rays` method in the previous
section. The following is a more advanced example with a custom list of phases
and distances:

.. plot:: tutorial/code_snippets/travel_time_body_waves.py
   :include-source:
