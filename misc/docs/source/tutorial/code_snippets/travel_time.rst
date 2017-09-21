========================
Travel Time Calculations
========================

Travel Time Plot
----------------

The following lines show how to create a simple travel time plot for a given
distance range, selected phases and the ``iasp91`` velocity model using the
:func:`~obspy.taup.plot_travel_times` function of the module
:class:`obspy.taup`.

.. plot:: tutorial/code_snippets/plot_travel_times.py
   :include-source:

Cartesian Ray Paths
-------------------

The following lines show how to create a simple plot of ray paths for a given
distance, phase(s), and the ``iasp91`` velocity model on a Cartesian map,
using the :func:`~obspy.taup.tau.Arrivals.plot_rays` function of the class
:class:`obspy.taup.tau.Arrivals`.

.. plot:: tutorial/code_snippets/travel_time_cartesian_raypath.py
   :include-source:

Spherical Ray Paths
-------------------

The following lines show how to create a simple plot of ray paths for a given
distance, phase(s), and the ``iasp91`` velocity model on a spherical map,
using the :func:`~obspy.taup.tau.Arrivals.plot_rays` function of the class
:class:`obspy.taup.tau.Arrivals`.

.. plot:: tutorial/code_snippets/travel_time_spherical_raypath.py
   :include-source:

Body Wave Ray Paths
-------------------

The following lines show how to create a plot of ray paths for several
distances, phases and the ``iasp91`` velocity model using the
:func:`~obspy.taup.tau.Arrivals.plot_ray_paths` function of the class
:class:`obspy.taup.tau.Arrivals`. For examples with rays for one
distance, try one of the plots in the preceding sections.

.. plot:: tutorial/code_snippets/plot_ray_paths.py
   :include-source:
