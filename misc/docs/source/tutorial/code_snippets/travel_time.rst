========================
Travel Time Calculations
========================

Travel Time Plot
----------------

The following lines show how to create a simple travel time plot for a given
distance range, selected phases and the ``iasp91`` velocity model using the
:func:`~obspy.taup.taup.travelTimePlot` function of the module
:class:`obspy.taup`.

.. plot:: tutorial/code_snippets/travel_time_plot.py
   :include-source:

Cartesian Ray Paths
-------------------

The following lines show how to create a simple plot of ray paths for a given
distance range, phase, and the ``iasp91`` velocity model on a Cartesian map,
using the :func:`~obspy.taup.tau.Arrivals.plot` function of the class
:class:`obspy.taup.tau.Arrivals`.

.. plot:: tutorial/code_snippets/travel_time_cartesian_raypath.py
   :include-source:

Spherical Ray Paths
-------------------

The following lines show how to create a simple plot of ray paths for a given
distance range, phase, and the ``iasp91`` velocity model on a spherical map,
using the :func:`~obspy.taup.tau.Arrivals.plot` function of the class
:class:`obspy.taup.tau.Arrivals`.

.. plot:: tutorial/code_snippets/travel_time_spherical_raypath.py
   :include-source:

Body Wave Ray Paths
-------------------

The following lines show how to create a large set of body wave ray paths for
several distance, phases and the ``iasp91`` velocity model using the
:func:`~obspy.taup.tau.Arrivals.plot` function of the class
:class:`obspy.taup.tau.Arrivals`. For simpler examples, try one of the plots
in the preceeding sections.

.. plot:: tutorial/code_snippets/travel_time_body_waves.py
   :include-source:
