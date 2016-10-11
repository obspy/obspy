=====================
Plotting Spectrograms
=====================

The following lines of code demonstrate how to make a spectrogram plot of an
ObsPy :class:`~obspy.core.stream.Stream` object.

Lots of options can be customized, see
:func:`~obspy.imaging.spectrogram.spectrogram` for more details. For
example, the colormap of the plot can easily be adjusted by importing a
predefined colormap from :mod:`matplotlib.cm`, nice overviews of available
matplotlib colormaps are given at:

* http://www.physics.ox.ac.uk/users/msshin/science/code/matplotlib_cm/
* http://matplotlib.org/examples/color/colormaps_reference.html
* https://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps

.. plot:: tutorial/code_snippets/plotting_spectrograms.py
   :include-source:
