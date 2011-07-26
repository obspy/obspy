=====================
Plotting Spectrograms
=====================

The following lines of code demonstrate how to make a spectrogram plot of an
Obspy :py:class:`~obspy.core.stream.Stream` object.

Lots of options can be customized, see
:py:func:`~obspy.imaging.spectrogram.spectrogram` for more details. For
example, the colormap of the plot can easily be adjusted by importing a
predefined colormap from :py:mod:`matplotlib.cm`, nice overviews of available
matplotlib colormaps are given at:

* http://www.astro.lsa.umich.edu/~msshin/science/code/matplotlib_cm/
* http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps

.. include:: plotting_spectrograms.py
   :literal:

.. plot:: source/tutorial/plotting_spectrograms.py
