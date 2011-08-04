==========================
Waveform Plotting Tutorial
==========================

Read the files as shown at the :ref:`reading-seismogramms` page. We will use
two different ObsPy :class:`~obspy.core.stream.Stream` objects throughout
this tutorial. The first one, ``singlechannel``, just contains one continuous
:class:`~obspy.core.trace.Trace` and the other one, ``threechannel``,
contains three channels of a seismograph.

.. doctest::

   >>> from obspy.core import read
   >>> singlechannel = read('http://examples.obspy.org/COP.BHE.DK.2009.050')
   >>> print singlechannel
   1 Trace(s) in Stream:
   DK.COP..BHE | 2009-02-19T00:00:00.035100Z - 2009-02-19T23:59:59.985100Z | 20.0 Hz, 1728000 samples

.. doctest::

   >>> threechannels = read('http://examples.obspy.org/COP.BHE.DK.2009.050')
   >>> threechannels += read('http://examples.obspy.org/COP.BHN.DK.2009.050')
   >>> threechannels += read('http://examples.obspy.org/COP.BHZ.DK.2009.050')
   >>> print threechannels
   3 Trace(s) in Stream:
   DK.COP..BHE | 2009-02-19T00:00:00.035100Z - 2009-02-19T23:59:59.985100Z | 20.0 Hz, 1728000 samples
   DK.COP..BHN | 2009-02-19T00:00:00.025100Z - 2009-02-19T23:59:59.975100Z | 20.0 Hz, 1728000 samples
   DK.COP..BHZ | 2009-02-19T00:00:00.025100Z - 2009-02-19T23:59:59.975100Z | 20.0 Hz, 1728000 samples

--------------
Basic Plotting
--------------

Using the :meth:`~obspy.core.stream.Stream.plot` method of the
:class:`~obspy.core.stream.Stream` objects will show the plot. The default
size of the plots is 800x250 pixel. Use the ``size`` attribute to adjust it to
your needs.

   >>> singlechannel.plot()

.. plot:: source/tutorial/waveform_plotting_tutorial_1.py

-------------------
Saving Plot to File
-------------------

To save a file use the ``outfile`` parameter. The format is determined
automatically from the filename. This example also shows the options to adjust
the color of the graph, the number of ticks shown, their format and rotation
and how to set the start- and endtime of the plot.

   >>> dt = singlechannel[0].stats.starttime
   >>> singlechannel.plot(outfile = 'singlechannel_adjusted.png',
   ...                    color = 'red', number_of_ticks = 7,
   ...                    tick_rotation = 5, tick_format = '%I:%M %p',
   ...                    starttime = dt + 60*60, endtime = dt + 60*60 + 120)

.. plot:: source/tutorial/waveform_plotting_tutorial_2.py

--------------------------
Plotting multiple Channels
--------------------------

If the :class:`~obspy.core.stream.Stream` object contains more than one
:class:`~obspy.core.trace.Trace`, each Trace will be plotted in a subplot.
The start- and endtime of each trace will be the same and the range on the
y-axis will also be identical on each trace. Each additional subplot will add
250 pixel to the height of the resulting plot. Use the ``size`` attribute to
change the size of the plot.

   >>> threechannels.plot()

.. plot:: source/tutorial/waveform_plotting_tutorial_3.py

------------
Plot Options
------------

Various options are available to change the appearance of the waveform:

   ``outfile``
      Output file string. Also used to automatically determine the output
      format. Currently supported is emf, eps, pdf, png, ps, raw, rgba, svg
      and svgz output. Defaults to ``None``.
   ``format``
      Format of the graph picture. If no ``format`` is given, the ``outfile``
      parameter will be used to try to automatically determine the output
      format. If the output format can not be detected, it defaults to png
      output. If no ``outfile`` is specified but a ``format`` is, then a binary
      imagestring will be returned. Defaults to ``None``.
   ``size``
      Size tupel in pixel for the output file. This corresponds to the
      resolution of the graph for vector formats. Defaults to ``(800, 250)``
      pixel.
   ``starttime``
      Starttime of the graph as a datetime object. If not set, the graph will
      be plotted from the beginning. Defaults to ``False``.
   ``endtime``
      Endtime of the graph as a datetime object. If not set, the graph will be
      plotted until the end. Defaults to ``False``.
   ``dpi``
      Dots per inch of the output file. This also affects the size of most
      elements in the graph (text, linewidth, ...). Defaults to ``100``.
   ``color``
      Color of the graph. Defaults to ``'k'`` (black).
   ``bgcolor``
      Background color of the graph. Defaults to ``'w'`` (white).
   ``transparent``
      Make all backgrounds transparent (``True`` or ``False``). This will
      overwrite the ``bgcolor`` parameter. Defaults to ``False``.
   ``minmaxlist``
      A list containing minimum, maximum and timestamp values. If none is
      supplied, it will be created automatically. Useful for caching.
      Defaults to ``False``.
   ``number_of_ticks``
      Number of the ticks on the time scale to display. Defaults to ``5``.
   ``tick_format``
      Format of the time ticks according to strftime methods. Defaults to
      ``'%H:%M:%S'``.
   ``tick_rotation``
      Number of degrees of rotation for ticks on the time axis. Ticks with big
      rotations might be cut off depending on the ``tick_format``.
      Defaults to ``0``. 

-------------
Color Options
-------------

Colors can be specified as defined in the :mod:`matplotlib.colors`
documentation.

Short Version: For all color values, you can either use:

* legit `HTML color names <http://www.w3.org/TR/css3-color/#html4>`_, e.g.
  ``'blue'``,
* HTML hex strings, e.g. ``'#ee00ff'``,
* pass an string of a R, G, B tuple, where each of the component is a float
  value in the range of 0 to 1, e.g. ``'(1,0.25,0.5)'``, or
* use a single letters for the basic built-in colors, such as ``'b'``
  (blue), ``'g'`` (green), ``'r'`` (red), ``'c'`` (cyan), ``'m'`` (magenta),
  ``'y'`` (yellow), ``'k'`` (black), ``'w'`` (white).
