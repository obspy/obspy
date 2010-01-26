"""
obspy.imaging - tools for displaying features used in seismology
================================================================
This module provides routines for plotting and displaying often used in
seismology. It can currently plot waveform data, generate spectograms and draw
beachballs.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU General Public License (GPLv2)

It uses `matplotlib <http://matplotlib.sourceforge.net/>`_ to draw the graphs.


Saving files
------------
All plotting routines support an outfile argument to save the file. This is used to
automatically determine the file format. Depending on your matplotlib version
the available file formats probably are png, svg, pdf and ps.

>>> from obspy.core import read
>>> st = read('test_file.gse'
>>> st.plot(outfile = 'graph.png')

.. plot::

    from obspy.core import read
    st = read('../obspy.gse2/trunk/obspy/gse2/tests/data/loc_RNON20040609200559.z')
    st.plot()

This submodule can plot multiple :class:`~obspy.core.trace.Trace` in one
:class:`~obspy.core.stream.Stream` object and has various other optional
arguments to adjust the plot like color and tick format changes.

Additionally the start- and endtime of the plot can be given as
:class:`obspy.core.utcdatetime.UTCDateTime` objects.

>>> from obspy.core import read
>>> st = read('threechannels.mseed')
>>> print st
    3 Trace(s) in Stream:
    BW.BGLD..EHE | 2010-01-01T00:00:00.000000Z - 2010-01-01T12:00:00.000000Z | 200.0 Hz, 8640001 samples
    BW.BGLD..EHN | 2010-01-01T00:00:00.000000Z - 2010-01-01T12:00:00.000000Z | 200.0 Hz, 8640001 samples
    BW.BGLD..EHZ | 2010-01-01T00:00:00.000000Z - 2010-01-01T12:00:00.000000Z | 200.0 Hz, 8640001 samples
>>> st.plot(color = 'gray', tick_format = '%I:%M %p', starttime =
    st[0].stats.starttime, endtime = st[0].stats.starttime + 60*60)

.. plot::

    from obspy.core import read, UTCDateTime
    from copy import deepcopy
    from numpy.random import ranf
    st = read('../obspy.mseed/trunk/obspy/mseed/tests/data/BW.BGLD.__.EHE.D.2008.001.first_10_percent')
    st[0].stats.starttime = UTCDateTime(2010, 1, 1)
    starttime = st[0].stats.starttime
    st.trim(starttime, starttime + 60*60)
    st += deepcopy(st)
    st += deepcopy(st)
    st[1].stats.channel = 'EHN'
    st[2].stats.channel = 'EHZ'
    st[1].data = st[1].data * (1.0 + ranf(st[0].stats.npts)/8)
    st[2].data = st[2].data * (0.95 + ranf(st[0].stats.npts)/8)
    st.plot(color = 'gray', tick_format = '%I:%M %p')

Spectograms
----------
This submodule plots spectograms.

The spectrogram will on default have 80% overlap and a maximum sliding window size
of 4096 points.

>>> from obspy.imaging import spectrogram
>>> from obspy.core import read
>>> st = read('test_file.gse')
>>> spectogram.spectogram(st[0].data, st[0].stats.sampling_rate)

.. plot::

    from obspy.imaging import spectrogram
    from obspy.core import read
    st = read('../obspy.gse2/trunk/obspy/gse2/tests/data/loc_RNON20040609200559.z')
    spectrogram.spectrogram(st[0].data, st[0].stats.sampling_rate)

The following keyword arguments are possible:

* sample_rate = 100.0: Samplerate in Hz
* log = False: True logarithmic frequency axis, False linear frequency axis
* per_lap = 0.8: Percent of overlap
* nwin = 10: Approximate number of windows.
* outfile = None: String for the filename of output file, if None interactive plotting is activated.
* format = None: Format of image to save


Beachballs
----------
Draws a beach ball diagram of an earthquake focal mechanism.


The focal mechanism can be given by 3 (strike, dip, and rake) components. The
strike is of the first plane, clockwise relative to north. The dip is of the
first plane, defined clockwise and perpendicular to strike, relative to
horizontal such that 0 is horizontal and 90 is vertical. The rake is of the
first focal plane solution. 90 moves the hanging wall up-dip (thrust), 0 moves
it in the strike direction (left-lateral), -90 moves it down-dip (normal),
and 180 moves it opposite to strike (right-lateral). 

>>> from obspy.imaging.beachball import Beachball
>>> np1 = [150, 87, 1]
>>> Beachball(np1)

.. plot::

    from obspy.imaging.beachball import Beachball
    np1 = [150, 87, 1]
    Beachball(np1)

The focal mechanism can also be specified using the 6 independant components of
the moment tensor (Mxx, Myy, Mzz, Mxy, Mxz, Myz).

>>> from obspy.imaging.beachball import Beachball
>>> mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
>>> Beachball(mt)

.. plot::

    from obspy.imaging.beachball import Beachball
    mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
    Beachball(mt)
    
The following keyword arguments are possible:

* size = 200: Diameter of the beachball.
* linewidth = 2: Line width.
* color = 'b' : Color used for the quadrants of tension.
* alpha = 1.0: Alpha value
* outfile = None: Filename of the output file. Also used to determine the file format.
* format = None: File format.
"""
# Please do not import anything here. It is needed to run the tests
# without X11 or any other display, see tests/__init__.py for details
#from spectrogram import spectrogram
