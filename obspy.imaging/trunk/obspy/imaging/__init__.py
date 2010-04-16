"""
obspy.imaging - tools for displaying features used in seismology
================================================================
This module provides routines for plotting and displaying often used in
seismology. It can currently plot waveform data, generate spectrograms and draw
beachballs. The module obspy.imaging depends on the plotting module
`matplotlib <http://matplotlib.sourceforge.net/>`_.


:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU General Public License (GPLv2)


Seismograms
-----------
This submodule can plot multiple :class:`~obspy.core.trace.Trace` in one
:class:`~obspy.core.stream.Stream` object and has various other optional
arguments to adjust the plot like color and tick format changes.

Additionally the start- and endtime of the plot can be given as
:class:`obspy.core.utcdatetime.UTCDateTime` objects.

Examples files may be retrieved via http://examples.obspy.org.

>>> from obspy.core import read
>>> st = read('http://examples.obspy.org/BW.BGLD.__.EHE.D.2008.001.first_10_percent')
>>> print st
3 Trace(s) in Stream:
BW.BGLD..EHE | 2010-01-01T00:00:00.000000Z - 2010-01-01T12:00:00.000000Z | 200.0 Hz, 8640001 samples
BW.BGLD..EHN | 2010-01-01T00:00:00.000000Z - 2010-01-01T12:00:00.000000Z | 200.0 Hz, 8640001 samples
BW.BGLD..EHZ | 2010-01-01T00:00:00.000000Z - 2010-01-01T12:00:00.000000Z | 200.0 Hz, 8640001 samples
>>> st.plot(color = 'gray', tick_format = '%I:%M %p', starttime =
... st[0].stats.starttime, endtime = st[0].stats.starttime + 60*60)

.. plot::

    from obspy.core import read, UTCDateTime
    from copy import deepcopy
    from numpy.random import ranf
    st = read('http://examples.obspy.org/BW.BGLD.__.EHE.D.2008.001.first_10_percent')
    st[0].stats.starttime = UTCDateTime(2010, 1, 1)
    starttime = st[0].stats.starttime
    st.trim(starttime, starttime + 60*60)
    st += deepcopy(st)
    st += deepcopy(st)
    st[1].stats.channel = 'EHN'
    st[2].stats.channel = 'EHZ'
    st[1].data = st[1].data * (1.0 + ranf(st[0].stats.npts)/8
    st[2].data = st[2].data * (0.95 + ranf(st[0].stats.npts)/8)
    st.plot(color = 'gray', tick_format = '%I:%M %p')

Spectrograms
------------
This submodule plots spectrograms.

The spectrogram will on default have 80% overlap and a maximum sliding window
size of 4096 points.

>>> from obspy.imaging import spectrogram
>>> from obspy.core import read
>>>
>>> st = read('loc_RNON20040609200559.z')
>>> spectrogram.spectrogram(st[0].data, st[0].stats.sampling_rate)

.. plot::

    from obspy.imaging import spectrogram
    from obspy.core import read
    st = read('../obspy.gse2/trunk/obspy/gse2/tests/data/loc_RNON20040609200559.z')
    spectrogram.spectrogram(st[0].data, st[0].stats.sampling_rate)

For more info see :func:`~obspy.imaging.spectrogram.spectrogram`.

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

For more info see :func:`~obspy.imaging.beachball.Beachball`.


Plot the beach ball as matplotlib collection into an existing plot.

>>> import matplotlib.pyplot as plt
>>> from obspy.imaging.beachball import Beach
>>>
>>> np1 = [150, 87, 1]
>>> mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
>>>
>>> plt.plot([-100, 100], [0, 100], "rv", ms=10)
>>> ax = plt.gca()
>>> ax.add_collection(Beach(np1, xy=(5,5), width=30))
>>> ax.add_collection(Beach(mt, xy=(5,5), width=50))
>>> plt.axis('scaled')
>>> plt.axis([-120, 120, -20, 120])

.. plot::

    import matplotlib.pyplot as plt
    from obspy.imaging.beachball import Beach
    np1 = [150, 87, 1]
    mt = [-2.39, 1.04, 1.35, 0.57, -2.94, -0.94]
    plt.plot([-100, 100], [0, 100], "rv", ms=10)
    ax = plt.gca()
    ax.add_collection(Beach(np1, xy=(50,50), width=30))
    ax.add_collection(Beach(mt, xy=(-50,50), width=50))
    plt.axis('scaled')
    plt.axis([-120, 120, -20, 120])

For more info see :func:`~obspy.imaging.beachball.Beach`.


Saving plots into files
-----------------------
All plotting routines offer an outfile argument to save the result into a file.

The outfile parameter is also used to automatically determine the file format.
Available output formats mainly depend on your matplotlib settings. Common
formats are png, svg, pdf or ps.

>>> from obspy.core import read
>>> st = read('loc_RNON20040609200559.z')
>>> st.plot(outfile = 'graph.png')

.. plot::

    from obspy.core import read
    st = read('../obspy.gse2/trunk/obspy/gse2/tests/data/loc_RNON20040609200559.z')
    st.plot()
"""

# Please do not import any modules using matplotlib - otherwise it will disturb
# the test suite (running without X11 or any other display)
# see tests/__init__.py for details

from obspy.core.util import _getVersionString

__version__ = _getVersionString("obspy.imaging")
