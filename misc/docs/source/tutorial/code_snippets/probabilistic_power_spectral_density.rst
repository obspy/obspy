==================================================
Visualizing Probabilistic Power Spectral Densities
==================================================

The following code example shows how to use the
:class:`~obspy.signal.spectral_estimation.PPSD` class defined in
:mod:`obspy.signal`. The routine is useful for interpretation of e.g. noise
measurements for site quality control checks. For more information on the topic
see [McNamara2004]_.

.. doctest::

    >>> from obspy.core import read
    >>> from obspy.xseed import Parser
    >>> from obspy.signal import PPSD

Read data and select a trace with the desired station/channel combination:

.. doctest::

    >>> st = read("http://examples.obspy.org/BW.KW1..EHZ.D.2011.037")
    >>> tr = st.select(id="BW.KW1..EHZ")[0]

Metadata can be provided as an
:class:`~obspy.core.inventory.inventory.Inventory` (e.g. from a StationXML file
or from a request to a FDSN web service), a :class:`~obspy.io.xseed.Parser`
(e.g. from a dataless SEED file), a filename of a local RESP file (or a legacy
poles and zeros dictionary). Then we initialize a new
:class:`~obspy.signal.spectral_estimation.PPSD` instance. The ppsd object will
then make sure that only appropriate data go into the probabilistic psd
statistics.

.. doctest::

    >>> parser = Parser("http://examples.obspy.org/dataless.seed.BW_KW1")
    >>> ppsd = PPSD(tr.stats, metadata=parser)

Now we can add data (either trace or stream objects) to the ppsd estimate. This
step may take a while. The return value ``True`` indicates that the data was
successfully added to the ppsd estimate.

.. doctest::

    >>> ppsd.add(st)
    True

We can check what time ranges are represented in the ppsd estimate.
``ppsd.times`` contains a sorted list of start times of the one hour long
slices that the psds are computed from (here only the first two are printed).

.. doctest::

    >>> print(ppsd.times[:2])
    [UTCDateTime(2011, 2, 6, 0, 0, 0, 935000), UTCDateTime(2011, 2, 6, 0, 30, 0, 935000)]
    >>> print("number of psd segments:", len(ppsd.times))
    number of psd segments: 47

Adding the same stream again will do nothing (return value ``False``), the ppsd
object makes sure that no overlapping data segments go into the ppsd estimate.

.. doctest::

    >>> ppsd.add(st)
    False
    >>> print("number of psd segments:", len(ppsd.times))
    number of psd segments: 47

Additional information from other files/sources can be added step by step.

.. doctest::

    >>> st = read("http://examples.obspy.org/BW.KW1..EHZ.D.2011.038")
    >>> ppsd.add(st)
    True
        
The graphical representation of the ppsd can be displayed in a matplotlib
window..

    >>> ppsd.plot()

..or saved to an image file:

    >>> ppsd.plot("/tmp/ppsd.png")  # doctest: +SKIP
    >>> ppsd.plot("/tmp/ppsd.pdf")  # doctest: +SKIP

.. plot:: tutorial/code_snippets/probabilistic_power_spectral_density.py

A (for each frequency bin) cumulative version of the histogram can also be
visualized:

    >>> ppsd.plot(cumulative=True)

.. plot:: tutorial/code_snippets/probabilistic_power_spectral_density3.py

To use the colormap used by PQLX / [McNamara2004]_ you can import and use that
colormap from :mod:`obspy.imaging.cm`:

    >>> from obspy.imaging.cm import pqlx
    >>> ppsd.plot(cmap=pqlx)

.. plot:: tutorial/code_snippets/probabilistic_power_spectral_density2.py

Below the actual PPSD (for a detailed discussion see
[McNamara2004]_) is a visualization of the data basis for the PPSD
(can also be switched off during plotting). The top row shows data fed into the
PPSD, green patches represent available data, red patches represent gaps in
streams that were added to the PPSD. The bottom row in blue shows the single
psd measurements that go into the histogram. The default processing method
fills gaps with zeros, these data segments then show up as single outlying psd
lines.

.. note::
   
   Providing metadata from e.g. a Dataless SEED or StationXML volume is safer
   than specifying static poles and zeros information (see
   :class:`~obspy.signal.spectral_estimation.PPSD`). 
