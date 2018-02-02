# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Purpose: Core classes of ObsPy: Python for Seismological Observatories
#   Author: Robert Barsch
#           Moritz Beyreuther
#           Lion Krischer
#           Tobias Megies
#
# Copyright (C) 2008-2012 Robert Barsch, Moritz Beyreuther, Lion Krischer,
#                         Tobias Megies
# -----------------------------------------------------------------------------
"""
obspy.core - Core classes of ObsPy
==================================

This package contains common methods and classes for ObsPy. It includes the
:class:`~obspy.core.stream.Stream`, :class:`~obspy.core.trace.Trace`,
:class:`~obspy.core.utcdatetime.UTCDateTime`, :class:`~obspy.core.trace.Stats`
classes and methods for :func:`reading <obspy.core.stream.read>` seismogram
files.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Waveform Data
-------------

Summary
^^^^^^^
Seismograms of various formats (e.g. SAC, MiniSEED, GSE2, SEISAN, Q, etc.) can
be imported into a :class:`~obspy.core.stream.Stream` object using the
:func:`~obspy.core.stream.read` function.

Streams are list-like objects which contain multiple
:class:`~obspy.core.trace.Trace` objects, i.e. gap-less continuous time series
and related header/meta information.

Each Trace object has the attribute ``data`` pointing to a NumPy_
:class:`~numpy.ndarray` of the actual time series and the attribute ``stats``
which contains all meta information in a dict-like
:class:`~obspy.core.trace.Stats` object. Both attributes ``starttime`` and
``endtime`` of the Stats object are
:class:`~obspy.core.utcdatetime.UTCDateTime` objects.
A multitude of helper methods are attached to
:class:`~obspy.core.stream.Stream` and :class:`~obspy.core.trace.Trace` objects
for handling and modifying the waveform data.

.. figure:: /_images/Stream_Trace.png

Example
^^^^^^^
A :class:`~obspy.core.stream.Stream` with an example seismogram can be created
by calling :func:`~obspy.core.stream.read()` without any arguments.
Local files can be read by specifying the filename, files stored on http
servers (e.g. at https://examples.obspy.org) can be read by specifying their
URL. For details and supported formats see the documentation of
:func:`~obspy.core.stream.read`.

>>> from obspy import read
>>> st = read()
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
BW.RJOB..EHN | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
BW.RJOB..EHE | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
>>> tr = st[0]
>>> print(tr)  # doctest: +ELLIPSIS
BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - ... | 100.0 Hz, 3000 samples
>>> tr.data
array([ 0.        ,  0.00694644,  0.07597424, ...,  1.93449584,
        0.98196204,  0.44196924])
>>> print(tr.stats)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
         network: BW
         station: RJOB
        location:
         channel: EHZ
       starttime: 2009-08-24T00:20:03.000000Z
         endtime: 2009-08-24T00:20:32.990000Z
   sampling_rate: 100.0
           delta: 0.01
            npts: 3000
           calib: 1.0
           ...
>>> tr.stats.starttime
UTCDateTime(2009, 8, 24, 0, 20, 3)

Event Metadata
--------------

Event metadata are handled in a hierarchy of classes closely modelled after the
de-facto standard format `QuakeML <https://quake.ethz.ch/quakeml/>`_.
See the IPython notebooks mentioned in the :ref:`ObsPy Tutorial <tutorial>` for
more detailed usage examples. See
:func:`~obspy.core.event.catalog.read_events()` and
:meth:`Catalog.write() <obspy.core.event.catalog.Catalog.write>` for supported
formats.

.. figure:: /_images/Event.png

Station Metadata
----------------

Station metadata are handled in a hierarchy of classes closely modelled after
the de-facto standard format
`FDSN StationXML <https://www.fdsn.org/xml/station/>`_ which was developed as a
human readable XML replacement for Dataless SEED.
See :mod:`obspy.core.inventory` for more details. See
:func:`~obspy.core.inventory.inventory.read_inventory()` and
:meth:`Inventory.write() <obspy.core.inventory.inventory.Inventory.write>` for
supported formats.

.. figure:: /_images/Inventory.png

.. _NumPy: http://www.numpy.org
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

# don't change order
from obspy.core.utcdatetime import UTCDateTime  # NOQA
from obspy.core.util.attribdict import AttribDict  # NOQA
from obspy.core.trace import Stats, Trace  # NOQA
from obspy.core.stream import Stream, read  # NOQA
from obspy.scripts.runtests import run_tests  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
