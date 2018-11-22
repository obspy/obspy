"""
obspy.io.rg16 - Receiver Gather v1.6 read support for ObsPy
===========================================================

Functions to read waveform data from Receiver Gather 1.6-1 format.
This format is used to store continuous data recorded by
`Farfield Nodal <fairfieldnodal.com>`_'s
`Zland <http://fairfieldnodal.com/equipment/zland>`_ product line.

:author:
    Derrick Chambers
    Romain Pestourie (Ecole et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. note::
    1. In order to homogenize the units, fields concerning frequencies
       parameters are expressed in Hertz (ie alias_filter_frequency,
       test_signal_generator_frequency_1...) and fields concerning
       time are expressed in second (channel_set_end_time,
       test_signal_generator_activation_time...), except for the dates.

    2. Documentation about fcnt format
       can be found in the directory obspy.io.rg16.doc

Instrument Orientation
----------------------
Because there is not a standard method for instrument orientation, mapping
orientation codes to Z, N, E is not possible without knowledge of how the
instruments were deployed. Orientation codes returned are 2, 3, 4 which
relate to the instrument position as illustrated in the following diagram:

.. figure:: /_images/rg16_node_orientation.png

Reading the waveforms
-------
The rg16 format can be read using two methods:

1. Using the standard :func:`~obspy.core.stream.read` function. Optionally,
   the format parameter can be specified as "rg16" for a speed-up.

2. Using the :mod:`obspy.io.rg16` specific function
   :func:`obspy.io.rg16.core._read_rg16`.

Noteworthy parameters of  :func:`obspy.io.rg16.core._read_rg16`,
which can also be passed as kwargs to :func:`~obspy.core.stream.read`:

* `starttime` and `endtime`: :class:`~obspy.core.utcdatetime.UTCDateTime`
  instances can be passedin order to only load slices of the file at a time,
  avoiding the need to store the entire file contents in memory.

* `merge`: If `True` will merge traces belonging to the same channel
  into a single trace. This is much more efficient than other merge methods
  when there are many (thousands) of traces because assumptions about data
  continuity and type can be made.

* `contacts_north`: If this parameter is set to True, it will map the
  components to Z (1C, 3C), N (3C), and E (3C) as well as correct
  the polarity for the vertical component.

* `details`: If this parameter is set to True, all the information contained
  in the headers is extracted. These information is stored in an attribute
  dict in the Stats object named "rg16".The "rg16" attribute dict owns two
  attribute dict named "initial_headers" and "trace_headers". The
  "initial_headers" contains information relative to the headers located at
  the beginning of the file. "initial_headers" owns four attribute dict named
  "general_header_1", " general_header_2", "channel_sets_descriptor",
  "extended_headers. The "trace_headers" contains information
  relative to the data block.

>>> import obspy
>>> from obspy.io.rg16.core import _read_rg16
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('three_chans_six_traces.fcnt')
>>> # these are all equivalent:
>>> st = obspy.read(filename)
>>> st = obspy.read(filename, format='rg16')
>>> st = _read_rg16(filename)

If the file is very large, using the `merge` parameter may speed up downstream
processing significantly.

>>> st = obspy.read(filename, merge=True)

If the instruments are single component, or if the gold contact terminals
were deployed facing north, setting `contacts_north` to `True` will result
in a stream with seed compliant channel codes with orientations Z, N, E.

>>> st = obspy.read(filename, contacts_north=True)

Reading the initial headers
-------
The initial headers at the beginning of the rg16 file can be read
separetely using the :mod:`obspy.io.rg16` specific function
:func:`obspy.io.rg16.core._read_initial_headers`

>>> initial_headers = _read_initial_headers(filename)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
