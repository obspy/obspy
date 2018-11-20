"""
obspy.io.rg16 - Receiver Gather v1.6 read support for ObsPy
===========================================================

Functions to read waveform data and initial headers
from Receiver Gather 1.6-1 format.
This format is used to store continuous data recorded by
`Farfield Nodal <fairfieldnodal.com>`_'s
`Zland <http://fairfieldnodal.com/equipment/zland>`_ product line.

:author:
    Derrick Chambers
    Romain Pestourie
:copyright:
    the ObsPy Development Team (devs@obspy.org)
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


Reading waveform data
-------
The waveforms of the rg16 format can be read using two methods:

1. Using the standard :func:`~obspy.core.stream.read` function. Optionally,
   the format parameter can be specified as "rg16" for a speed-up.

2. Using the :mod:`obspy.io.rg16` specific function
   :func:`obspy.io.rg16.core._read_rg16`.

Noteworthy parameters of  :func:`obspy.io.rg16.core._read_rg16`,
which can also be passed as kwargs to :func:`~obspy.core.stream.read`:

* `starttime` and `endtime`: Can be passed
  :class:`~obspy.core.utcdatetime.UTCDateTime` instances in order to only
  load slices of the file at a time, avoiding the need to store the entire
  file contents in memory.

>>> import obspy
>>> from obspy.io.rg16.core import _read_rg16
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('three_chans_six_traces.fcnt')
>>> # these are all equivalent:
>>> st = obspy.read(filename)
>>> st = obspy.read(filename, format='rg16')
>>> st = _read_rg16(filename)

Reading initial headers
-------
The initial headers of the rg16 format can be read using the
:mod:`obspy.io.rg16` specific function
:func:`obspy.io.rg16.core.read_initial_headers`.

>>> from obspy.io.rg16.core import read_initial_headers
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('three_chans_six_traces.fcnt')
>>> initial_headers = read_initial_headers(filename)
"""
