"""
obspy.io.rg16 - Receiver Gather v1.6 read support for ObsPy
===========================================================

Functions to read waveform data from Receiver Gather 1.6-1 format.
This format is used to store continuous data recorded by
`Farfield Nodal <fairfieldnodal.com>`_'s
`Zland <http://fairfieldnodal.com/equipment/zland>`_ product line.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. note::
    1. Some useful diagrams, provided by Faifield Nodal technical support,
    can be found `here <https://imgur.com/a/4aneG>`_.

    2. Another code to read receiver gather format 1.6 can be found
    `here <https://github.com/iceseismic/Fairfield-Receiver-Gather>`_.

Instrument Orientation
----------------------
Because there is not a standard method for instrument orientation, mapping
orientation codes to Z, N, E is not possible without knowledge of how the
instruments were deployed. Orientation codes returned are 2, 3, 4 which
relate to the instrument position as illustrated in the following diagram:

.. figure:: /_images/rg16_node_orientation.png

Reading
-------
The rg16 format can be read using two methods:

1. Using the standard :func:`~obspy.core.stream.read` function. Optionally,
   the format parameter can be specified as "rg16" for a modest speed-up.

2. Using the :mod:`obspy.io.rg16` specific function
   :func:`obspy.io.rg16.core.read_rg16`.

Noteworthy parameters of  :func:`obspy.io.rg16.core.read_rg16`,
which can also be passed as kwargs to :func:`~obspy.core.stream.read`:

* merge: If True will merge traces belonging to the same channel
  into a single trace. This is much more efficient than other merge methods
  when there are many (thousands) of traces because assumptions about data
  continuity and type can be made.

* starttime and enditme: Can be passed
  :class:`~obspy.core.utcdatetime.UTCDateTime` instances in order to only
  load slices of the file at a time, avoiding the need to store the entire
  file contents in memory.

* contacts_north: If True indicates the file either contains single component
  traces or that the instruments were deployed with the gold contact terminals
  facing north. If this parameter is used, it will map the components to Z, N,
  and E (if 3 component) as well as correct the polarity for the vertical
  component.

>>> import obspy
>>> from obspy.io.rg16.core import read_rg16
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('three_chans_six_traces.fcnt')
>>> # these are all equivalent:
>>> st = obspy.read(filename)
>>> st = obspy.read(filename, format='rg16')
>>> st = read_rg16(filename)

If the file is very large, using the merge parameter may speed up downstream
processing significantly.

>>> st = obspy.read(filename, merge=True)

If the instruments are single component, or if the gold contact terminals
were deployed facing north, setting standard_orientation to True will result
in a stream with seed compliant channel codes with orientations Z, N, E.

>>> st = obspy.read(filename, contacts_north=True)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
