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
       parameters are expressed in Hertz (ie ``alias_filter_frequency``,
       ``test_signal_generator_frequency_1``...) and fields concerning
       time are expressed in second (``channel_set_end_time``,
       ``test_signal_generator_activation_time``...), except for the dates.

    2. Documentation about fcnt format can be found in the directory
       ``obspy/io/rg16/docs``.

Instrument Orientation
----------------------
Because there is not a standard method for instrument orientation, mapping
orientation codes to Z, N, E is not possible without knowledge of how the
instruments were deployed. Orientation codes returned are 2, 3, 4 which
relate to the instrument position as illustrated in the following diagram:

.. figure:: /_images/rg16_node_orientation.png

Reading the waveforms
---------------------
Reading RG16 is handled by using ObsPy's standard
:func:`~obspy.core.stream.read` function. The format can be detected
automatically, however setting the format parameter as "rg16" lead to a
speed up.
Several key word arguments are available: ``headonly``, ``starttime``,
``endtime``, ``merge``, ``contacts_north``, ``details``. They are passed to the
:func:`obspy.io.rg16.core._read_rg16` function so refer to it for details to
each parameter.

>>> import obspy
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('three_chans_six_traces.fcnt')
>>> # these are all equivalent:
>>> st = obspy.read(filename)
>>> st = obspy.read(filename, format='rg16')

If the file is very large, using the ``merge`` parameter may speed up
downstream processing significantly.

>>> st = obspy.read(filename, merge=True)

If the instruments are single component, or if the gold contact terminals
were deployed facing north, setting ``contacts_north`` to True will result
in a stream with seed compliant channel codes with orientations Z, N, E.

>>> st = obspy.read(filename, contacts_north=True)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
