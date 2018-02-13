"""
obspy.io.rg16 - receiver gather v1.6 read support for ObsPy
===========================================================

Functions to read waveform data from Receiver Gather 1.6-1 format.
This format is used for continuous data by
`Farfield Nodal <fairfieldnodal.com>`_
`Zland product line <http://fairfieldnodal.com/equipment/zland>`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Inspired by a similar project by a `similar script
<https://github.com/iceseismic/Fairfield-Receiver-Gather`_.


.. note::
    1. Some useful diagrams, provided by Faifield technical support, for
    understanding Base Scan intervals, data format, and sensor type numbers
    can be found `here <https://imgur.com/a/4aneG>`_.

    2. Because there is not a standard deployment orientation, channel codes
    contained in the file format are 2, 3, and 4. See the above link for the
    orientation in respect to instrument orientation. By default these are
    mapped to Z, 1, 2 but a custom mapping can be passed to
    :func:`obspy.io.rg16.read_rg16`.

    3. The default polarity of channel 2 (vertical) is down. This is opposite
    of what one might expect.

Reading
-------
The rg16 format can be read using two methods:

1. Using the standard :func:`~obspy.core.stream.read` function. Optionally,
   the format parameter can be specified as "rg16" in order to avoid spending
   time checking all the other foramts.

2. Using the :mod:`obspy.io.rg16` specific function
   :func:`obspy.io.rg16.read_rg16` function.

There are two noteworthy parameters to the read_rg16 function:

1. The parameter "merge" will merge many traces belonging to the same channel
   into a single trace. This is much more efficient than the
   :func: `obspy.core.Steam.merge` when there are many (thousands) of traces
   because the function can make some assumptions about data continuity and
   type. If you are reading a rg16 file that is several GB in size and
   contains many traces it may be beneficial to use this parameter.

2. The parameter "component_map" allows you to pass a dictionary to map the
   component codes to a deployment orientation. For example, if the
   instruments were deployed with the terminal contacts facing south, and
   appropriate value for this would be {2: 'Z', 3: 'N', 4: 'E'}. Each channel
   should then be multiplied by -1 in order to get the expected polarities.
   For one component instruments (Zland 1C) there is no need to use this
   parameter.

>>> import obspy
>>> from obspy.io.rg16.core import read_rg16
>>> from obspy.core.util import get_example_file
>>> filename = get_example_file('three_chans_six_traces.fcnt')
>>> # these are all equivalent, from slowest to fastest:
>>> st = obspy.read(filename)
>>> st = obspy.read(filename, format='rg16')
>>> st = read_rg16(filename)

If the file was very large using the merge parameter is advisable.

>>> st = obspy.read(filename, merge=True)

If three component instruments were deployed with the interface terminals
facing south, using a custom component mapping and reversing the polarity of
each trace can be used to get accurate Z, N, E components.

>>> component_map = {2: 'Z', 3: 'N', 4: 'E'}
>>> st = obspy.read(filename, component_map=component_map)
>>> for tr in st:
...     tr.data = -tr.data


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
