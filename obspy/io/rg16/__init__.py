"""
Functions to read waveform data from Receiver Gather 1.6-1 foramt.
This format is used for continuous data by Farfield's (fairfieldnodal.com)
Zland product line (http://fairfieldnodal.com/equipment/zland).

Inspired by a similar project by Thomas Lecocq
found here: https://github.com/iceseismic/Fairfield-Receiver-Gather.

Some useful diagrams, provided by Faifield technical support, for
understanding Base Scan intervals, data format, and sensor type numbers
can be found here: https://imgur.com/a/4aneG.

Note: Because there is not a standard deployment orientation, channel codes
returned are 2, 3, and 4. See the link above to map these codes to
the appropriate direction based on instrument orientation.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
