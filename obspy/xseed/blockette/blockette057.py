# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer
from obspy.xseed.utils import formatRESP


RESP = """\
#\t\t+                      +------------------------------+                  \
     +
#\t\t+                      |   Decimation,%6s ch %s   |                      \
 +
#\t\t+                      +------------------------------+                  \
     +
#\t\t
B057F03     Stage sequence number:                 %s
B057F04     Input sample rate:                     %s
B057F05     Decimation factor:                     %s
B057F06     Decimation offset:                     %s
B057F07     Estimated delay (seconds):             %s
B057F08     Correction applied (seconds):          %s
#\t\t
"""


class Blockette057(Blockette):
    """
    Blockette 057: Decimation Blockette.

    Many digital filtration schemes process a high sample rate data stream;
    filter; then decimate, to produce the desired output. Use this blockette
    to describe the decimation phase of the stage. You would usually place it
    between a Response (Coefficients) Blockette [54] and the Sensitivity/Gain
    Blockette [58] phases of the filtration stage of the channel. Include
    this blockette with non-decimated stages because you must still specify
    the time delay. (In this case, the decimation factor is 1 and the offset
    value is 0.)

    Sample:
    057005132 .0000E+02    1    0 0.0000E+00 0.0000E+00
    """

    id = 57
    name = "Decimation"
    fields = [
        Integer(3, "Stage sequence number", 2),
        Float(4, "Input sample rate", 10, mask='%1.4e'),
        Integer(5, "Decimation factor", 5),
        Integer(6, "Decimation offset", 5),
        Float(7, "Estimated delay", 11, mask='%+1.4e'),
        Float(8, "Correction applied", 11, mask='%+1.4e')
    ]

    def getRESP(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        out = RESP % (station, channel,
                      self.stage_sequence_number,
                      formatRESP(self.input_sample_rate, 6),
                      self.decimation_factor,
                      self.decimation_offset,
                      formatRESP(self.estimated_delay, 6),
                      formatRESP(self.correction_applied, 6))
        return out.encode()
