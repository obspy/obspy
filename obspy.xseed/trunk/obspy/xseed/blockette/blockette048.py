# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, VariableString, Loop


class Blockette048(Blockette):
    """
    Blockette 048: Channel Sensitivity/Gain Dictionary Blockette.
    
    See Channel Sensitivity/Gain Blockette [58] for more information.
    """

    id = 48
    name = "Channel Sensitivity Gain Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        Float(5, "Sensitivity gain", 12, mask='%+1.5e'),
        Float(6, "Frequency", 12, mask='%+1.5e'),
        Integer(7, "Number of history values", 2),
        # REPEAT fields 8 — 10 for the Number of history values:
        Loop('History', "Number of history values", [
            Float(8, "Sensitivity for calibration", 12, mask='%+1.5e'),
            Float(9, "Frequency of calibration sensitivity", 12,
                  mask='%+1.5e'),
            VariableString(10, "Time of above calibration", 1, 22 , 'T')
        ])
    ]
