# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette 
from obspy.xseed.fields import Float, Integer, VariableString, MultipleLoop


class Blockette058(Blockette):
    """Blockette 058: Channel Sensitivity/Gain Blockette.
    
    When used as a gain (stage ≠ 0), this blockette is the gain for this stage
    at the given frequency. Different stages may be at different frequencies. 
    However, it is strongly recommended that the same frequency be used in all 
    stages of a cascade, if possible. When used as a sensitivity(stage=0), 
    this blockette is the sensitivity (in counts per ground motion) for the 
    entire channel at a given frequency, and is also referred to as the 
    overall gain. The frequency here may be different from the frequencies in 
    the gain specifications, but should be the same if possible. If you use 
    cascading (more than one filter stage), then SEED requires a gain for each 
    stage. A final sensitivity (Blockette [58], stage = 0, is required. If you 
    do not use cascading (only one stage), then SEED must see a gain, a 
    sensitivity, or both.
    
    Sample:
    0580035 3 3.27680E+03 0.00000E+00 0
    """
    
    id = 58
    name = "Channel Sensitivity or Gain Blockette"
    fields = [
        Integer(3, "Stage sequence number", 2),
        Float(4, "Sensitivity or gain", 12, mask='%+1.5e'),
        Float(5, "Frequency ", 12, mask='%+1.5e'),
        Integer(6, "Number of history values", 2),
        # REPEAT fields 7 — 9 for the Number of history values:
        MultipleLoop('History values', "Number of history values", [ 
            Float(7, "Sensitivity for calibration", 12, mask='%+1.5e'),
            Float(8, "Frequency of calibration", 12, mask='%+1.5e'),
            VariableString(9, "Time of above calibration", 1, 22 , 'T')
        ])
    ]
