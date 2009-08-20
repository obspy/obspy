# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Integer, MultipleLoop, SimpleLoop


class Blockette060(Blockette):
    """
    Blockette 060: Response Reference Blockette.
    
    Use this blockette whenever you want to replace blockettes [53] through 
    [58] and [61] with their dictionary counterparts, blockettes [43] through 
    [48] and [41]. We recommend placing responses in stage order, even if this 
    means using more than one Response Reference Blockette [60].
    
    Here is an example:
        Stage 1:    Response (Poles & Zeros) Blockette [53]
                    Channel Sensitivity/Gain Blockette [58]
                    First response reference blockette:
                    Response Reference Blockette [60]
        Stage 2:        [44] [47] [48]
        Stage 3:        [44] [47] [48]
        Stage 4:        [44] [47]
                        Channel Sensitivity/Gain Blockette [58]
        Stage 5:    Response (Coefficients) Blockette [54]
                    (End of first response reference blockette)
                    Second response reference blockette:
                    Response Reference Blockette [60]
        Stage 5         (continued): [47] [48]
        Stage 6:        [44] [47] [48]
                    (End of second response reference blockette)
    
    Substitute Response Reference Blockette [60] anywhere the original 
    blockette would go, but be sure to place it in the same position as the 
    original would have gone. (Note that this blockette uses a repeating field 
    (response reference) within another repeating field (stage value). This is 
    the only blockette in the current version (2.1) that has this "two 
    dimensional" structure.)
    """

    id = 60
    name = "Response Reference Blockette"
    fields = [
        Integer(3, "Number of stages", 2),
        #REPEAT field 4, with appropriate fields 5 and 6, for each filter stage
        MultipleLoop("FIR Coefficient", "Number of stages", [
            Integer(4, "Stage sequence number", 2),
            Integer(5, "Number of responses", 2),
            #REPEAT field 6, one for each response within each stage:
            SimpleLoop("Number of responses",
                Integer(6, "Response lookup key", 4)),
        ]),
    ]
