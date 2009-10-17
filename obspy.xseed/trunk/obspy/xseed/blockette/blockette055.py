# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, Loop


class Blockette055(Blockette):
    """
    Blockette 055: Response List Blockette.
    
    This blockette alone is not an acceptable response description; always use 
    this blockette along with the standard response blockettes ([53], [54], 
    [57], or [58]). If this is the only response available, we strongly 
    recommend that you derive the appropriate poles and zeros and include 
    blockette 53 and blockette 58.
    """

    id = 55
    name = "Response list"
    fields = [
        Integer(3, "Stage sequence number", 2),
        Integer(4, "Stage input units", 3),
        Integer(5, "Stage output units", 3),
        Integer(6, "Number of responses listed", 4),
        # REPEAT fields 7 — 11 for the Number of responses listed:
        Loop('Response', "Number of responses listed", [
            Float(7, "Frequency", 12, mask='%+1.5e'),
            Float(8, "Amplitude", 12, mask='%+1.5e'),
            Float(9, "Amplitude error", 12, mask='%+1.5e'),
            Float(10, "Phase angle", 12, mask='%+1.5e'),
            Float(11, "Phase error", 12, mask='%+1.5e')
        ], repeat_title=True)
    ]

    def __init__(self, *args, **kwargs):
        # fix typo for XML-SEED 1.1
        self.XSEED_version = kwargs.get('xseed_version', '1.0')
        if self.XSEED_version == '1.1':
            self.name = "Response list"
        Blockette.__init__(self, *args, **kwargs)
