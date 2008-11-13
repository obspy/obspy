# -*- coding: utf-8 -*-

from obspy.seed.blockette import Blockette 
from obspy.seed.fields import Integer, VariableString, FixedString, Float


class Blockette050(Blockette):
    """Blockette 050: Station Identifier Blockette.
    
    Sample:
    0500097ANMO +34.946200-106.456700+1740.00006001Albuquerque, NewMexico, USA~
    0013210101989,241~~NIU
    """
    
    id = 50
    name = "Station Identifier Blockette"
    fields = [
        FixedString(3, "Station call letters", 5, 'UN'),
        Float(4, "Latitude", 10, mask='%+02.6f'),
        Float(5, "Longitude", 11, mask='%+03.6f'),
        Float(6, "Elevation", 7, mask='%+04.1f'),
        Integer(7, "Number of channels", 4, optional=True),
        Integer(8, "Number of station comments", 3, optional=True),
        VariableString(9, "Site name", 1, 60, 'UNLPS'),
        Integer(10, "Network identifier code", 3),
        Integer(11, "word order 32 bit", 4),
        Integer(12, "word order 16 bit", 2),
        VariableString(13, "Start effective date", 1, 22, 'T'),
        VariableString(14, "End effective date", 0, 22, 'T'),
        FixedString(15, "Update flag", 1),
        FixedString(16, "Network Code", 2, 'ULN', version=2.3)
    ]

