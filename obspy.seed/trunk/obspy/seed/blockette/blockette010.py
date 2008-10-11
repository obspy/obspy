# -*- coding: utf-8 -*-

from obspy.seed.blockette import Blockette 
from obspy.seed.fields import Float, Integer, VariableString


class Blockette010(Blockette):
    """Blockette 010: Volume Identifier Blockette.
        
    This is the normal header blockette for station or event oriented network 
    volumes. Include it once at the beginning of each logical volume or sub-
    volume.
    
    Sample:
    010009502.1121992,001,00:00:00.0000~1992,002,00:00:00.0000~1993,029~
    IRIS _ DMC~Data for 1992,001~
    """
    
    id = 10
    name = "Volume Identifier Blockette"
    fields = [
        Float(3, "Version of format", 4),
        Integer(4, "Logical record length", 4),
        VariableString(5, "Beginning time", 1, 22, 'T'),
        VariableString(6, "End time", 1, 22, 'T'),
        VariableString(7, "Volume Time", 1, 22, 'T', version=2.3),
        VariableString(8, "Originating Organization", 1, 80, version=2.3),
        VariableString(9, "Label", 1, 80, version=2.3)
    ]
