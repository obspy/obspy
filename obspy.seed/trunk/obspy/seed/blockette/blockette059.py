# -*- coding: utf-8 -*-

from obspy.seed.blockette import Blockette 
from obspy.seed.fields import Integer, VariableString


class Blockette059(Blockette):
    """Blockette 059: Channel Comment Blockette.
        
    Sample:
    05900351989,001~1989,004~4410000000
    """
    
    id= 59
    name = "Channel Comment Blockette"
    fields = [
        VariableString(3, "Beginning effective time", 1, 22, 'T'),
        VariableString(4, "End effective time", 0, 22, 'T'),
        Integer(5, "Comment code key", 4),
        Integer(6, "Comment level", 6)
    ]
