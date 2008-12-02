# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette 
from obspy.xseed.fields import Integer, VariableString


class Blockette051(Blockette):
    """Blockette 051: Station Comment Blockette.
        
    Sample:
    05100351992,001~1992,002~0740000000
    """
    
    id= 51
    name = "Station Comment Blockette"
    fields = [
        VariableString(3, "Beginning effective time", 1, 22, 'T'),
        VariableString(4, "End effective time", 1, 22, 'T'),
        Integer(5, "Comment code key", 4),
        Integer(6, "Comment level", 6)
    ]
