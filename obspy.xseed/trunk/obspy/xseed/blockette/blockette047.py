# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, VariableString


class Blockette047(Blockette):
    """
    Blockette 047: Decimation Dictionary Blockette.
    
    See Decimation Blockette [57] for more information.
    """

    id = 47
    name = "Decimation"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        Float(5, "Input sample rate", 10, mask='%1.4e'),
        Integer(6, "Decimation factor", 5),
        Integer(7, "Decimation offset", 5),
        Float(8, "Estimated delay", 11, mask='%+1.4e'),
        Float(9, "Correction applied", 11, mask='%+1.4e')
    ]
