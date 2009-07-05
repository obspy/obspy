# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette 
from obspy.xseed.fields import Integer, VariableString, FixedString, Float
from obspy.xseed.fields import SimpleLoop


class Blockette041(Blockette):
    """Blockette 041: FIR Dictionary Blockette.
    
    The FIR blockette is used to specify FIR (Finite Impulse Response)
    digital filter coefficients. It is an alternative to blockette [44] when
    specifying FIR filters. The blockette recognizes the various forms of
    filter symmetry and can exploit them to reduce the number of factors
    specified in the blockette. See Response (Coefficients) Blockette [54]
    for more information.
    """
    
    id = 41
    name = "FIR Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        FixedString(5, "Symmetry Code", 1, 'U'),
        Integer(6, "Signal In Units", 3),
        Integer(7, "Signal Out Units", 3),
        Integer(8, "Number of Factors", 4),
        #REPEAT field 9 for the Number of Factors
        SimpleLoop("Number of Factors", 
            Float(9, "FIR Coefficient", 14, mask='%+1.7e')),
    ]