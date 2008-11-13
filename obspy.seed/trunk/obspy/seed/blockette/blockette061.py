# -*- coding: utf-8 -*-

from obspy.seed.blockette import Blockette 
from obspy.seed.fields import Float, Integer, VariableString, SimpleLoop
from obspy.seed.fields import FixedString


class Blockette061(Blockette):
    """Blockette 061: FIR Response Blockette.
    
    The FIR blockette is used to specify FIR (Finite Impulse Response) digital 
    filter coefficients. It is an alternative to blockette [54] when 
    specifying FIR filters. The blockette recognizes the various forms of 
    filter symmetry and can exploit them to reduce the number of factors 
    specified to the blockette. In July 2007, the FDSN adopted a convention 
    that requires the coefficients to be listed in forward time order. 
    As a reference, minimum-phase filters (which are asymmetric) should be
    written with the largest values near the beginning of the coefficient list.
    """
    
    id = 61
    name = "FIR Response"
    fields = [
        Integer(3, "Stage sequence number", 2),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        FixedString(5, "Symmetry Code", 1, 'U'),
        Integer(6, "Signal In Units", 3),
        Integer(7, "Signal Out Units", 3),
        Integer(8, "Number of Coefficients", 4),
        #REPEAT field 9 for the Number of Coefficients
        SimpleLoop("Number of Coefficients", 
            Float(9, "FIR Coefficient", 14, mask='%+1.7e')),
    ]
