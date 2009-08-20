# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, FixedString, Loop


class Blockette054(Blockette):
    """
    Blockette 054: Response (Coefficients) Blockette.
    
    This blockette is usually used only for finite impulse response (FIR) 
    filter stages. You can express Laplace transforms this way, but you should 
    use the Response (Poles & Zeros) Blockettes [53] for this. You can express 
    IIR filters this way, but you should use the Response (Poles & Zeros) 
    Blockette [53] here, too, to avoid numerical stability problems. Usually, 
    you will follow this blockette with a Decimation Blockette [57] and a 
    Sensitivity/Gain Blockette [58] to complete the definition of the filter 
    stage.
    
    This blockette is the only blockette that might overflow the maximum 
    allowed value of 9,999 characters. If there are more coefficients than fit 
    in one record, list as many as will fit in the first occurrence of this 
    blockette (the counts of Number of numerators and Number of denominators 
    would then be set to the number included, not the total number). In the 
    next record, put the remaining number. Be sure to write and read these 
    blockettes in sequence, and be sure that the first few fields of both 
    records are identical. Reading (and writing) programs have to be able to 
    work with both blockettes as one after reading (or before writing). In 
    July 2007, the FDSN adopted a convention that requires the coefficients to 
    be listed in forward time order. As a reference, minimum-phase filters 
    (which are asymmetric) should be written with the largest values near the 
    beginning of the coefficient list.
    """

    id = 54
    name = "Response Coefficients"
    fields = [
        FixedString(3, "Response type", 1, 'U'),
        Integer(4, "Stage sequence number", 2),
        Integer(5, "Signal input units", 3),
        Integer(6, "Signal output units", 3),
        Integer(7, "Number of numerators", 4),
        # REPEAT fields 8 — 9 for the Number of numerators:
        Loop('Numerators', "Number of numerators", [
            Float(8, "Numerator coefficient", 12, mask='%+1.5e'),
            Float(9, "Numerator error", 12, mask='%+1.5e')
        ], flat=True),
        Integer(10, "Number of denominators", 4),
        # REPEAT fields 11 — 12 for the Number of denominators:
        Loop('Denominators', "Number of denominators", [
            Float(11, "Denominator coefficient", 12, mask='%+1.5e'),
            Float(12, "Denominator error", 12, mask='%+1.5e')
        ], flat=True)
    ]
