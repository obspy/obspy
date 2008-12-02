# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette 
from obspy.xseed.fields import Float, Integer, FixedString, MultipleLoop


class Blockette053(Blockette):
    """Blockette 053: Response (Poles & Zeros) Blockette.
    
    Sample:
    0530382B 1007008 7.87395E+00 5.00000E-02  3
     0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
     0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
    -1.27000E+01 0.00000E+00 0.00000E+00 0.00000E+00  4
    -1.96418E-03 1.96418E-03 0.00000E+00 0.00000E+00
    S-1.96418E-03-1.96418E-03 0.00000E+00 0.00000E+00
    53-6.23500E+00 7.81823E+00 0.00000E+00 0.00000E+00
    -6.23500E+00-7.81823E+00 0.00000E+00 0.00000E+00
    """
    
    id = 53
    name = "Response Poles and Zeros Blockette"
    fields = [
        FixedString(3, "Transfer function type", 1, 'U'),
        Integer(4, "Stage sequence number", 2),
        Integer(5, "Stage signal input units", 3),
        Integer(6, "Stage signal output units", 3),
        Float(7, "AO normalization factor", 12, mask='%+1.5e'),
        Float(8, "Normalization frequency fn", 12, mask='%+1.5e'),
        Integer(9, "Number of complex zeros", 3),
        # REPEAT fields 10 — 13 for the Number of complex zeros:
        MultipleLoop('Complex zeros', "Number of complex zeros", [
            Float(10, "Real zero", 12, mask='%+1.5e'),
            Float(11, "Imaginary zero", 12, mask='%+1.5e'),
            Float(12, "Real zero error", 12, mask='%+1.5e'),
            Float(13, "Imaginary zero error", 12, mask='%+1.5e')
        ]),
        Integer(14, "Number of complex poles", 3),
        # REPEAT fields 15 — 18 for the Number of complex poles:
        MultipleLoop('Complex poles', "Number of complex poles", [
            Float(15, "Real pole", 12, mask='%+1.5e'),
            Float(16, "Imaginary pole", 12, mask='%+1.5e'),
            Float(17, "Real pole error", 12, mask='%+1.5e'),
            Float(18, "Imaginary pole error", 12, mask='%+1.5e')
        ])
    ]
