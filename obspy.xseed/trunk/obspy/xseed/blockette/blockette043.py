# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, FixedString, Loop
from obspy.xseed.fields import VariableString


class Blockette043(Blockette):
    """Blockette 043: Response (Poles & Zeros) Dictionary Blockette.
    
    See Response (Poles & Zeros) Blockette [53] for more information.
    """

    id = 43
    name = "Response Poles and Zeros"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        FixedString(5, "Response type", 1, 'U'),
        Integer(6, "Stage signal input units", 3),
        Integer(7, "Stage signal output units", 3),
        Float(8, "A0 normalization factor", 12, mask='%+1.5e'),
        Float(9, "Normalization frequency", 12, mask='%+1.5e'),
        Integer(10, "Number of complex zeros", 3),
        # REPEAT fields 11 — 14 for the Number of complex zeros:
        Loop('Complex zero', "Number of complex zeros", [
            Float(11, "Real zero", 12, mask='%+1.5e'),
            Float(12, "Imaginary zero", 12, mask='%+1.5e'),
            Float(13, "Real zero error", 12, mask='%+1.5e'),
            Float(14, "Imaginary zero error", 12, mask='%+1.5e')
        ]),
        Integer(15, "Number of complex poles", 3),
        # REPEAT fields 16 — 19 for the Number of complex poles:
        Loop('Complex pole', "Number of complex poles", [
            Float(16, "Real pole", 12, mask='%+1.5e'),
            Float(16, "Imaginary pole", 12, mask='%+1.5e'),
            Float(18, "Real pole error", 12, mask='%+1.5e'),
            Float(19, "Imaginary pole error", 12, mask='%+1.5e')
        ])
    ]
