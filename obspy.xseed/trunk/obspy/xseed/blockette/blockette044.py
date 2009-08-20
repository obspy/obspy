# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, FixedString, Loop
from obspy.xseed.fields import VariableString


class Blockette044(Blockette):
    """
    Blockette 044: Response (Coefficients) Dictionary Blockette.
    
    See Response (Coefficients) Dictionary Blockette [54] for more information.
    """

    id = 44
    name = "Response Coefficients Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        FixedString(5, "Response type", 1, 'U'),
        Integer(6, "Signal input units", 3),
        Integer(7, "Signal output units", 3),
        Integer(8, "Number of numerators", 4),
        # REPEAT fields 9 - 10 for the Number of numerators:
        Loop('Numerators', "Number of numerators", [
            Float(9, "Numerator coefficient", 12, mask='%+1.5e'),
            Float(10, "Numerator error", 12, mask='%+1.5e')
        ], flat=True),
        Integer(11, "Number of denominators", 4),
        # REPEAT fields 12 — 13 for the Number of denominators:
        Loop('Denominators', "Number of denominators", [
            Float(12, "Denominator coefficient", 12, mask='%+1.5e'),
            Float(13, "Denominator error", 12, mask='%+1.5e')
        ], flat=True)
    ]
