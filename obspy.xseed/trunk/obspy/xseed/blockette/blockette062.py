# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import FixedString, Float, Integer, Loop


class Blockette062(Blockette):
    """
    Blockette 062: Response [Polynomial] Blockette.
    
    Use this blockette to characterize the response of a non-linear sensor. 
    The polynomial response blockette describes the output of an Earth sensor 
    in fundamentally a different manner than the other response blockettes. 
    The functional describing the sensor for the polynomial response blockette 
    will have Earth units while the independent variable of the function will 
    be in volts. This is precisely opposite to the other response blockettes. 
    While it is a simple matter to convert a linear response to either form, 
    the non-linear response (which we can describe in the polynomial 
    blockette) would require extensive curve fitting or polynomial inversion 
    to convert from one function to the other. Most data users are interested 
    in knowing the sensor output in Earth units, and the polynomial response 
    blockette facilitates the access to Earth units for sensors with 
    non-linear responses.
    """

    id = 62
    name = "Response Polynomial"
    fields = [
        FixedString(3, "Transfer Function Type", 1),
        Integer(4, "Stage Sequence Number", 2),
        Integer(5, "Stage Signal In Units", 3),
        Integer(6, "Stage Signal Out Units", 3),
        FixedString(7, "Polynomial Approximation Type", 1),
        FixedString(8, "Valid Frequency Units", 1),
        Float(9, "Lower Valid Frequency Bound", 12, mask='%+1.5e'),
        Float(10, "Upper Valid Frequency Bound", 12, mask='%+1.5e'),
        Float(11, "Lower Bound of Approximation", 12, mask='%+1.5e'),
        Float(12, "Upper Bound of Approximation", 12, mask='%+1.5e'),
        Float(13, "Maximum Absolute Error", 12, mask='%+1.5e'),
        Integer(14, "Number of Polynomial Coefficients", 3),
        #REPEAT fields 15 and 16 for each polynomial coefficient
        Loop("Polynomial Coefficients", "Number of Polynomial Coefficients", [
            Float(12, "Polynomial Coefficient", 12, mask='%+1.5e'),
            Float(12, "Polynomial Coefficient Error", 12, mask='%+1.5e'),
        ])
    ]
