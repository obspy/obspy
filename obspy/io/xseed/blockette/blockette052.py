# -*- coding: utf-8 -*-
from .blockette import Blockette
from ..fields import FixedString, Float, Integer, VariableString


class Blockette052(Blockette):
    """
    Blockette 052: Channel Identifier Blockette.

    Sample:
    0520119  BHE0000004~001002+34.946200-106.456700+1740.0100.0090.0+00.0000112
    2.000E+01 2.000E-030000CG~1991,042,20:48~~N
    """

    id = 52
    name = "Channel Identifier"
    fields = [
        FixedString(3, "Location identifier", 2, 'UN'),
        FixedString(4, "Channel identifier", 3, 'UN'),
        Integer(5, "Subchannel identifier", 4),
        Integer(6, "Instrument identifier", 3, xpath=33),
        VariableString(7, "Optional comment", 0, 30, 'UNLPS'),
        Integer(8, "Units of signal response", 3, xpath=34),
        Integer(9, "Units of calibration input", 3, xpath=34),
        Float(10, "Latitude", 10, mask='%+2.6f'),
        Float(11, "Longitude", 11, mask='%+3.6f'),
        Float(12, "Elevation", 7, mask='%+4.1f'),
        Float(13, "Local depth", 5, mask='%3.1f'),
        Float(14, "Azimuth", 5, mask='%3.1f'),
        Float(15, "Dip", 5, mask='%+2.1f'),
        Integer(16, "Data format identifier code", 4, xpath=30),
        # The typo is intentional for XSEED 1.0 compatibility.
        Integer(17, "Data record length", 2, xseed_version='1.0',
                xml_tag="data_recored_length"),
        Integer(17, "Data record length", 2, xseed_version='1.1'),
        Float(18, "Sample rate", 10, mask='%1.4e'),
        Float(19, "Max clock drift", 10, mask='%1.4e'),
        Integer(20, "Number of comments", 4),
        VariableString(21, "Channel flags", 0, 26, 'U'),
        VariableString(22, "Start date", 1, 22, 'T'),
        VariableString(23, "End date", 0, 22, 'T', optional=True,
                       xseed_version='1.0'),
        VariableString(23, "End date", 0, 22, 'T', xseed_version='1.1'),
        FixedString(24, "Update flag", 1)
    ]
