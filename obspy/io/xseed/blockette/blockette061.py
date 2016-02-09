# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop, VariableString
from ..utils import lookup_code, format_resp


RESP = """\
#\t\t+                     +--------------------------------+                 \
     +
#\t\t+                     |   FIR response,%6s ch %s   |                     \
 +
#\t\t+                     +--------------------------------+                 \
     +
#\t\t
B061F03     Stage sequence number:                 %s
B061F05     Symmetry type:                         %s
B061F06     Response in units lookup:              %s - %s
B061F07     Response out units lookup:             %s - %s
B061F08     Number of numerators:                  %s
"""


class Blockette061(Blockette):
    """
    Blockette 061: FIR Response Blockette.

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
        Integer(6, "Signal In Units", 3, xpath=34),
        Integer(7, "Signal Out Units", 3, xpath=34),
        Integer(8, "Number of Coefficients", 4),
        # REPEAT field 9 for the Number of Coefficients
        Loop("FIR Coefficient", "Number of Coefficients", [
            Float(9, "FIR Coefficient", 14, mask='%+1.7e')], flat=True),
    ]

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        out = RESP % (station, channel,
                      self.stage_sequence_number,
                      self.symmetry_code,
                      lookup_code(abbreviations, 34, 'unit_name',
                                  'unit_lookup_code', self.signal_in_units),
                      lookup_code(abbreviations, 34, 'unit_description',
                                  'unit_lookup_code', self.signal_in_units),
                      lookup_code(abbreviations, 34, 'unit_name',
                                  'unit_lookup_code', self.signal_out_units),
                      lookup_code(abbreviations, 34, 'unit_description',
                                  'unit_lookup_code', self.signal_out_units),
                      self.number_of_coefficients)
        if self.number_of_coefficients > 1:
            out += '#\t\tNumerator coefficients:\n'
            out += '#\t\t  i, coefficient\n'
            for _i in range(self.number_of_coefficients):
                out += 'B061F09    %4s %13s\n' % \
                    (_i, format_resp(self.FIR_coefficient[_i], 6))
        elif self.number_of_coefficients == 1:
            out += '#\t\tNumerator coefficients:\n'
            out += '#\t\t  i, coefficient\n'
            out += 'B061F09    %4s %13s\n' % \
                (0, format_resp(self.FIR_coefficient, 6))
        out += '#\t\t\n'
        return out.encode()
