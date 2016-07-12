# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop
from ..utils import lookup_code, format_resp


RESP = """\
#\t\t+               +--------------------------------------------+           \
     +
#\t\t+               |   Response (Poles & Zeros),%6s ch %s   |               \
 +
#\t\t+               +--------------------------------------------+           \
     +
#\t\t
B053F03     Transfer function type:                %s
B053F04     Stage sequence number:                 %s
B053F05     Response in units lookup:              %s - %s
B053F06     Response out units lookup:             %s - %s
B053F07     A0 normalization factor:               %G
B053F08     Normalization frequency:               %G
B053F09     Number of zeroes:                      %s
B053F14     Number of poles:                       %s
#\t\tComplex zeroes:
#\t\t  i  real          imag          real_error    imag_error
"""


class Blockette053(Blockette):
    """
    Blockette 053: Response (Poles & Zeros) Blockette.

    Sample::

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
    name = "Response Poles and Zeros"
    fields = [
        FixedString(3, "Transfer function types", 1, 'U'),
        Integer(4, "Stage sequence number", 2),
        Integer(5, "Stage signal input units", 3, xpath=34),
        Integer(6, "Stage signal output units", 3, xpath=34),
        Float(7, "A0 normalization factor", 12, mask='%+1.5e'),
        Float(8, "Normalization frequency", 12, mask='%+1.5e'),
        Integer(9, "Number of complex zeros", 3),
        # REPEAT fields 10 — 13 for the Number of complex zeros:
        Loop('Complex zero', "Number of complex zeros", [
            Float(10, "Real zero", 12, mask='%+1.5e'),
            Float(11, "Imaginary zero", 12, mask='%+1.5e'),
            Float(12, "Real zero error", 12, mask='%+1.5e'),
            Float(13, "Imaginary zero error", 12, mask='%+1.5e')
        ]),
        Integer(14, "Number of complex poles", 3),
        # REPEAT fields 15 — 18 for the Number of complex poles:
        Loop('Complex pole', "Number of complex poles", [
            Float(15, "Real pole", 12, mask='%+1.5e'),
            Float(16, "Imaginary pole", 12, mask='%+1.5e'),
            Float(17, "Real pole error", 12, mask='%+1.5e'),
            Float(18, "Imaginary pole error", 12, mask='%+1.5e')
        ])
    ]

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        # Field three needs some extra parsing.
        field_three_dict = {'A': 'A [Laplace Transform (Rad/sec)]',
                            'B': 'B [Analog (Hz)]',
                            'C': 'C [Composite]',
                            'D': 'D [Digital (Z-transform)]'}
        out = RESP % (station, channel,
                      field_three_dict[self.transfer_function_types],
                      self.stage_sequence_number,
                      lookup_code(abbreviations, 34, 'unit_name',
                                  'unit_lookup_code',
                                  self.stage_signal_input_units),
                      lookup_code(abbreviations, 34, 'unit_description',
                                  'unit_lookup_code',
                                  self.stage_signal_input_units),
                      lookup_code(abbreviations, 34, 'unit_name',
                                  'unit_lookup_code',
                                  self.stage_signal_output_units),
                      lookup_code(abbreviations, 34, 'unit_description',
                                  'unit_lookup_code',
                                  self.stage_signal_output_units),
                      self.A0_normalization_factor,
                      self.normalization_frequency,
                      self.number_of_complex_zeros,
                      self.number_of_complex_poles)
        if self.number_of_complex_zeros > 0:
            if self.number_of_complex_zeros != 1:
                # Loop over all zeros.
                for _i in range(self.number_of_complex_zeros):
                    out += 'B053F10-13 %4s %13s %13s %13s %13s\n' % (
                        _i,
                        format_resp(self.real_zero[_i], 6),
                        format_resp(self.imaginary_zero[_i], 6),
                        format_resp(self.real_zero_error[_i], 6),
                        format_resp(self.imaginary_zero_error[_i], 6))
            else:
                out += 'B053F10-13 %4s %13s %13s %13s %13s\n' % (
                    0,
                    format_resp(self.real_zero, 6),
                    format_resp(self.imaginary_zero, 6),
                    format_resp(self.real_zero_error, 6),
                    format_resp(self.imaginary_zero_error, 6))
        out += '#\t\tComplex poles:\n'
        out += '#\t\t  i  real          imag          real_error    '
        out += 'imag_error\n'
        if self.number_of_complex_poles > 0:
            if self.number_of_complex_poles != 1:
                # Loop over all poles.
                for _i in range(self.number_of_complex_poles):
                    out += 'B053F15-18 %4s %13s %13s %13s %13s\n' % (
                        _i,
                        format_resp(self.real_pole[_i], 6),
                        format_resp(self.imaginary_pole[_i], 6),
                        format_resp(self.real_pole_error[_i], 6),
                        format_resp(self.imaginary_pole_error[_i], 6))
            else:
                out += 'B053F15-18 %4s %13s %13s %13s %13s\n' % (
                    0,
                    format_resp(self.real_pole, 6),
                    format_resp(self.imaginary_pole, 6),
                    format_resp(self.real_pole_error, 6),
                    format_resp(self.imaginary_pole_error, 6))
        out += '#\t\t\n'
        return out.encode()
