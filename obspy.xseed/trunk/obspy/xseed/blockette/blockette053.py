# -*- coding: utf-8 -*-

from obspy.xseed.blockette import Blockette 
from obspy.xseed.fields import Float, Integer, FixedString, Loop
from obspy.xseed.utils import LookupCode, formatRESP


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
    name = "Response Poles and Zeros"
    fields = [
        FixedString(3, "Transfer function types", 1, 'U'),
        Integer(4, "Stage sequence number", 2),
        Integer(5, "Stage signal input units", 3),
        Integer(6, "Stage signal output units", 3),
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

    def getRESP(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        # Field three needs some extra parsing.
        field_three_dict = {'A' : 'A [Laplace Transform (Rad/sec)]',
                            'B' : 'B [Analog (Hz)]',
                            'C' : 'C [Composite]',
                            'D' : 'D [Digital (Z-transform)]'}
        string = \
        '#\t\t+               +--------------------------------------------+                +\n' + \
        '#\t\t+               |   Response (Poles & Zeros),%6s ch %s   |                +\n' \
                    %(station, channel) + \
        '#\t\t+               +--------------------------------------------+                +\n' + \
        '#\t\t\n' + \
        'B053F03     Transfer function type:                %s\n' \
                    % field_three_dict[self.transfer_function_types] + \
        'B053F04     Stage sequence number:                 %s\n' \
                    % self.stage_sequence_number + \
        'B053F05     Response in units lookup:              %s - %s\n'\
            %(LookupCode(abbreviations, 34, 'unit_name', 'unit_lookup_code',
                         self.stage_signal_input_units),
              LookupCode(abbreviations, 34, 'unit_description',
                    'unit_lookup_code', self.stage_signal_input_units))  + \
        'B053F06     Response out units lookup:             %s - %s\n'\
            % (LookupCode(abbreviations, 34, 'unit_name', 'unit_lookup_code',
                         self.stage_signal_output_units),
              LookupCode(abbreviations, 34, 'unit_description',
                    'unit_lookup_code', self.stage_signal_output_units))  + \
        'B053F07     A0 normalization factor:               %G\n'\
            % self.A0_normalization_factor + \
        'B053F08     Normalization frequency:               %G\n'\
            % self.normalization_frequency + \
        'B053F09     Number of zeroes:                      %s\n'\
            % self.number_of_complex_zeros + \
        'B053F14     Number of poles:                       %s\n'\
            % self.number_of_complex_poles + \
        '#\t\tComplex zeroes:\n' + \
        '#\t\t  i  real          imag          real_error    imag_error\n'
        if self.number_of_complex_zeros > 0:
            if self.number_of_complex_zeros != 1:
                # Loop over all zeros.
                for _i in range(self.number_of_complex_zeros):
                    string += 'B053F10-13 %4s %13s %13s %13s %13s\n' % (_i,
                            formatRESP(self.real_zero[_i], 6),
                            formatRESP(self.imaginary_zero[_i], 6),
                            formatRESP(self.real_zero_error[_i], 6),
                            formatRESP(self.imaginary_zero_error[_i], 6))
            else:
                string += 'B053F10-13 %4s %13s %13s %13s %13s\n' % (0,
                            formatRESP(self.real_zero, 6),
                            formatRESP(self.imaginary_zero, 6),
                            formatRESP(self.real_zero_error, 6),
                            formatRESP(self.imaginary_zero_error, 6))
        string += '#\t\tComplex poles:\n' + \
        '#\t\t  i  real          imag          real_error    imag_error\n'
        if self.number_of_complex_poles > 0:
            if self.number_of_complex_poles != 1:
                # Loop over all poles.
                for _i in range(self.number_of_complex_poles):
                    string += 'B053F15-18 %4s %13s %13s %13s %13s\n' % (_i,
                            formatRESP(self.real_pole[_i], 6),
                            formatRESP(self.imaginary_pole[_i], 6),
                            formatRESP(self.real_pole_error[_i], 6),
                            formatRESP(self.imaginary_pole_error[_i], 6))
            else:
                string += 'B053F15-18 %4s %13s %13s %13s %13s\n' % (0,
                            formatRESP(self.real_pole, 6),
                            formatRESP(self.imaginary_pole, 6),
                            formatRESP(self.real_pole_error, 6),
                            formatRESP(self.imaginary_pole_error, 6))
        string += '#\t\t\n'
        return string