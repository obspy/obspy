# -*- coding: utf-8 -*-
from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop, VariableString
from ..utils import blockette_34_lookup, format_resp


class Blockette043(Blockette):
    """
    Blockette 043: Response (Poles & Zeros) Dictionary Blockette.

    See Response (Poles & Zeros) Blockette [53] for more information.
    """

    id = 43
    name = "Response Poles and Zeros Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        FixedString(5, "Response type", 1, 'U'),
        Integer(6, "Stage signal input units", 3, xpath=34),
        Integer(7, "Stage signal output units", 3, xpath=34),
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
            Float(17, "Imaginary pole", 12, mask='%+1.5e'),
            Float(18, "Real pole error", 12, mask='%+1.5e'),
            Float(19, "Imaginary pole error", 12, mask='%+1.5e')
        ])
    ]

# Changes the name of the blockette because of an error in XSEED 1.0
    def get_xml(self, *args, **kwargs):
        xml = Blockette.get_xml(self, *args, **kwargs)
        if self.xseed_version == '1.0':
            xml.tag = 'response_poles_and_zeros'
        return xml

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        # Field five needs some extra parsing.
        field_five_dict = {'A': 'A [Laplace Transform (Rad/sec)]',
                           'B': 'B [Analog (Hz)]',
                           'C': 'C [Composite]',
                           'D': 'D [Digital (Z-transform)]'}
        string = \
            '#\t\t+               ' + \
            '+-----------------------------------------' + \
            '---+                +\n' + \
            '#\t\t+               |   Response (Poles & Zeros),' + \
            '%6s ch %s   |                +\n' % (station, channel) + \
            '#\t\t+               ' + \
            '+-----------------------------------------' + \
            '---+                +\n' + \
            '#\t\t\n' + \
            'B043F05     Response type:                         %s\n' \
            % field_five_dict[self.response_type] + \
            'B043F06     Response in units lookup:              %s\n' \
            % blockette_34_lookup(abbreviations,
                                  self.stage_signal_input_units) + \
            'B043F07     Response out units lookup:             %s\n' \
            % blockette_34_lookup(abbreviations,
                                  self.stage_signal_output_units) + \
            'B043F08     A0 normalization factor:               %G\n'\
            % self.A0_normalization_factor + \
            'B043F09     Normalization frequency:               %G\n'\
            % self.normalization_frequency + \
            'B043F10     Number of zeroes:                      %s\n'\
            % self.number_of_complex_zeros + \
            'B043F15     Number of poles:                       %s\n'\
            % self.number_of_complex_poles + \
            '#\t\tComplex zeroes:\n' + \
            '#\t\t  i  real          imag          real_error    imag_error\n'
        if self.number_of_complex_zeros > 0:
            if self.number_of_complex_zeros != 1:
                # Loop over all zeros.
                for _i in range(self.number_of_complex_zeros):
                    string += 'B043F11-14 %4s %13s %13s %13s %13s\n' % (
                        _i,
                        format_resp(self.real_zero[_i], 6),
                        format_resp(self.imaginary_zero[_i], 6),
                        format_resp(self.real_zero_error[_i], 6),
                        format_resp(self.imaginary_zero_error[_i], 6))
            else:
                string += 'B043F11-14 %4s %13s %13s %13s %13s\n' % (
                    0,
                    format_resp(self.real_zero, 6),
                    format_resp(self.imaginary_zero, 6),
                    format_resp(self.real_zero_error, 6),
                    format_resp(self.imaginary_zero_error, 6))
        string += '#\t\tComplex poles:\n' + \
            '#\t\t  i  real          imag          real_error    imag_error\n'
        if self.number_of_complex_poles > 0:
            if self.number_of_complex_poles != 1:
                # Loop over all poles.
                for _i in range(self.number_of_complex_poles):
                    string += 'B043F16-19 %4s %13s %13s %13s %13s\n' % (
                        _i,
                        format_resp(self.real_pole[_i], 6),
                        format_resp(self.imaginary_pole[_i], 6),
                        format_resp(self.real_pole_error[_i], 6),
                        format_resp(self.imaginary_pole_error[_i], 6))
            else:
                string += 'B043F16-19 %4s %13s %13s %13s %13s\n' % (
                    0,
                    format_resp(self.real_pole, 6),
                    format_resp(self.imaginary_pole, 6),
                    format_resp(self.real_pole_error, 6),
                    format_resp(self.imaginary_pole_error, 6))
        string += '#\t\t\n'
        return string.encode()
