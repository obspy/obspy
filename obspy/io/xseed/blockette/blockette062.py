# -*- coding: utf-8 -*-
import sys

from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop
from ..utils import blockette_34_lookup, format_resp


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
        Integer(5, "Stage Signal In Units", 3, xpath=34),
        Integer(6, "Stage Signal Out Units", 3, xpath=34),
        FixedString(7, "Polynomial Approximation Type", 1),
        FixedString(8, "Valid Frequency Units", 1),
        Float(9, "Lower Valid Frequency Bound", 12, mask='%+1.5e'),
        Float(10, "Upper Valid Frequency Bound", 12, mask='%+1.5e'),
        Float(11, "Lower Bound of Approximation", 12, mask='%+1.5e'),
        Float(12, "Upper Bound of Approximation", 12, mask='%+1.5e'),
        Float(13, "Maximum Absolute Error", 12, mask='%+1.5e'),
        Integer(14, "Number of Polynomial Coefficients", 3),
        # REPEAT fields 15 and 16 for each polynomial coefficient
        Loop("Polynomial Coefficients", "Number of Polynomial Coefficients", [
            Float(15, "Polynomial Coefficient", 12, mask='%+1.5e'),
            Float(16, "Polynomial Coefficient Error", 12, mask='%+1.5e'),
        ])
    ]

    # Changes the name of the blockette because of an error in XSEED 1.0
    def get_xml(self, *args, **kwargs):
        xml = Blockette.get_xml(self, *args, **kwargs)
        if self.xseed_version == '1.0':
            msg = 'The xsd-validation file for XML-SEED version 1.0 does ' + \
                'not support Blockette 62. It will be written but ' + \
                'please be aware that the file cannot be validated.\n' + \
                'If you want to validate your file please use XSEED ' + \
                'version 1.1.\n'
            sys.stdout.write(msg)
        return xml

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        # Field three needs some extra parsing.
        field_three_dict = {'A': 'A [Laplace Transform (Rad/sec)]',
                            'B': 'B [Analog (Hz)]',
                            'C': 'C [Composite]',
                            'D': 'D [Digital (Z-transform)]',
                            'P': 'P [Polynomial]'}
        # Frequency too!
        frequency_dict = {'A': 'A [rad/sec]',
                          'B': 'B [Hz]'}
        # Polynomial Approximation too.
        polynomial_dict = {'M': 'M [MacLaurin]'}
        string = \
            '#\t\t+              +-----------------------' + \
            '----------------+                      +\n' + \
            '#\t\t+              |   Polynomial response,' + \
            '%6s ch %s   |                      +\n' % (station, channel) + \
            '#\t\t+              +-----------------------' + \
            '----------------+                      +\n' + \
            '#\t\t\n' + \
            'B062F03     Transfer function type:                %s\n' \
            % field_three_dict[self.transfer_function_type] + \
            'B062F04     Stage sequence number:                 %s\n' \
            % self.stage_sequence_number + \
            'B062F05     Response in units lookup:              %s\n' \
            % blockette_34_lookup(abbreviations,
                                  self.stage_signal_in_units) + \
            'B062F06     Response out units lookup:             %s\n' \
            % blockette_34_lookup(abbreviations,
                                  self.stage_signal_out_units) + \
            'B062F07     Polynomial Approximation Type:         %s\n' \
            % polynomial_dict[self.polynomial_approximation_type] + \
            'B062F08     Valid Frequency Units:                 %s\n' \
            % frequency_dict[self.valid_frequency_units] + \
            'B062F09     Lower Valid Frequency Bound:           %G\n' \
            % self.lower_valid_frequency_bound + \
            'B062F10     Upper Valid Frequency Bound:           %G\n' \
            % self.upper_valid_frequency_bound + \
            'B062F11     Lower Bound of Approximation:          %G\n' \
            % self.lower_bound_of_approximation + \
            'B062F12     Upper Bound of Approximation:          %G\n' \
            % self.upper_bound_of_approximation + \
            'B062F13     Maximum Absolute Error:                %G\n' \
            % self.maximum_absolute_error + \
            'B062F14     Number of coefficients:                %d\n' \
            % self.number_of_polynomial_coefficients
        if self.number_of_polynomial_coefficients:
            string += '#\t\tPolynomial coefficients:\n' + \
                '#\t\t  i, coefficient,  error\n'
            if self.number_of_polynomial_coefficients > 1:
                for _i in range(self.number_of_polynomial_coefficients):
                    string += 'B062F15-16   %2s %13s %13s\n' \
                        % (_i, format_resp(self.polynomial_coefficient[_i], 6),
                           format_resp(self.polynomial_coefficient_error[_i],
                                       6))
            else:
                string += 'B062F15-16   %2s %13s %13s\n' \
                    % (0, format_resp(self.polynomial_coefficient, 6),
                       format_resp(self.polynomial_coefficient_error, 6))
        string += '#\t\t\n'
        return string.encode()
