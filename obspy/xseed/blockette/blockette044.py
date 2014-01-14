# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from future.builtins import range

from obspy.xseed.blockette import Blockette
from obspy.xseed.fields import Float, Integer, FixedString, Loop
from obspy.xseed.fields import VariableString
from obspy.xseed.utils import Blockette34Lookup, formatRESP


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
        Integer(6, "Signal input units", 3, xpath=34),
        Integer(7, "Signal output units", 3, xpath=34),
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

# Changes the name of the blockette because of an error in XSEED 1.0
    def getXML(self, *args, **kwargs):
        xml = Blockette.getXML(self, *args, **kwargs)
        if self.xseed_version == '1.0':
            xml.tag = 'response_coefficients'
        return xml

    def getRESP(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = \
            '#\t\t+               +----------------------------------------' +\
            '---+                 +\n' + \
            '#\t\t+               |   Response (Coefficients),' + \
            '%6s ch %s   |                 +\n' % (station, channel) + \
            '#\t\t+               +----------------------------------------' +\
            '---+                 +\n' + \
            '#\t\t\n' + \
            'B044F05     Response type:                         %s\n' \
            % self.response_type + \
            'B044F06     Response in units lookup:              %s\n'\
            % Blockette34Lookup(abbreviations, self.signal_input_units) + \
            'B044F07     Response out units lookup:             %s\n'\
            % Blockette34Lookup(abbreviations, self.signal_output_units) + \
            'B044F08     Number of numerators:                  %s\n' \
            % self.number_of_numerators + \
            'B044F11     Number of denominators:                %s\n' \
            % self.number_of_denominators + \
            '#\t\tNumerator coefficients:\n' + \
            '#\t\t  i, coefficient,  error\n'
        if self.number_of_numerators:
            string += \
                '#\t\tNumerator coefficients:\n' + \
                '#\t\t  i, coefficient,  error\n'
            if self.number_of_numerators > 1:
                # Loop over all zeros.
                for _i in range(self.number_of_numerators):
                    string += 'B044F09-10  %3s %13s %13s\n' % (
                        _i,
                        formatRESP(self.numerator_coefficient[_i], 6),
                        formatRESP(self.numerator_error[_i], 6))
            else:
                string += 'B044F09-10  %3s %13s %13s\n' % (
                    0,
                    formatRESP(self.numerator_coefficient, 6),
                    formatRESP(self.numerator_error, 6))
        if self.number_of_denominators:
            string += \
                '#\t\tDenominator coefficients:\n' + \
                '#\t\t i, coefficient, error\n'
            if self.number_of_denominators > 1:
                # Loop over all zeros.
                for _i in range(self.number_of_numerators):
                    string += 'B044F12-13  %3s %13s %13s\n' % (
                        _i,
                        formatRESP(self.denominator_coefficient[_i], 6),
                        formatRESP(self.denominator_error[_i], 6))
            else:
                string += 'B044F12-13  %3s %13s %13s\n' % (
                    0,
                    formatRESP(self.denominator_coefficient, 6),
                    formatRESP(self.denominator_error, 6))
        string += '#\t\t\n'
        return string
