from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop, VariableString
from ..utils import blockette_34_lookup, format_resp


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
    def get_xml(self, *args, **kwargs):
        xml = Blockette.get_xml(self, *args, **kwargs)
        if self.xseed_version == '1.0':
            xml.tag = 'response_coefficients'
        return xml

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = \
            '#\t\t+               +----------------------------------------' +\
            '---+                 +\n' + \
            '#\t\t+               |   Response (Coefficients),' + \
            f'{station:>6} ch {channel}   |                 +\n' + \
            '#\t\t+               +----------------------------------------' +\
            '---+                 +\n' + \
            '#\t\t\n' + \
            'B044F05     Response type:                         %s\n' \
            % self.response_type + \
            'B044F06     Response in units lookup:              %s\n'\
            % blockette_34_lookup(abbreviations, self.signal_input_units) + \
            'B044F07     Response out units lookup:             %s\n'\
            % blockette_34_lookup(abbreviations, self.signal_output_units) + \
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
                    string += 'B044F09-10  {:>3} {:>13} {:>13}\n'.format(
                        _i,
                        format_resp(self.numerator_coefficient[_i], 6),
                        format_resp(self.numerator_error[_i], 6))
            else:
                string += 'B044F09-10  {:>3} {:>13} {:>13}\n'.format(
                    0,
                    format_resp(self.numerator_coefficient, 6),
                    format_resp(self.numerator_error, 6))
        if self.number_of_denominators:
            string += \
                '#\t\tDenominator coefficients:\n' + \
                '#\t\t i, coefficient, error\n'
            if self.number_of_denominators > 1:
                # Loop over all zeros.
                for _i in range(self.number_of_numerators):
                    string += 'B044F12-13  {:>3} {:>13} {:>13}\n'.format(
                        _i,
                        format_resp(self.denominator_coefficient[_i], 6),
                        format_resp(self.denominator_error[_i], 6))
            else:
                string += 'B044F12-13  {:>3} {:>13} {:>13}\n'.format(
                    0,
                    format_resp(self.denominator_coefficient, 6),
                    format_resp(self.denominator_error, 6))
        string += '#\t\t\n'
        return string.encode()
