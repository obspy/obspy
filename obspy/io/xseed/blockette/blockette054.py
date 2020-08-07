# -*- coding: utf-8 -*-
from .blockette import Blockette
from ..fields import FixedString, Float, Integer, Loop
from ..utils import blockette_34_lookup, format_resp


class Blockette054(Blockette):
    """
    Blockette 054: Response (Coefficients) Blockette.

    This blockette is usually used only for finite impulse response (FIR)
    filter stages. You can express Laplace transforms this way, but you should
    use the Response (Poles & Zeros) Blockettes [53] for this. You can express
    IIR filters this way, but you should use the Response (Poles & Zeros)
    Blockette [53] here, too, to avoid numerical stability problems. Usually,
    you will follow this blockette with a Decimation Blockette [57] and a
    Sensitivity/Gain Blockette [58] to complete the definition of the filter
    stage.

    This blockette is the only blockette that might overflow the maximum
    allowed value of 9,999 characters. If there are more coefficients than fit
    in one record, list as many as will fit in the first occurrence of this
    blockette (the counts of Number of numerators and Number of denominators
    would then be set to the number included, not the total number). In the
    next record, put the remaining number. Be sure to write and read these
    blockettes in sequence, and be sure that the first few fields of both
    records are identical. Reading (and writing) programs have to be able to
    work with both blockettes as one after reading (or before writing). In
    July 2007, the FDSN adopted a convention that requires the coefficients to
    be listed in forward time order. As a reference, minimum-phase filters
    (which are asymmetric) should be written with the largest values near the
    beginning of the coefficient list.
    """

    id = 54
    name = "Response Coefficients"
    fields = [
        FixedString(3, "Response type", 1, 'U'),
        Integer(4, "Stage sequence number", 2),
        Integer(5, "Signal input units", 3, xpath=34),
        Integer(6, "Signal output units", 3, xpath=34),
        Integer(7, "Number of numerators", 4),
        # REPEAT fields 8 — 9 for the Number of numerators:
        Loop('Numerators', "Number of numerators", [
            Float(8, "Numerator coefficient", 12, mask='%+1.5e'),
            Float(9, "Numerator error", 12, mask='%+1.5e')
        ], flat=True),
        Integer(10, "Number of denominators", 4),
        # REPEAT fields 11 — 12 for the Number of denominators:
        Loop('Denominators', "Number of denominators", [
            Float(11, "Denominator coefficient", 12, mask='%+1.5e'),
            Float(12, "Denominator error", 12, mask='%+1.5e')
        ], flat=True)
    ]

    def get_resp(self, station, channel, abbreviations):
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
            'B054F03     Transfer function type:                %s\n' \
            % self.response_type + \
            'B054F04     Stage sequence number:                 %s\n' \
            % self.stage_sequence_number + \
            'B054F05     Response in units lookup:              %s\n'\
            % blockette_34_lookup(abbreviations, self.signal_input_units) +\
            'B054F06     Response out units lookup:             %s\n'\
            % blockette_34_lookup(abbreviations, self.signal_output_units) +\
            'B054F07     Number of numerators:                  %s\n' \
            % self.number_of_numerators + \
            'B054F10     Number of denominators:                %s\n' \
            % self.number_of_denominators
        if self.number_of_numerators:
            string += \
                '#\t\tNumerator coefficients:\n' + \
                '#\t\t  i, coefficient,  error\n'
            if self.number_of_numerators > 1:
                # Loop over all zeros.
                for _i in range(self.number_of_numerators):
                    string += 'B054F08-09  %3s %13s %13s\n' % (
                        _i,
                        format_resp(self.numerator_coefficient[_i], 6),
                        format_resp(self.numerator_error[_i], 6))
            else:
                string += 'B054F08-09  %3s %13s %13s\n' % (
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
                    string += 'B054F11-12  %3s %13s %13s\n' % (
                        _i,
                        format_resp(self.denominator_coefficient[_i], 6),
                        format_resp(self.denominator_error[_i], 6))
            else:
                string += 'B054F11-12  %3s %13s %13s\n' % (
                    0,
                    format_resp(self.denominator_coefficient, 6),
                    format_resp(self.denominator_error, 6))
        string += '#\t\t\n'
        return string.encode()
