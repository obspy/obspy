# -*- coding: utf-8 -*-
from .blockette import Blockette
from ..fields import Float, Integer, VariableString
from ..utils import format_resp


class Blockette047(Blockette):
    """
    Blockette 047: Decimation Dictionary Blockette.

    See Decimation Blockette [57] for more information.
    """

    id = 47
    name = "Decimation Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        Float(5, "Input sample rate", 10, mask='%1.4e'),
        Integer(6, "Decimation factor", 5, xseed_version='1.0',
                xml_tag="decimiation_factor"),
        Integer(6, "Decimation factor", 5, xseed_version='1.1'),
        Integer(7, "Decimation offset", 5),
        Float(8, "Estimated delay", 11, mask='%+1.4e'),
        Float(9, "Correction applied", 11, mask='%+1.4e')
    ]

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = \
            '#\t\t+                      +------------------------------+' + \
            '                       +\n' + \
            '#\t\t+                      |   Decimation,' + \
            '%6s ch %s   |                       +\n' % (station, channel) + \
            '#\t\t+                      +------------------------------+' + \
            '                       +\n' + \
            '#\t\t\n' + \
            'B047F05     Response input sample rate:            %s\n' \
            % format_resp(self.input_sample_rate, 6) + \
            'B047F06     Response decimation factor:            %s\n' \
            % self.decimation_factor + \
            'B047F07     Response decimation offset:            %s\n' \
            % self.decimation_offset + \
            'B047F08     Response delay:                        %s\n' \
            % format_resp(self.estimated_delay, 6) + \
            'B047F09     Response correction:                   %s\n' \
            % format_resp(self.correction_applied, 6) + \
            '#\t\t\n'
        return string.encode()
