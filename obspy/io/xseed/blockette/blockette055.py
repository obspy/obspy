# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import Float, Integer, Loop
from ..utils import blockette_34_lookup, format_resp


class Blockette055(Blockette):
    """
    Blockette 055: Response List Blockette.

    This blockette alone is not an acceptable response description; always use
    this blockette along with the standard response blockettes ([53], [54],
    [57], or [58]). If this is the only response available, we strongly
    recommend that you derive the appropriate poles and zeros and include
    blockette 53 and blockette 58.
    """

    id = 55
    # Typo is itentional.
    name = "Response list"
    fields = [
        Integer(3, "Stage sequence number", 2),
        Integer(4, "Stage input units", 3, xpath=34),
        Integer(5, "Stage output units", 3, xpath=34),
        Integer(6, "Number of responses listed", 4),
        # REPEAT fields 7 — 11 for the Number of responses listed:
        Loop('Response', "Number of responses listed", [
            Float(7, "Frequency", 12, mask='%+1.5e'),
            Float(8, "Amplitude", 12, mask='%+1.5e'),
            Float(9, "Amplitude error", 12, mask='%+1.5e'),
            Float(10, "Phase angle", 12, mask='%+1.5e'),
            Float(11, "Phase error", 12, mask='%+1.5e')
        ], repeat_title=True)
    ]

    # Changes the name of the blockette because of an error in XSEED 1.0
    def get_xml(self, *args, **kwargs):
        xml = Blockette.get_xml(self, *args, **kwargs)
        if self.xseed_version == '1.0':
            xml.tag = 'reponse_list'
        return xml

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = (
            '#\t\t+                     +---------------------------------+'
            '                     +\n'
            '#\t\t+                     |   Response List,%6s ch %s   |'
            '                     +\n'
            '#\t\t+                     +---------------------------------+'
            '                     +\n'
            '#\t\t\n'
            'B055F03     Stage sequence number:                 %s\n'

            'B055F04     Response in units lookup:              %s\n'

            'B055F05     Response out units lookup:             %s\n'

            'B055F06     Number of responses:                   %s\n') % (
                station, channel, self.stage_sequence_number,
                blockette_34_lookup(abbreviations, self.stage_input_units),
                blockette_34_lookup(abbreviations, self.stage_output_units),
                self.number_of_responses_listed)

        if self.number_of_responses_listed:
            string += \
                '#\t\tResponses:\n' + \
                '#\t\t  frequency\t amplitude\t amp error\t    ' + \
                'phase\t phase error\n'
            if self.number_of_responses_listed > 1:
                for _i in range(self.number_of_responses_listed):
                    string += 'B055F07-11  %s\t%s\t%s\t%s\t%s\n' % \
                        (format_resp(self.frequency[_i], 6),
                         format_resp(self.amplitude[_i], 6),
                         format_resp(self.amplitude_error[_i], 6),
                         format_resp(self.phase_angle[_i], 6),
                         format_resp(self.phase_error[_i], 6))
            else:
                string += 'B055F07-11  %s\t%s\t%s\t%s\t%s\n' % \
                    (format_resp(self.frequency, 6),
                     format_resp(self.amplitude, 6),
                     format_resp(self.amplitude_error, 6),
                     format_resp(self.phase_angle, 6),
                     format_resp(self.phase_error, 6))
        string += '#\t\t\n'
        return string.encode()
