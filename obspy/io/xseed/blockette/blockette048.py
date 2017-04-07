# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import Float, Integer, Loop, VariableString
from ..utils import format_resp


class Blockette048(Blockette):
    """
    Blockette 048: Channel Sensitivity/Gain Dictionary Blockette.

    See Channel Sensitivity/Gain Blockette [58] for more information.
    """

    id = 48
    name = "Channel Sensivitity Gain Dictionary"
    fields = [
        Integer(3, "Response Lookup Key", 4),
        VariableString(4, "Response Name", 1, 25, 'UN_'),
        Float(5, "Sensitivity gain", 12, mask='%+1.5e'),
        Float(6, "Frequency", 12, mask='%+1.5e'),
        Integer(7, "Number of history values", 2),
        # REPEAT fields 8 — 10 for the Number of history values:
        Loop('History', "Number of history values", [
            Float(8, "Sensitivity for calibration", 12, mask='%+1.5e'),
            Float(9, "Frequency of calibration sensitivity", 12,
                  mask='%+1.5e'),
            VariableString(10, "Time of above calibration", 1, 22, 'T')
        ])
    ]

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        string = \
            '#\t\t+                  ' + \
            '+---------------------------------------+' + \
            '                  +\n' + \
            '#\t\t+                  |   Channel Sensitivity,' + \
            '%6s ch %s   |                  +\n' % (station, channel) + \
            '#\t\t+                  ' + \
            '+---------------------------------------+' + \
            '                  +\n' + \
            '#\t\t\n' + \
            'B048F05     Sensitivity:                           %s\n' \
            % format_resp(self.sensitivity_gain, 6) + \
            'B048F06     Frequency of sensitivity:              %s\n' \
            % format_resp(self.frequency, 6) + \
            'B048F07     Number of calibrations:                %s\n' \
            % self.number_of_history_values
        if self.number_of_history_values > 1:
            string += \
                '#\t\tCalibrations:\n' + \
                '#\t\t i, sensitivity, frequency, time of calibration\n'
            for _i in range(self.number_of_history_values):
                string += \
                    'B048F08-09   %2s %13s %13s %s\n' \
                    % (format_resp(self.sensitivity_for_calibration[_i], 6),
                       format_resp(
                           self.frequency_of_calibration_sensitivity[_i], 6),
                       self.time_of_above_calibration[_i].format_seed())
        elif self.number_of_history_values == 1:
            string += \
                '#\t\tCalibrations:\n' + \
                '#\t\t i, sensitivity, frequency, time of calibration\n' + \
                'B048F08-09    0 %13s %13s %s\n' % (
                    format_resp(self.sensitivity_for_calibration, 6),
                    format_resp(self.frequency_of_calibration_sensitivity, 6),
                    self.time_of_above_calibration.format_seed())
        string += '#\t\t\n'
        return string.encode()
