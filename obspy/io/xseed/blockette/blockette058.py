# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import Float, Integer, Loop, VariableString
from ..utils import format_resp


class Blockette058(Blockette):
    """
    Blockette 058: Channel Sensitivity/Gain Blockette.

    When used as a gain (stage ≠ 0), this blockette is the gain for this stage
    at the given frequency. Different stages may be at different frequencies.
    However, it is strongly recommended that the same frequency be used in all
    stages of a cascade, if possible. When used as a sensitivity(stage=0),
    this blockette is the sensitivity (in counts per ground motion) for the
    entire channel at a given frequency, and is also referred to as the
    overall gain. The frequency here may be different from the frequencies in
    the gain specifications, but should be the same if possible. If you use
    cascading (more than one filter stage), then SEED requires a gain for each
    stage. A final sensitivity (Blockette [58], stage = 0, is required. If you
    do not use cascading (only one stage), then SEED must see a gain, a
    sensitivity, or both.

    Sample:
    0580035 3 3.27680E+03 0.00000E+00 0
    """

    id = 58
    name = "Channel Sensitivity Gain"
    fields = [
        Integer(3, "Stage sequence number", 2),
        Float(4, "Sensitivity gain", 12, mask='%+1.5e'),
        Float(5, "Frequency", 12, mask='%+1.5e'),
        Integer(6, "Number of history values", 2),
        # REPEAT fields 7 — 9 for the Number of history values:
        Loop('History', "Number of history values", [
            Float(7, "Sensitivity for calibration", 12, mask='%+1.5e'),
            Float(8, "Frequency of calibration sensitivity", 12,
                  mask='%+1.5e'),
            VariableString(9, "Time of above calibration", 1, 22, 'T')
        ])
    ]

    def get_resp(self, station, channel, abbreviations):
        """
        Returns RESP string.
        """
        # This blockette can result in two different RESPs.
        blkt_type = self.stage_sequence_number
        if blkt_type != 0:
            string = \
                '#\t\t+                  +-------------------------------' + \
                '--------+                  +\n' + \
                '#\t\t+                  |       Channel Gain,' + \
                '%6s ch %s      |                  +\n' % (station, channel) +\
                '#\t\t+                  +-------------------------------' + \
                '--------+                  +\n'
        else:
            string = \
                '#\t\t+                  +--------------------------------' + \
                '-------+                  +\n' + \
                '#\t\t+                  |   Channel Sensitivity,' + \
                '%6s ch %s   |                  +\n' % (station, channel) + \
                '#\t\t+                  +--------------------------------' + \
                '-------+                  +\n'
        string += '#\t\t\n' + \
            'B058F03     Stage sequence number:                 %s\n' \
            % blkt_type
        if blkt_type != 0:
            string += \
                'B058F04     Gain:                                  %s\n' \
                % format_resp(self.sensitivity_gain, 6) + \
                'B058F05     Frequency of gain:                     %s HZ\n' \
                % format_resp(self.frequency, 6)
        else:
            string += \
                'B058F04     Sensitivity:                           %s\n' \
                % format_resp(self.sensitivity_gain, 6) + \
                'B058F05     Frequency of sensitivity:              %s HZ\n' \
                % format_resp(self.frequency, 6)
        string += \
            'B058F06     Number of calibrations:                %s\n' \
            % self.number_of_history_values
        if self.number_of_history_values > 1:
            string += \
                '#\t\tCalibrations:\n' + \
                '#\t\t i, sensitivity, frequency, time of calibration\n'
            for _i in range(self.number_of_history_values):
                string += \
                    'B058F07-08   %2s %13s %13s %s\n' \
                    % (format_resp(self.sensitivity_for_calibration[_i], 6),
                       format_resp(
                           self.frequency_of_calibration_sensitivity[_i], 6),
                       self.time_of_above_calibration[_i].format_seed())
        elif self.number_of_history_values == 1:
            string += \
                '#\t\tCalibrations:\n' + \
                '#\t\t i, sensitivity, frequency, time of calibration\n' + \
                'B058F07-08    0 %13s %13s %s\n' \
                % (format_resp(self.sensitivity_for_calibration, 6),
                   format_resp(self.frequency_of_calibration_sensitivity, 6),
                   self.time_of_above_calibration.format_seed())
        string += '#\t\t\n'
        return string.encode()
