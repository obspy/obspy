# -*- coding: utf-8 -*-
import numpy as np

from .parser import parse_pse_frame_to_alsep_words


class PseFrame(object):
    def __init__(self, frame, _is_old_format):
        self.data = frame
        self._is_old_format = _is_old_format
        # Frame header parameters
        self.software_time_flag = None
        self.msec_of_year = None
        self.alsep_tracking_station_id = None
        self.bit_error_rate = None
        self.data_rate = None
        self.alsep_word5 = None
        self.barker_code = None
        self.barker_code_comp = None
        self.frame_count = None
        self.mode_bit = None
        # Extracted ALSEP words
        self.alsep_words = None
        # Parse header and data
        self.parse_header()
        self.parse_data()

    def parse_header(self):
        self.software_time_flag = self.data[0] >> 7

        self.msec_of_year = np.int64(self.data[0] & 0x7f)
        self.msec_of_year = (self.msec_of_year << 8) + self.data[1]
        self.msec_of_year = (self.msec_of_year << 8) + self.data[2]
        self.msec_of_year = (self.msec_of_year << 8) + self.data[3]
        self.msec_of_year = (self.msec_of_year << 4) + (self.data[4] >> 4)

        self.alsep_tracking_station_id = self.data[4] & 0x0f
        self.bit_error_rate = (self.data[5] >> 2) & 0x3f
        self.data_rate = (self.data[5] >> 1) & 0x01

        self.alsep_word5 = self.data[6] & 0x03
        self.alsep_word5 = (self.alsep_word5 << 8) | self.data[7]

        self.barker_code = self.data[8]
        self.barker_code = (self.barker_code << 3) + (self.data[9] >> 5)

        self.barker_code_comp = self.data[9] & 0x0f
        self.barker_code_comp = \
            (self.barker_code_comp << 7) + (self.data[10] >> 1)

        self.frame_count = self.data[11] >> 1

        self.mode_bit = self.data[11] & 0x01

    def parse_data(self):
        self.alsep_words = \
            parse_pse_frame_to_alsep_words(self.data, self._is_old_format)
        self.alsep_words[5] = self.alsep_word5

    def dump(self):
        print('--- Frame info ---')
        print('software time flag: {}'.format(self.software_time_flag))
        print('msec of year: {:d}'.format(self.msec_of_year))
        print('ALSEP tracking station ID: {}'.format(
            self.alsep_tracking_station_id))
        print('bit Error Rate: {}'.format(self.bit_error_rate))
        print('data Rate: {}'.format(self.data_rate))
        print('ALSEP word 5: {}'.format(self.alsep_word5))
        print('sync pattern barker code: {}'.format(self.barker_code))
        print('sync pattern barker code compliment: {}'.format(
            self.barker_code_comp))
        print('mode bit: {}'.format(self.mode_bit))
        print('ALSEP words: {}'.format(self.alsep_words))
