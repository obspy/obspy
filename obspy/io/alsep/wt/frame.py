# -*- coding: utf-8 -*-
import numpy as np

from .parser import \
    parse_wtn_frame_to_alsep_words, parse_wth_frame_to_geophone_data


class _WtFrame(object):

    def __init__(self, frame):
        self.data = frame
        # Frame header parameters
        self.flag_bit = None
        self.msec_of_year = None
        self.alsep_tracking_station_id = None
        self.alsep_package_id = None
        self.bit_search = None
        self.bit_verify = None
        self.bit_confirm = None
        self.bit_lock = None
        self.bit_input_level = None
        self.original_record_number = None
        self.barker_code = None
        self.barker_code_comp = None
        self.frame_count = None
        self.mode_bit = None

    def parse_header(self):
        # a flag bit indicating usage of computer-generated time code
        self.flag_bit = self.data[0] >> 7

        # time of the year in milliseconds
        self.msec_of_year = np.int64(self.data[0] & 0x7f)
        self.msec_of_year = (self.msec_of_year << 8) + self.data[1]
        self.msec_of_year = (self.msec_of_year << 8) + self.data[2]
        self.msec_of_year = (self.msec_of_year << 8) + self.data[3]
        self.msec_of_year = (self.msec_of_year << 4) + (self.data[4] >> 4)

        # ALSEP tracking station ID
        self.alsep_tracking_station_id = self.data[4] & 0x0f

        # ALSEP package ID
        self.alsep_package_id = self.data[5] >> 5

        # bit synchronizer status
        self.bit_search = self.data[5] & 0x0f
        self.bit_search = self.bit_search >> 4
        self.bit_verify = self.data[5] & 0x08
        self.bit_verify = self.bit_verify >> 3
        self.bit_confirm = self.data[5] & 0x04
        self.bit_confirm = self.bit_confirm >> 2
        self.bit_lock = self.data[5] & 0x02
        self.bit_lock = self.bit_lock >> 1
        self.bit_input_level = self.data[5] & 0x01

        # original 7-track record number (first subframe only)
        self.original_record_number = self.data[6]
        self.original_record_number = \
            (self.original_record_number << 8) + self.data[7]

        # sync pattern - Barker code (11100010010) and its complement
        self.barker_code = self.data[8]
        self.barker_code = (self.barker_code << 3) + (self.data[9] >> 5)
        self.barker_code_comp = self.data[9] & 0x0f
        self.barker_code_comp = \
            (self.barker_code_comp << 7) + (self.data[10] >> 1)

    def is_valid(self):
        if self.alsep_package_id < 1 or self.alsep_package_id > 5:
            return False
        return True

    def dump(self):
        print('--- Frame info ---')
        print('flag_bit: {}'.format(self.flag_bit))
        print('msec of year: {:d}'.format(self.msec_of_year))
        print('ALSEP tracking station ID: {}'.format(
            self.alsep_tracking_station_id))
        print('ALSEP Package ID: {}'.format(self.alsep_package_id))
        print('Bit search: {}'.format(self.bit_search))
        print('Bit verify: {}'.format(self.bit_verify))
        print('Bit confirm: {}'.format(self.bit_confirm))
        print('Bit lock: {}'.format(self.bit_lock))
        print('Bit input level: {}'.format(self.bit_input_level))
        print('Original record number: {}'.format(
            self.original_record_number))
        print('sync pattern barker code: {}'.format(self.barker_code))
        print('sync pattern barker code compliment: {}'.format(
            self.barker_code_comp))
        print('mode bit: {}'.format(self.mode_bit))


class WtnFrame(_WtFrame):
    def __init__(self, frame):
        super(WtnFrame, self).__init__(frame)
        # Extracted ALSEP words
        self.alsep_words = None
        # Parse header and data
        self.parse_header()
        self.parse_data()

    def parse_header(self):
        super(WtnFrame, self).parse_header()

        # frame count
        self.frame_count = self.data[11] >> 1

        # mode bit
        #   - 1 for frame 1 means normal bit rate
        #   - 1 for frame 2 means slow bit rate
        #   - ALSEP ID for frames 3-5 as follows
        #       Station 11 ... 011
        #       Station 12 ... 010
        #       Station 14 ... 110
        #       Station 15 ... 011
        #       Station 16 ... 001
        self.mode_bit = self.data[11] & 0x01

    def parse_data(self):
        self.alsep_words = parse_wtn_frame_to_alsep_words(self.data)

    def dump(self):
        super(WtnFrame, self).dump()
        print('ALSEP words: {}'.format(self.alsep_words))


class WthFrame(_WtFrame):
    def __init__(self, frame):
        super(WthFrame, self).__init__(frame)
        # Extracted Geophone data
        self.geophone = None
        # Parse header and data
        self.parse_header()
        self.parse_data()

    def parse_header(self):
        super(WthFrame, self).parse_header()

    def parse_data(self):
        self.geophone = parse_wth_frame_to_geophone_data(self.data)

    def dump(self):
        super(WthFrame, self).dump()
        print('Geophone: {}'.format(self.geophone))
