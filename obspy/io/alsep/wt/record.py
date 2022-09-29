# -*- coding: utf-8 -*-

import numpy as np

from .define import SIZE_WT_HEADER, SIZE_WT_FRAME
from .frame import WtnFrame, WthFrame


class _WtRecord(object):
    def __init__(self, record):
        self.data = record
        # Record header parameters
        self.tape_type = None
        self.active_stations = [-1] * 5
        self.number_of_active_stations = None
        self.original_id = None
        self.year = None
        self.first_msec = None
        # Internal parameters
        self._frame_index = 0
        # Processing
        self.parse_header()
        self._number_of_frames = \
            (record.size - SIZE_WT_HEADER) / SIZE_WT_FRAME

    def parse_header(self):
        # 1-2 ... 3 to identify Normal-Bit-Rate Work tape
        self.tape_type = np.frombuffer(self.data[0:2], dtype='>u2')[0]

        # 3-4 Active Station code
        self.active_stations[0] = self.data[2] & 0x70
        self.active_stations[0] = self.active_stations[0] >> 4

        self.active_stations[1] = self.data[2] & 0x0e
        self.active_stations[1] = self.active_stations[1] >> 1

        self.active_stations[2] = self.data[2] & 0x01
        self.active_stations[2] = \
            (self.active_stations[2] << 2) + (self.data[3] >> 6)

        self.active_stations[3] = self.data[3] & 0x38
        self.active_stations[3] = self.active_stations[3] >> 3

        self.active_stations[4] = self.data[3] & 0x07

        # 5-6 Number of active stations
        self.number_of_active_stations = \
            np.frombuffer(self.data[4:6], dtype='>u2')[0]

        # 7-8 Original 9 track ID (!! Not a tracking station ID)
        self.original_id = np.frombuffer(self.data[6:8], dtype='>u2')[0]

        # 9-10 year
        self.year = np.frombuffer(self.data[8:10], dtype='>u2')[0]

        # 11-14 and first 4bit of byte 15
        # Time of the year of the first data in msec
        self.first_msec = self.data[10]
        self.first_msec = (self.first_msec << 8) + self.data[11]
        self.first_msec = (self.first_msec << 8) + self.data[12]
        self.first_msec = (self.first_msec << 8) + self.data[13]
        self.first_msec = (self.first_msec << 4) + (self.data[14] >> 4)

    def dump(self):
        print('--- Record info ---')
        tape_type_str = {3: 'WTN tape'}
        print('Tape: {}'.format(tape_type_str[self.tape_type]))
        print('Active Stations: {}'.format(self.active_stations))
        print('Number of Active Stations: {}'.format(
            self.number_of_active_stations))
        print('Original 9 track ID: {:d}'.format(self.original_id))
        print('Year: {:d}'.format(self.year))
        print('Time of the year of the first data in msec: {:d}'.format(
            self.first_msec))


class WtnRecord(_WtRecord):

    def __iter__(self):
        return self

    def __next__(self):
        if self._frame_index >= self._number_of_frames:
            raise StopIteration()
        pos_start = SIZE_WT_HEADER + self._frame_index * SIZE_WT_FRAME
        pos_end = pos_start + SIZE_WT_FRAME
        frame = WtnFrame(self.data[pos_start:pos_end])
        self._frame_index += 1
        return frame


class WthRecord(_WtRecord):

    def __iter__(self):
        return self

    def __next__(self):
        if self._frame_index >= self._number_of_frames:
            raise StopIteration()
        pos_start = SIZE_WT_HEADER + self._frame_index * SIZE_WT_FRAME
        pos_end = pos_start + SIZE_WT_FRAME
        frame = WthFrame(self.data[pos_start:pos_end])
        self._frame_index += 1
        return frame
