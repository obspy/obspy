# -*- coding: utf-8 -*-
import numpy as np

from .define import SIZE_RECORD_HEADER, \
    SIZE_DATA_PART_OLD, SIZE_DATA_PART_NEW, \
    NUMBER_OF_FRAMES_OLD, NUMBER_OF_FRAMES_NEW
from .frame import PseFrame


class PseRecord(object):

    def __init__(self, record):
        self.data = record
        # Record header parameters
        self.tape_type = None
        self.apollo_station = None
        self.tape_seq = None
        self.record_number = None
        self.year = None
        self.format = None
        self.phys_records = None
        self.read_err = None
        # Internal parameters
        self._frame_index = 0
        self._frame_size = None
        self._number_of_frames = None
        # Processing
        self.parse_header()

    def __iter__(self):
        return self

    def __next__(self):
        if self._frame_index >= self._number_of_frames:
            raise StopIteration()
        pos_start = \
            SIZE_RECORD_HEADER + self._frame_index * self._frame_size
        pos_end = pos_start + self._frame_size
        frame = PseFrame(self.data[pos_start:pos_end], self.is_old_format())
        self._frame_index += 1
        return frame

    def parse_header(self):
        # 1-2 ... 1 for PSE tapes; 2 for Event tapes
        self.tape_type = np.frombuffer(self.data[0:2], dtype='>u2')[0]

        # 3-4 Apollo station number
        self.apollo_station = np.frombuffer(self.data[2:4], dtype='>u2')[0]

        # 5-6 original tape sequence number for PSE tapes;
        #     2-digit station code plus 3-digit-original event tape
        #     sequence number for Event tapes
        self.tape_seq = np.frombuffer(self.data[4:6], dtype='>u2')[0]

        # 7-8 self.data number
        self.record_number = np.frombuffer(self.data[6:8], dtype='>u2')[0]

        # 9-10 year
        self.year = np.frombuffer(self.data[8:10], dtype='>u2')[0]

        # 11-12 format (0=old, 1=new)
        self.format = np.frombuffer(self.data[10:12], dtype='>u2')[0]

        # 13-14 number of physical records from original tape
        self.phys_records = np.frombuffer(self.data[12:14], dtype='>u2')[0]

        # 15-16 original tape read error flags
        self.read_err = np.frombuffer(self.data[14:16], dtype='>u2')[0]

        if self.format == 0:
            self._frame_size = SIZE_DATA_PART_OLD
            self._number_of_frames = NUMBER_OF_FRAMES_OLD
        else:
            self._frame_size = SIZE_DATA_PART_NEW
            self._number_of_frames = NUMBER_OF_FRAMES_NEW

    def is_old_format(self):
        return True if self.format == 0 else False

    def dump(self):
        print('--- Record info ---')
        tape_type_str = {1: 'PSE tape', 2: 'Event tape'}
        print('Tape: {:s}'.format(tape_type_str[self.tape_type]))
        print('Station number: {:d}'.format(self.apollo_station))
        print('Original tape sequence number: {:d}'.format(self.tape_seq))
        print('Record number: {:d}'.format(self.record_number))
        print('Year: {:d}'.format(self.year))
        format_str = {0: 'Old', 1: 'New'}
        print('Format: {:s}'.format(format_str[self.format]))
        print('Physical records: {:d}'.format(self.phys_records))
        print('Original tape read error flag: {:d}'.format(self.read_err))
