# -*- coding: utf-8 -*-
from io import UnsupportedOperation
import numpy as np
from obspy.core.util import get_bytes_stream
from .record import WtnRecord, WthRecord
from .define import SIZE_WT_HEADER


class _WtTape(object):
    def __init__(self):
        self._record = None
        self._handle = None
        self._handle_has_fileno = False

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, file):
        self._handle = get_bytes_stream(file)
        try:
            # Get if fileno is implemented
            # (see self.fromfile)
            self._handle.fileno()
            self._handle_has_fileno = True
        except UnsupportedOperation:
            self._handle_has_fileno = False
        return self

    def close(self):
        self._handle.close()

    def fromfile(self):
        # np.fromfile accepts file path or file-like objects
        # with fileno attr. In all other cases, use np.frombuffer:
        if not self._handle_has_fileno:
            return np.frombuffer(self._handle.read(), dtype=np.uint8)
        return np.fromfile(self._handle, dtype=np.uint8)


class WtnTape(_WtTape):

    def __next__(self):
        self._record = self.fromfile()
        if self._record.size == 0:
            raise StopIteration()
        if np.array_equal(self._record[0:SIZE_WT_HEADER],
                          self._record[SIZE_WT_HEADER:SIZE_WT_HEADER * 2]):
            return WtnRecord(self._record[SIZE_WT_HEADER:])
        return WtnRecord(self._record)


class WthTape(_WtTape):

    def __next__(self):
        self._record = self.fromfile()
        if self._record.size == 0:
            raise StopIteration()
        if np.array_equal(self._record[0:SIZE_WT_HEADER],
                          self._record[SIZE_WT_HEADER:SIZE_WT_HEADER * 2]):
            return WthRecord(self._record[SIZE_WT_HEADER:])
        return WthRecord(self._record)
