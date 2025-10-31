# -*- coding: utf-8 -*-
from io import UnsupportedOperation
import numpy as np
from obspy.core.util import get_bytes_stream
from .define import SIZE_PSE_RECORD
from .record import PseRecord


class PseTape(object):

    def __init__(self):
        self._record = None
        self._handle = None
        self._handle_has_fileno = False

    def __iter__(self):
        return self

    def __next__(self):
        self._record = self.fromfile()
        if self._record.size < SIZE_PSE_RECORD:
            raise StopIteration()
        return PseRecord(self._record)

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
            # read SIZE_PSE_RECORD bytes
            # (we will parse uint8 -> 1 byte per item):
            buffer = self._handle.read(SIZE_PSE_RECORD)
            ret = np.frombuffer(buffer, dtype=np.uint8)
            # assert len(ret) == SIZE_PSE_RECORD
            return ret
        return np.fromfile(self._handle, dtype=np.uint8,
                           count=SIZE_PSE_RECORD)
