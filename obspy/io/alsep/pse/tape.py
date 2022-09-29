# -*- coding: utf-8 -*-
import numpy as np

from .define import SIZE_PSE_RECORD
from .record import PseRecord


class PseTape(object):

    def __init__(self):
        self._record = None
        self._handle = None

    def __iter__(self):
        return self

    def __next__(self):
        self._record = np.fromfile(self._handle, dtype=np.uint8,
                                   count=SIZE_PSE_RECORD)
        if self._record.size < SIZE_PSE_RECORD:
            raise StopIteration()
        return PseRecord(self._record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self, filename):
        self._handle = open(filename, 'rb')
        return self

    def close(self):
        self._handle.close()
