# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np

from obspy import Trace, Stream


DATA_TRACE_HEADER_DTYPE = np.dtype([
    ("trace_no", np.int32),
    ("no_of_samples", np.int32),
    ("i_komp", np.int32),
    ("ensemble_no", np.int32),
    ("cdp_no", np.int32),
    ("shot_no", np.int32),
    ("geophone_no", np.int32),
    ("trace_marker", np.int16),
    ("trace_time", np.int32),
    ("time_del", np.float32),
    ("shot_elevation", np.float64),
    ("distance", np.float64),
    ("shot_ort", (np.float64, 2)),
    ("geophone_ort", (np.float64, 2)),
    ("cdp_ort", (np.float64, 2)),
    ("rec_elevation", np.float64),
    ("trace_gain", np.float32),
    ("timecollect", np.float64),
    ("dummys", (np.float32, 8))])
DATA_TRACE_HEADER_LENGTH = 154


def _read_reflexw_datafile(filename, format_code):
    """
    :type filename: str
    :param filename: Filename of REFLEXW data file.
    :type format_code: int
    :param format_code: Format code of REFLEXW data file
        (`2` for "new 16 bit integer",
         `3` for "new 32 bit floating point").
    """
    if format_code == 2:
        data_dtype = np.int16
    elif format_code == 3:
        data_dtype = np.float32
    else:
        msg = "Unsupported REFLEXW data format code."
        raise NotImplementedError(msg)

    st = Stream()
    with open(filename, "rb") as fh:
        next_header = fh.read(DATA_TRACE_HEADER_LENGTH)
        while next_header != "":
            trace_header = np.fromstring(
                next_header, dtype=DATA_TRACE_HEADER_DTYPE, count=1)[0]
            trace_header = dict(zip(DATA_TRACE_HEADER_DTYPE.names,
                                    trace_header))
            trace_data = np.fromfile(fh, dtype=data_dtype,
                                     count=trace_header["no_of_samples"] + 1)
            tr = Trace(data=trace_data[1:])
            tr.stats.reflexw = trace_header
            st.append(tr)
            next_header = fh.read(DATA_TRACE_HEADER_LENGTH)
    return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
