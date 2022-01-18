from codecs import encode
import copy
import decorator

import numpy as np
from obspy import Trace


@decorator.decorator
def _open_file(func, *args, **kwargs):
    """
    Ensure a file buffer is passed as first argument to the
    decorated function.

    :param func: callable that takes at least one argument;
        the first argument must be treated as a buffer.
    :return: callable
    """
    first_arg = args[0]
    try:
        with open(first_arg, 'rb') as fi:
            args = tuple([fi] + list(args[1:]))
            return func(*args, **kwargs)
    except TypeError:  # assume we have been passed a buffer
        if not hasattr(args[0], 'read'):
            raise  # type error was in function call, not in opening file
        out = func(*args, **kwargs)
        first_arg.seek(0)  # reset position to start of file
    return out


def _read(fi, position, length, dtype, left_part=True):
    """
    Read one or more bytes using provided datatype.

    This function supports a few datatype options numpy does not support,
    otherwise the arguments are just passed to numpy.

    :param fi: A buffer containing the bytes to read.
    :param position: Byte position to start reading.
    :type position: int
    :param length: Length, in bytes, of data to read.
    :type length: int or float
    :param dtype: bcd, binary, IEEE or any numpy supported datatype.
    :type dtype: str
    :param left_part: If True, start the reading from the first half part
        of the byte position. If False, start the reading from the second
        half part of the byte position.
    :type left_part: bool
    """
    fi.seek(position)
    if dtype == 'bcd':
        return _read_bcd(fi, length, left_part)
    elif dtype == 'binary':
        return _read_binary(fi, length, left_part)
    if dtype == 'IEEE':
        dtype = '>f4'
    # If we get here dtype should be understood by numpy
    data = np.frombuffer(fi.read(int(length)), dtype)
    return data[0] if len(data) == 1 else data


def _read_bcd(fi, length, left_part):
    """
    Interprets a byte string as binary coded decimals.

    See: https://en.wikipedia.org/wiki/Binary-coded_decimal#Basics

    :param fi: A buffer containing the bytes to read.
    :param length: number of bytes to read.
    :type length: int or float
    :param left_part: If True, start the reading from the first half part
        of the first byte. If False, start the reading from
        the second half part of the first byte.
    :type left_part: bool
    """
    tens = np.power(10, range(12))[::-1]
    nbr_half_bytes = round(2 * length)
    if isinstance(length, float):
        length = int(length) + 1
    byte_values = fi.read(length)
    ints = np.frombuffer(byte_values, dtype='<u1', count=length)
    if left_part is True:
        unpack_bits = np.unpackbits(ints).reshape(-1, 4)[0:nbr_half_bytes]
    else:
        unpack_bits = np.unpackbits(ints).reshape(-1, 4)[1:nbr_half_bytes + 1]
    bits = np.dot(unpack_bits, np.array([1, 2, 4, 8])[::-1].reshape(4, 1))
    if np.any(bits > 9):
        raise ValueError('invalid bcd values encountered')
    return np.dot(tens[-len(bits):], bits)[0]


def _read_binary(fi, length, left_part):
    """
    Read raw bytes and convert them in integer.

    :param fi: A buffer containing the bytes to read.
    :param length: number of bytes to read.
    :type length: int or float
    :param left_part: If True, start the reading from the first half part
        of the byte.
    :type left_part: bool
    """
    if isinstance(length, float):
        if abs(length - 0.5) <= 1e-7:
            ints = np.frombuffer(fi.read(1), dtype='<u1')[0]
            if left_part is True:
                return np.bitwise_and(ints >> 4, 0x0f)
            else:
                return np.bitwise_and(ints, 0x0f)
        else:
            raise ValueError('invalid length of bytes to read.\
                             It has to be an integer or 0.5')
    else:
        return int(encode(fi.read(length), 'hex'), 16)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)


def _quick_merge(traces, small_number=.000001):
    """
    Specialized function for merging traces produced by _read_rg16.

    Requires that traces are of the same datatype, have the same
    sampling_rate, and dont have data overlaps.

    :param traces: list of ObsPy :class:`~obspy.core.trace.Trace` objects.
    :param small_number: a small number for determining if traces
        should be merged. Should be much less than one sample spacing.
    :return: list of ObsPy :class:`~obspy.core.trace.Trace` objects.
    """
    # make sure sampling rates are all the same
    assert len({tr.stats.sampling_rate for tr in traces}) == 1
    assert len({tr.data.dtype for tr in traces}) == 1
    sampling_rate = traces[0].stats.sampling_rate
    diff = 1. / sampling_rate + small_number
    # get the array
    ar, trace_ar = _trace_list_to_rec_array(traces)
    # get groups of traces that can be merged together
    group = _get_trace_groups(ar, diff)
    group_numbers = np.unique(group)
    out = [None] * len(group_numbers)  # init output list
    for index, gnum in enumerate(group_numbers):
        trace_ar_to_merge = trace_ar[group == gnum]
        new_data = np.concatenate(list(trace_ar_to_merge['data']))
        # get updated stats object
        new_stats = copy.deepcopy(trace_ar_to_merge['stats'][0])
        new_stats.npts = len(new_data)
        out[index] = Trace(data=new_data, header=new_stats)
    return out


def _trace_list_to_rec_array(traces):
    """
    Return a recarray from the trace list.

    These are separated into two arrays due to a weird issue with
    numpy.sort returning and error set.
    """
    # get the id, starttime, endtime into a recarray
    # rec array column names must be native strings due to numpy issue 2407
    dtype1 = [('id', object), ('starttime', float),
              ('endtime', float)]
    dtype2 = [('data', object), ('stats', object)]
    data1 = [(tr.id, tr.stats.starttime.timestamp, tr.stats.endtime.timestamp)
             for tr in traces]
    data2 = [(tr.data, tr.stats) for tr in traces]
    ar1 = np.array(data1, dtype=dtype1)  # array of id, starttime, endtime
    ar2 = np.array(data2, dtype=dtype2)  # array of data, stats objects
    #
    sort_index = np.argsort(ar1, order=['id', 'starttime'])
    return ar1[sort_index], ar2[sort_index]


def _get_trace_groups(ar, diff):
    """
    Return an array of ints where each element corresponds to a pre-merged
    trace row. All trace rows with the same group number can be merged.
    """
    # get a bool of if ids are the same as the next row down
    ids_different = np.ones(len(ar), dtype=bool)
    ids_different[1:] = ar['id'][1:] != ar['id'][:-1]
    # get bool of endtimes within one sample of starttime of next row
    disjoint = np.zeros(len(ar), dtype=bool)
    start_end_diffs = ar['starttime'][1:] - ar['endtime'][:-1]
    disjoint[:-1] = np.abs(start_end_diffs) <= diff
    # get groups (not disjoint, not different ids)
    return np.cumsum(ids_different & disjoint)
