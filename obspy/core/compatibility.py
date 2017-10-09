# -*- coding: utf-8 -*-
"""
Py3k compatibility module
"""
from future.utils import PY2

import io

import numpy as np

# optional dependencies
try:
    if PY2:
        import mock  # NOQA
    else:
        from unittest import mock  # NOQA
except ImportError:
    pass

if PY2:
    from string import maketrans
    from urlparse import urlparse
else:
    maketrans = bytes.maketrans
    from urllib.parse import urlparse


# NumPy does not offer the from_buffer method under Python 3 and instead
# relies on the built-in memoryview object.
if PY2:
    def from_buffer(data, dtype):
        # For compatibility with NumPy 1.4
        if isinstance(dtype, unicode):  # noqa
            dtype = str(dtype)
        if data:
            return np.frombuffer(data, dtype=dtype).copy()
        else:
            return np.array([], dtype=dtype)
    import ConfigParser as configparser  # NOQA
else:
    def from_buffer(data, dtype):
        return np.array(memoryview(data)).view(dtype).copy()  # NOQA
    import configparser  # NOQA


def is_text_buffer(obj):
    """
    Helper function determining if the passed object is an object that can
    read and write text or not.

    :param obj: The object to be tested.
    :return: True/False
    """
    # Default open()'ed files and StringIO (in Python 2) don't inherit from any
    # of the io classes thus we only test the methods of the objects which
    # in Python 2 should be safe enough.
    if PY2 and not isinstance(obj, io.BufferedIOBase) and \
            not isinstance(obj, io.TextIOBase):
        if hasattr(obj, "read") and hasattr(obj, "write") \
                and hasattr(obj, "seek") and hasattr(obj, "tell"):
            return True
        return False

    return isinstance(obj, io.TextIOBase)


def is_bytes_buffer(obj):
    """
    Helper function determining if the passed object is an object that can
    read and write bytes or not.

    :param obj: The object to be tested.
    :return: True/False
    """
    # Default open()'ed files and StringIO (in Python 2) don't inherit from any
    # of the io classes thus we only test the methods of the objects which
    # in Python 2 should be safe enough.
    if PY2 and not isinstance(obj, io.BufferedIOBase) and \
            not isinstance(obj, io.TextIOBase):
        if hasattr(obj, "read") and hasattr(obj, "write") \
                and hasattr(obj, "seek") and hasattr(obj, "tell"):
            return True
        return False

    return isinstance(obj, io.BufferedIOBase)


def round_away(number):
    """
    Simple function that rounds a number to the nearest integer. If the number
    is halfway between two integers, it will round away from zero. Of course
    only works up machine precision. This should hopefully behave like the
    round() function in Python 2.

    This is potentially desired behavior in the trim functions but some more
    thought should be poured into it.

    The np.round() function rounds towards the even nearest even number in case
    of half-way splits.

    >>> round_away(2.5)
    3
    >>> round_away(-2.5)
    -3

    >>> round_away(10.5)
    11
    >>> round_away(-10.5)
    -11

    >>> round_away(11.0)
    11
    >>> round_away(-11.0)
    -11
    """

    floor = np.floor(number)
    ceil = np.ceil(number)
    if (floor != ceil) and (abs(number - floor) == abs(ceil - number)):
        return int(int(number) + int(np.sign(number)))
    else:
        return int(np.round(number))
