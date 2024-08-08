# -*- coding: utf-8 -*-
"""
ObsPy's compatibility layer.

Includes things to easy dealing with Python version differences as well as
making it work with various versions of our dependencies.
"""
import numpy as np


# NumPy does not offer the from_buffer method under Python 3 and instead
# relies on the built-in memoryview object.
def from_buffer(data, dtype):
    try:
        data = data.encode()
    except Exception:
        pass
    return np.array(memoryview(data)).view(dtype).copy()  # NOQA


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
