# -*- coding: utf-8 -*-
"""
ObsPy's compatibility layer.

Includes things to easy dealing with Py2/Py3 differences as well as making
it work with various versions of our dependencies.
"""
from future.utils import PY2

import collections
import importlib
import io
import json
import sys
import unittest

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
    from string import maketrans  # NOQA
    from urlparse import urlparse  # NOQA
else:
    maketrans = bytes.maketrans  # NOQA
    from urllib.parse import urlparse  # NOQA


# Define the string types.
if PY2:
    string_types = (basestring,)  # NOQA
else:
    string_types = (str,)  # NOQA


# Importing the ABCs from collections will no longer work with Python 3.8.
if PY2:
    collections_abc = collections  # NOQA
else:
    try:
        collections_abc = collections.abc  # NOQA
    except AttributeError:
        # Python 3.4 compat, see https://bugs.python.org/msg212284
        # some older Linux distribution (like Debian jessie) are still in LTS,
        # so be nice, this doesn't hurt and can be removed again later on
        collections_abc = importlib.import_module("collections.abc")  # NOQA


if PY2:
    class RegExTestCase(unittest.TestCase):
        def assertRaisesRegex(self, exception, regex, callable,  # NOQA
                              *args, **kwargs):
            return self.assertRaisesRegexp(exception, regex, callable,
                                           *args, **kwargs)
else:
    class RegExTestCase(unittest.TestCase):
        pass


# NumPy does not offer the from_buffer method under Python 3 and instead
# relies on the built-in memoryview object.
if PY2:
    def from_buffer(data, dtype):
        # For compatibility with NumPy 1.4
        if isinstance(dtype, unicode):  # noqa
            dtype = str(dtype)
        if data:
            try:
                data = data.encode()
            except Exception:
                pass
            return np.frombuffer(data, dtype=dtype).copy()
        else:
            return np.array([], dtype=dtype)
else:
    def from_buffer(data, dtype):
        try:
            data = data.encode()
        except Exception:
            pass
        return np.array(memoryview(data)).view(dtype).copy()  # NOQA


if PY2:
    from ConfigParser import SafeConfigParser as ConfigParser  # NOQA
else:
    from configparser import ConfigParser  # NOQA


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


if sys.version_info[0] < 3:
    def py3_round(number, ndigits=None):
        """
        Similar function to python 3's built-in round function.

        Returns a float if ndigits is greater than 0, else returns an integer.

        Note:
        This function should be replace by the builtin round when obspy
        drops support for python 2.7.
        Unlike python'3 rounding, this function always rounds up on half
        increments rather than implementing banker's rounding.

        :type number: int or float
        :param number: A real number to be rounded
        :type ndigits: int
        :param ndigits: number of digits
        :return: An int if ndigites <= 0, else a float rounded to ndigits.
        """
        if ndigits is None or ndigits <= 0:
            mult = 10 ** -(ndigits or 0)
            return ((int(number) + mult // 2) // mult) * mult
        else:
            return round(number, ndigits)
else:
    py3_round = round


def get_json_from_response(r):
    """
    Get a JSON response in a way that also works for very old request
    versions.

    :type r: :class:`requests.Response
    :param r: The server's response.
    """
    if hasattr(r, "json"):
        if isinstance(r.json, dict):
            return r.json
        return r.json()

    c = r.content
    try:
        c = c.decode()
    except Exception:
        pass
    return json.loads(c)


def get_text_from_response(r):
    """
    Get a text response in a way that also works for very old request versions.

    :type r: :class:`requests.Response
    :param r: The server's response.
    """
    if hasattr(r, "text"):
        return r.text

    c = r.content
    try:
        c = c.decode()
    except Exception:
        pass
    return c


def get_reason_from_response(r):
    """
    Get the status text.

    :type r: :class:`requests.Response
    :param r: The server's response.
    """
    # Very old requests version might not have the reason attribute.
    if hasattr(r, "reason"):
        c = r.reason
    else:  # pragma: no cover
        c = r.raw.reason

    if hasattr(c, "encode"):
        c = c.encode()

    return c
